import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler

from PlantVillage.dataset import PlantVillageDataset
from PlantVillage.metadata_conditioning import MetadataConditionEncoder, plantvillage_field_configs
from ...lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV  = "./data/plantvillage_train.csv"
SEED       = 42
OUT_DIR    = f"results/plantvillage_stage1/seed_{SEED}"

STEPS       = 30000
BATCH_SIZE  = 4
LR          = 3e-5
LORA_RANK   = 16
CFG_DROPOUT = 0.15

CKPT_30K = 30000

NUM_WORKERS = 0

CONDITION_MAP = {
    "Pepper__bell___Bacterial_spot":               ("Pepper", "Bacterial_spot"),
    "Pepper__bell___healthy":                      ("Pepper", "healthy"),
    "Potato___Early_blight":                       ("Potato", "Early_blight"),
    "Potato___Late_blight":                        ("Potato", "Late_blight"),
    "Potato___healthy":                            ("Potato", "healthy"),
    "Tomato_Bacterial_spot":                       ("Tomato", "Bacterial_spot"),
    "Tomato_Early_blight":                         ("Tomato", "Early_blight"),
    "Tomato_Late_blight":                          ("Tomato", "Late_blight"),
    "Tomato_Leaf_Mold":                            ("Tomato", "Leaf_Mold"),
    "Tomato_Septoria_leaf_spot":                   ("Tomato", "Septoria_leaf_spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider_mites"),
    "Tomato__Target_Spot":                         ("Tomato", "Target_Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       ("Tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato__Tomato_mosaic_virus":                 ("Tomato", "mosaic_virus"),
    "Tomato_healthy":                              ("Tomato", "healthy"),
}

os.environ["PYTHONHASHSEED"]        = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic        = True
    torch.backends.cudnn.benchmark            = False
    torch.backends.cuda.matmul.allow_tf32     = False
    torch.backends.cudnn.allow_tf32           = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context

def save_sample(unet, vae, cond_encoder, out_dir, device, step,
                class2idx, plant2idx, condition2idx):
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)
    uncond = torch.zeros(1, 77, 768, device=device)

    folder = os.path.join(out_dir, "samples_by_class", f"step_{step}")
    os.makedirs(folder, exist_ok=True)

    for class_name in sorted(class2idx.keys()):
        plant_name, cond_name = CONDITION_MAP[class_name]
        meta = {
            "plant":     torch.tensor([plant2idx[plant_name]],     device=device),
            "condition": torch.tensor([condition2idx[cond_name]],  device=device),
        }
        cond    = cond_encoder(meta)
        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                set_attention_context(unet, uncond)
                n_u = unet(latents, t, encoder_hidden_states=uncond).sample
                set_attention_context(unet, cond)
                n_c = unet(latents, t, encoder_hidden_states=cond).sample
                latents = scheduler.step(
                    n_u + 7.5 * (n_c - n_u), t, latents
                ).prev_sample
            img = vae.decode(latents / 0.18215).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")
        Image.fromarray(img).save(os.path.join(folder, f"{class_name}.png"))

    print(f"  Samples saved {folder}")


def save_checkpoint(lora_layers, cond_encoder, dataset, field_configs, step, path):
    torch.save({
        "lora_layers":    [l.state_dict() for l in lora_layers],
        "cond_encoder":   cond_encoder.state_dict(),
        "field_configs":  field_configs,
        "class2idx":      dataset.class2idx,
        "plant2idx":      dataset.plant2idx,
        "condition2idx":  dataset.condition2idx,
        "lora_rank":      LORA_RANK,
        "step":           step,
        "seed":           SEED,
    }, path)
    print(f"  Checkpoint saved {path}")

def train():
    seed_everything(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Seed: {SEED}  |  Output: {OUT_DIR}")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for layer in lora_layers:
        layer.to(device)
    inject_metadata_into_attention(unet, device)

    print("Building dataset...")
    dataset = PlantVillageDataset(TRAIN_CSV, vae, device=device)
    print(f"  {len(dataset)} images, {len(dataset.class2idx)} classes")
    for cls, idx in sorted(dataset.class2idx.items(), key=lambda x: x[1]):
        print(f"    [{idx}] {cls}: {(dataset.df['class_name'] == cls).sum()} images")

    class_counts = dataset.df["class_name"].value_counts().to_dict()
    weights      = dataset.df["class_name"].map(lambda x: 1.0 / class_counts[x]).tolist()

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(SEED)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    sampler    = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=sampler_generator,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=False,
    )

    print("Building metadata encoder...")
    field_configs = plantvillage_field_configs(dataset)
    cond_encoder  = MetadataConditionEncoder(
        field_configs, hidden_dim=256, final_dim=768).to(device)

    scheduler_ddpm = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) +
        [p for layer in lora_layers for p in layer.parameters()],
        lr=LR,
    )

    print(f"\nTraining {STEPS} steps  (checkpoint at {CKPT_30K}, final at {STEPS})...")
    step = 0
    pbar = tqdm(total=STEPS, desc="Stage 1 PlantVillage")

    while step < STEPS:
        for latents, noise, noisy_latents, t, meta in dataloader:
            if step >= STEPS:
                break

            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)
            meta_batch    = {k: v.to(device) for k, v in meta.items()}

            cond         = cond_encoder(meta_batch)
            dropout_mask = (torch.rand(cond.shape[0], 1, 1, device=device) < CFG_DROPOUT)
            cond         = torch.where(dropout_mask, torch.zeros_like(cond), cond)

            set_attention_context(unet, cond)
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=cond).sample
            loss       = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % 50 == 0:
                print(f"Step {step}/{STEPS}  loss={loss.item():.6f}", flush=True)

            if step % 2000 == 0:
                save_sample(unet, vae, cond_encoder, OUT_DIR, device, step,
                            dataset.class2idx, dataset.plant2idx, dataset.condition2idx)

            if step == CKPT_30K:
                ckpt_path = os.path.join(
                    OUT_DIR, f"lora_plantvillage_seed_{SEED}_step{CKPT_30K}.pth")
                save_checkpoint(lora_layers, cond_encoder, dataset,
                                field_configs, step, ckpt_path)

    pbar.close()
    final_path = os.path.join(OUT_DIR, f"lora_plantvillage_seed_{SEED}.pth")
    save_checkpoint(lora_layers, cond_encoder, dataset, field_configs, step, final_path)
    print(f"\nDone. Final weights {final_path}")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU")
    train()
