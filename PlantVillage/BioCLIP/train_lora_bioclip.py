import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from PlantVillage.dataset_prompt import PlantVillageDatasetPrompt
from PlantVillage.BioCLIP.PromptBuilder_bioclip import create_prompt_bioclip_plantvillage
from ...lora_utils import LoRALinear, apply_lora_to_unet

try:
    import open_clip
except ImportError:
    raise ImportError(
        "open_clip_torch is required for BioCLIP.\n"
        "Install with: pip install open_clip_torch"
    )

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV   = "./data/plantvillage_train.csv"
OUT_DIR     = "results/bioclip_scientific"
STEPS       = 50000
BATCH_SIZE  = 4
LR          = 3e-5
LORA_RANK   = 16
CFG_DROPOUT = 0.15
CKPT_EVERY   = 5000

BIOCLIP_HUB  = "hf-hub:imageomics/bioclip"
BIOCLIP_DIM  = 512
SD_CROSS_DIM = 768
SEQ_LEN      = 77 

class BioCLIPProjection(nn.Module):
    def __init__(self, in_dim=BIOCLIP_DIM, out_dim=SD_CROSS_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.norm(self.proj(x))

def encode_prompts_bioclip(prompts, bioclip_model, tokenizer, projection, device):
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        x = bioclip_model.token_embedding(tokens)
        x = x + bioclip_model.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = bioclip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = bioclip_model.ln_final(x)

    return projection(x)

def save_sample(unet, vae, bioclip_model, tokenizer, projection,
                out_dir, device, step, class2idx, df):
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)

    folder = os.path.join(out_dir, f"samples_step_{step}")
    os.makedirs(folder, exist_ok=True)

    uncond_hidden = encode_prompts_bioclip(
        [""], bioclip_model, tokenizer, projection, device)

    for class_name in sorted(class2idx.keys()):
        row    = df[df["class_name"] == class_name].sample(
            n=1, random_state=step).iloc[0]
        prompt = create_prompt_bioclip_plantvillage(row, inference=True)
        cond_hidden = encode_prompts_bioclip(
            [prompt], bioclip_model, tokenizer, projection, device)

        latents = torch.randn(1, 4, 64, 64, device=device)
        with torch.no_grad():
            for t in scheduler.timesteps:
                n_u = unet(latents, t,
                           encoder_hidden_states=uncond_hidden).sample
                n_c = unet(latents, t,
                           encoder_hidden_states=cond_hidden).sample
                latents = scheduler.step(
                    n_u + 7.5 * (n_c - n_u), t, latents).prev_sample
            img = vae.decode(latents / 0.18215).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(folder, f"{class_name.replace(' ', '_')}.png"))

    print(f"  Samples saved {folder}")

def save_checkpoint(out_dir, step, lora_layers, projection, dataset):

    ckpt_path = os.path.join(out_dir, f"lora_bioclip_plantvillage_step_{step}.pth")

    torch.save(

        {
            "lora_layers":  [l.state_dict() for l in lora_layers],
            "projection":   projection.state_dict(),
            "lora_rank":    LORA_RANK,
            "steps":        step,
            "bioclip_hub":  BIOCLIP_HUB,
            "bioclip_dim":  BIOCLIP_DIM,
            "sd_cross_dim": SD_CROSS_DIM,
            "conditioning": "bioclip_prompt",
            "class2idx":    dataset.class2idx,
        },

        ckpt_path,

    )

    print(f"  Checkpoint saved {ckpt_path}")

def train():
    global BIOCLIP_DIM
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

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

    print(f"Loading BioCLIP from {BIOCLIP_HUB}...")
    bioclip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_HUB)
    bioclip_model       = bioclip_model.to(device).eval()
    for p in bioclip_model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(BIOCLIP_HUB)
    print(f"  BioCLIP loaded — text encoder dim: {BIOCLIP_DIM}")

    with torch.no_grad():
        test_tok = tokenizer(["test"]).to(device)
        test_x   = bioclip_model.token_embedding(test_tok)
        actual_dim = test_x.shape[-1]
    if actual_dim != BIOCLIP_DIM:
        print(f"  WARNING: expected dim {BIOCLIP_DIM}, got {actual_dim}. "
              f"Updating BIOCLIP_DIM automatically.")
        BIOCLIP_DIM = actual_dim

    print(f"Building projection: {BIOCLIP_DIM}D {SD_CROSS_DIM}D...")
    projection = BioCLIPProjection(in_dim=BIOCLIP_DIM, out_dim=SD_CROSS_DIM).to(device)

    print("Building dataset...")
    dataset      = PlantVillageDatasetPrompt(TRAIN_CSV, vae, mode="prompt", device=device,
                                             prompt_fn=create_prompt_bioclip_plantvillage)
    class_counts = dataset.df["class_name"].value_counts().to_dict()
    weights      = dataset.df["class_name"].map(
        lambda x: 1.0 / class_counts[x]).tolist()
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset), replacement=True)
    dataloader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    trainable_params = (
        [p for layer in lora_layers for p in layer.parameters()] +
        list(projection.parameters())
    )
    optimizer = optim.AdamW(trainable_params, lr=LR)

    print(f"\nStarting BioCLIP LoRA training for {STEPS} steps...")
    print(f"  Conditioning : BioCLIP (frozen, {BIOCLIP_DIM}D) + projection ({BIOCLIP_DIM}{SD_CROSS_DIM}D, trainable)")
    print(f"  Trainable    : {sum(p.numel() for p in trainable_params):,} params "
          f"(LoRA + projection)\n")

    step = 0
    while step < STEPS:
        for latents, noise, noisy_latents, t, prompts in dataloader:
            if step >= STEPS:
                break

            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)
            prompts       = list(prompts)

            drop_mask = torch.rand(len(prompts)) < CFG_DROPOUT
            prompts   = ["" if drop_mask[i] else prompts[i]
                         for i in range(len(prompts))]

            encoder_hidden_states = encode_prompts_bioclip(
                prompts, bioclip_model, tokenizer, projection, device)

            noise_pred = unet(noisy_latents, t,
                              encoder_hidden_states=encoder_hidden_states).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{STEPS}  loss={loss.item():.6f}", flush=True)

            if step % 500 == 0:
                save_sample(unet, vae, bioclip_model, tokenizer, projection,
                            OUT_DIR, device, step, dataset.class2idx, dataset.df)
            if step % CKPT_EVERY == 0:
                save_checkpoint(OUT_DIR, step, lora_layers, projection, dataset)

    print("Saving checkpoint...")
    torch.save(
        {
            "lora_layers":  [l.state_dict() for l in lora_layers],
            "projection":   projection.state_dict(),
            "lora_rank":    LORA_RANK,
            "steps":        step,
            "bioclip_hub":  BIOCLIP_HUB,
            "bioclip_dim":  BIOCLIP_DIM,
            "sd_cross_dim": SD_CROSS_DIM,
            "conditioning": "bioclip_prompt",
            "class2idx":    dataset.class2idx,
        },
        os.path.join(OUT_DIR, "lora_bioclip_plantvillage.pth"),
    )
    print(f"Done {OUT_DIR}/lora_bioclip_plantvillage.pth")


if __name__ == "__main__":
    train()
