import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset_cub import CUBDataset
from ..lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME   = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV    = "./data/cub_train.csv"
OUT_DIR      = "results/cub_class_embed"
STEPS        = 20000
BATCH_SIZE   = 4
LR           = 3e-5
LORA_RANK    = 16
CFG_DROPOUT  = 0.15
SEQ_LEN      = 77
HIDDEN_DIM   = 768


class ClassEmbeddingEncoder(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 768, seq_len: int = 77):
        super().__init__()
        self.seq_len    = seq_len
        self.embedding  = nn.Embedding(num_classes, hidden_dim)
        self.norm       = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, meta: dict) -> torch.Tensor:
        species_idx = meta["species"]
        emb = self.norm(self.embedding(species_idx))
        emb = emb.unsqueeze(1).expand(-1, self.seq_len, -1)
        return emb


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context

def save_sample(unet, vae, cond_encoder, out_dir, device, step, species2idx):
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)
    uncond = torch.zeros(1, SEQ_LEN, HIDDEN_DIM, device=device)

    folder = os.path.join(out_dir, f"samples_step_{step}")
    os.makedirs(folder, exist_ok=True)

    for species_name, idx in sorted(species2idx.items()):
        meta    = {"species": torch.tensor([idx], device=device)}
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
        Image.fromarray(img).save(
            os.path.join(folder, f"{species_name.replace(' ', '_')}.png"))

    print(f"  Samples saved {folder}")

def train():
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

    print("Building dataset...")
    dataset = CUBDataset(TRAIN_CSV, vae, device=device)

    class_counts = dataset.df["species_name"].value_counts().to_dict()
    weights      = dataset.df["species_name"].map(lambda x: 1.0 / class_counts[x]).tolist()
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(dataset), replacement=True)
    dataloader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    num_classes  = len(dataset.species2idx)
    print(f"  {num_classes} species, {len(dataset)} training images")

    print("Building class embedding encoder...")
    cond_encoder = ClassEmbeddingEncoder(
        num_classes=num_classes, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN
    ).to(device)
    n_params = sum(p.numel() for p in cond_encoder.parameters())
    print(f"  ClassEmbeddingEncoder params: {n_params:,}  "
          f"({num_classes} classes × {HIDDEN_DIM}D + LayerNorm)")

    print("Injecting metadata into attention...")
    inject_metadata_into_attention(unet, device)

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for layer in lora_layers:
        layer.to(device)

    scheduler_ddpm = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) +
        [p for layer in lora_layers for p in layer.parameters()],
        lr=LR
    )

    print(f"Training for {STEPS} steps  (batch={BATCH_SIZE}, lr={LR})...")
    step = 0
    while step < STEPS:
        for latents, noise, noisy_latents, t, meta in dataloader:
            if step >= STEPS:
                break

            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)

            meta_batch = {"species": meta["species"].to(device)}

            cond = cond_encoder(meta_batch)

            dropout_mask = (torch.rand(cond.shape[0], 1, 1, device=device) < CFG_DROPOUT)
            cond = torch.where(dropout_mask, torch.zeros_like(cond), cond)

            set_attention_context(unet, cond)
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=cond).sample
            loss       = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{STEPS}  loss={loss.item():.6f}")

            if step % 500 == 0:
                save_sample(unet, vae, cond_encoder, OUT_DIR, device, step,
                            dataset.species2idx)

    print("Saving checkpoint...")
    torch.save(
        {
            "lora_layers":  [l.state_dict() for l in lora_layers],
            "cond_encoder": cond_encoder.state_dict(),
            "species2idx":  dataset.species2idx,
            "num_classes":  num_classes,
            "lora_rank":    LORA_RANK,
        },
        os.path.join(OUT_DIR, "lora_class_embed_cub.pth"),
    )
    print(f"Done {OUT_DIR}/lora_class_embed_cub.pth")


if __name__ == "__main__":
    train()
