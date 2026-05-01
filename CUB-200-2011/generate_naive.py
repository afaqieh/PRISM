import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from ..lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

from train_naive_architecture import (
    ClassEmbeddingEncoder,
    set_attention_context
)

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
M_PER_CLASS = 100
DDIM_STEPS  = 50
GUIDANCE    = 6.0
SEED        = 42
SEQ_LEN     = 77
HIDDEN_DIM  = 768

def load_checkpoint(lora_path: str, device: str):
    ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_rank   = ckpt.get("lora_rank", 16)
    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer, sd in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(sd)
    for layer in lora_layers:
        layer.to(device)

    inject_metadata_into_attention(unet, device)

    num_classes  = ckpt["num_classes"]
    species2idx  = ckpt["species2idx"]
    cond_encoder = ClassEmbeddingEncoder(
        num_classes=num_classes, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.to(device).eval()

    return unet, cond_encoder, species2idx


@torch.no_grad()
def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    print(f"Loading checkpoint: {args.lora_path}")
    unet, cond_encoder, species2idx = load_checkpoint(args.lora_path, device)
    unet.eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    uncond = torch.zeros(1, SEQ_LEN, HIDDEN_DIM, device=device)

    for species_name, class_idx in tqdm(sorted(species2idx.items()),
                                        desc="Generating per species"):
        out_dir = os.path.join(args.out_root, species_name.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)

        meta = {"species": torch.tensor([class_idx], device=device)}
        cond = cond_encoder(meta)

        for i in range(args.m_per_class):
            latents = torch.randn(1, 4, 64, 64, device=device,
                                  generator=torch.Generator(device).manual_seed(SEED + i))

            for t in scheduler.timesteps:
                set_attention_context(unet, uncond)
                n_u = unet(latents, t, encoder_hidden_states=uncond).sample
                set_attention_context(unet, cond)
                n_c = unet(latents, t, encoder_hidden_states=cond).sample
                guided = n_u + GUIDANCE * (n_c - n_u)
                latents = scheduler.step(guided, t, latents).prev_sample

            img = vae.decode(latents / 0.18215).sample
            img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")
            Image.fromarray(img).save(os.path.join(out_dir, f"{i:04d}.png"))

    print(f"\nDone. Images saved to {args.out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_path",   required=True,
                    help="Path to lora_class_embed_cub.pth")
    ap.add_argument("--out_root",    required=True,
                    help="Root directory for generated images")
    ap.add_argument("--m_per_class", type=int, default=M_PER_CLASS,
                    help=f"Images per species (default: {M_PER_CLASS})")
    generate(ap.parse_args())
