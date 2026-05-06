import os
import sys
import argparse
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lora_utils import apply_lora_to_unet, inject_metadata_into_attention
from CTIP.train_ctip import (TabularEncoder, CTIPConditioner,
                              NUM_SPECIES, TABULAR_INPUT_DIM)

MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    required=True)
    parser.add_argument("--out_dir", default="results/ctip_generated")
    parser.add_argument("--n",       type=int, default=100,
                        help="Images per species")
    parser.add_argument("--cfg",     type=float, default=6.0)
    parser.add_argument("--steps",   type=int, default=50)
    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt       = torch.load(args.ckpt, map_location=device, weights_only=False)
    species2idx = ckpt["species2idx"]
    lora_rank   = ckpt.get("lora_rank", 16)

    print("Loading VAE + UNet...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    inject_metadata_into_attention(unet, device)
    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer, state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state)
        layer.to(device)

    tab_enc = TabularEncoder().to(device)
    conditioner = CTIPConditioner(tab_enc).to(device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(args.steps)

    uncond = torch.zeros(1, 77, 768, device=device)

    for species_name, species_idx in tqdm(sorted(species2idx.items(),
                                                  key=lambda x: x[1]),
                                          desc="Species"):
        folder = os.path.join(args.out_dir, species_name.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)

        meta = {
            "species":        torch.tensor([species_idx], device=device),
            "throat_color":   torch.tensor([0], device=device),
            "forehead_color": torch.tensor([0], device=device),
            "belly_color":    torch.tensor([0], device=device),
            "nape_color":     torch.tensor([0], device=device),
        }
        cond = conditioner(meta)

        for i in range(args.n):
            latents = torch.randn(1, 4, 64, 64, device=device)

            with torch.no_grad():
                for t in scheduler.timesteps:
                    set_attention_context(unet, uncond)
                    n_u = unet(latents, t, encoder_hidden_states=uncond).sample
                    set_attention_context(unet, cond)
                    n_c = unet(latents, t, encoder_hidden_states=cond).sample
                    noise_pred = n_u + args.cfg * (n_c - n_u)
                    latents    = scheduler.step(noise_pred, t, latents).prev_sample
                img = vae.decode(latents / 0.18215).sample

            img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")
            Image.fromarray(img).save(os.path.join(folder, f"{i:04d}.png"))


if __name__ == "__main__":
    main()
