import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from metadata_conditioning import MetadataConditionEncoder
from dataset_cub import NUM_THROAT_COLORS, NUM_FOREHEAD_COLORS, NUM_BELLY_COLORS, NUM_NAPE_COLORS
from ...lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
CSV_PATH    = "./data/cub_train.csv"
M_PER_CLASS = 100
DDIM_STEPS  = 50
SEED        = 42
GUIDANCE    = 6.0

def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context


def generate_image(cond_encoder, unet, vae, scheduler, metadata, device, guidance):
    cond    = cond_encoder(metadata)
    uncond  = torch.zeros(1, 77, 768, device=device)
    latents = torch.randn(1, 4, 64, 64, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            set_attention_context(unet, uncond)
            n_u = unet(latents, t, encoder_hidden_states=uncond).sample
            set_attention_context(unet, cond)
            n_c = unet(latents, t, encoder_hidden_states=cond).sample
            latents = scheduler.step(
                n_u + guidance * (n_c - n_u), t, latents
            ).prev_sample

        latents = latents / 0.18215
        image   = vae.decode(latents).sample

    image = (image.clamp(-1, 1) + 1) / 2
    image = image[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((image * 255).astype("uint8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path",       required=True)
    parser.add_argument("--generated_root",  required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading train split CSV from {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run create_split_cub.py first to generate the train/val CSV files.")
        return

    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    print(f"  Train images available: {len(df)}")
    for species in sorted(df["species_name"].unique()):
        print(f"    {species:<35}: {(df['species_name'] == species).sum()}")

    print(f"\nLoading checkpoint from {args.lora_path}...")
    ckpt          = torch.load(args.lora_path, map_location="cpu", weights_only=False)
    species2idx   = ckpt["species2idx"]
    lora_rank     = ckpt.get("lora_rank", 16)
    field_configs = ckpt["field_configs"]

    print(f"  Species: {sorted(species2idx.keys())}")

    print("Loading VAE & UNet...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer in lora_layers:
        layer.to(device)
    inject_metadata_into_attention(unet, device)

    for layer, state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state)

    cond_encoder = MetadataConditionEncoder(
        field_configs, hidden_dim=256, final_dim=768,
    )
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.to(device).eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    df_by_species = {
        name: df[df["species_name"] == name].reset_index(drop=True)
        for name in species2idx.keys()
    }

    print(f"\nMetadata sampled from real train rows per species (training distribution)")
    print(f"Generating M={M_PER_CLASS} images per species (equal budget for all classes)")
    print(f"Guidance scale: {GUIDANCE}")

    for species_name, species_idx in sorted(species2idx.items(), key=lambda x: x[1]):
        out_dir = os.path.join(args.generated_root, species_name.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)

        class_rows   = df_by_species[species_name]
        sampled_rows = class_rows.sample(
            n=M_PER_CLASS, replace=True, random_state=SEED
        ).reset_index(drop=True)

        print(f"\n[{species_idx+1}/{len(species2idx)}] {species_name} "
              f"({len(class_rows)} train rows sample {M_PER_CLASS})...")

        for i, row in enumerate(tqdm(sampled_rows.itertuples(),
                                     total=M_PER_CLASS, desc=f"  {species_name}")):
            meta = {
                "species":        torch.tensor([species_idx],               device=device),
                "throat_color":   torch.tensor([int(row.throat_color)],     device=device),
                "forehead_color": torch.tensor([int(row.forehead_color)],   device=device),
                "belly_color":    torch.tensor([int(row.belly_color)],      device=device),
                "nape_color":     torch.tensor([int(row.nape_color)],       device=device),
            }
            img = generate_image(
                cond_encoder, unet, vae, scheduler, meta, device, GUIDANCE)
            img.save(os.path.join(out_dir, f"{i:04d}.png"))

        print(f"  Saved {M_PER_CLASS} images {out_dir}")

    print(f"\nAll done. Generated images {args.generated_root}/")
    print(f"Each species has exactly {M_PER_CLASS} images — equal budget for all.")
    print(f"\nNext step:")
    print(f"  python downstream_fewshot_cub.py "
          f"--generated_root {args.generated_root} "
          f"--output results/fewshot_cub")


if __name__ == "__main__":
    main()
