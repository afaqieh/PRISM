import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from PlantVillage.metadata_conditioning import MetadataConditionEncoder
from ...lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
CSV_PATH    = "./data/plantvillage_train.csv"
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
    parser.add_argument("--lora_path",      required=True,
                        help="Path to Stage 1 (or Stage 2) checkpoint .pth")
    parser.add_argument("--generated_root", required=True,
                        help="Root directory to save generated images")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading train split CSV from {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run create_split_plantvillage.py first.")
        return

    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    print(f"  Train images available: {len(df)}")
    for cls in sorted(df["class_name"].unique()):
        print(f"    {cls:<55}: {(df['class_name'] == cls).sum()}")

    plant_list     = sorted(df["plant"].unique())
    condition_list = sorted(df["condition"].unique())
    plant2idx      = {p: i for i, p in enumerate(plant_list)}
    condition2idx  = {c: i for i, c in enumerate(condition_list)}
    class_list     = sorted(df["class_name"].unique())
    class2idx      = {c: i for i, c in enumerate(class_list)}

    print(f"\nLoading checkpoint from {args.lora_path}...")
    ckpt          = torch.load(args.lora_path, map_location="cpu", weights_only=False)
    lora_rank     = ckpt.get("lora_rank", 16)
    field_configs = ckpt["field_configs"]

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

    cond_encoder = MetadataConditionEncoder(field_configs, hidden_dim=256, final_dim=768)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.to(device).eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    df_by_class = {
        cls: df[df["class_name"] == cls].reset_index(drop=True)
        for cls in class_list
    }

    print(f"\nMetadata sampled from real train rows per class (training distribution)")
    print(f"Generating M={M_PER_CLASS} images per class (equal budget for all classes)")
    print(f"Guidance scale: {GUIDANCE}")
    print(f"Seed: {SEED} (fully deterministic)")

    for class_name, class_idx in sorted(class2idx.items(), key=lambda x: x[1]):
        out_dir = os.path.join(args.generated_root, class_name)
        os.makedirs(out_dir, exist_ok=True)

        class_rows   = df_by_class[class_name]
        sampled_rows = class_rows.sample(
            n=M_PER_CLASS, replace=True, random_state=SEED
        ).reset_index(drop=True)

        print(f"\n[{class_idx+1}/{len(class2idx)}] {class_name} "
              f"({len(class_rows)} train rows sample {M_PER_CLASS})...")

        for i, row in enumerate(tqdm(sampled_rows.itertuples(),
                                     total=M_PER_CLASS, desc=f"  {class_name}")):
            meta = {
                "plant":     torch.tensor([plant2idx[row.plant]],         device=device),
                "condition": torch.tensor([condition2idx[row.condition]], device=device),
            }
            img = generate_image(
                cond_encoder, unet, vae, scheduler, meta, device, GUIDANCE)
            img.save(os.path.join(out_dir, f"{i:04d}.png"))

        print(f"  Saved {M_PER_CLASS} images {out_dir}")

    print(f"\nAll done. Generated images {args.generated_root}/")
    print(f"Each class has exactly {M_PER_CLASS} images — equal budget for all.")
    print(f"\nNext step:")
    print(f"  python evaluate_metrics_plantvillage.py "
          f"--generated_root {args.generated_root}")


if __name__ == "__main__":
    main()
