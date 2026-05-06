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
from metadata_conditioning import MetadataConditionEncoder
from ..lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
CSV_PATH    = "./data/awa2_train.csv"
M_PER_CLASS = 100
DDIM_STEPS  = 50
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
    parser.add_argument("--lora_path",      required=True)
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading train CSV from {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run prepare_awa2_csv.py first."
        )
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)
    print(f"  {len(df)} train images, {df['class_name'].nunique()} classes")

    print(f"Loading checkpoint from {args.lora_path} ...")
    ckpt          = torch.load(args.lora_path, map_location="cpu", weights_only=False)
    class2idx     = ckpt["class2idx"]
    attr_names    = ckpt["attr_names"]
    field_configs = ckpt["field_configs"]
    lora_rank     = ckpt.get("lora_rank", 16)

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

    class_attrs = (
        df.groupby("class_name")[attr_names].first().to_dict(orient="index")
    )

    print(f"\nGenerating {M_PER_CLASS} images per class | guidance={GUIDANCE}")

    for class_name, class_idx in sorted(class2idx.items(), key=lambda x: x[1]):
        out_dir = os.path.join(args.generated_root, class_name.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)

        if class_name not in class_attrs:
            print(f"  WARNING: {class_name} not found in CSV, skipping.")
            continue

        attrs = class_attrs[class_name]
        meta  = {"class": torch.tensor([class_idx], device=device)}
        for attr in attr_names:
            meta[attr] = torch.tensor([int(attrs[attr])], device=device)

        print(f"\n[{class_idx+1}/{len(class2idx)}] {class_name} | "
              f"attrs={[int(attrs[a]) for a in attr_names]}")

        for i in tqdm(range(M_PER_CLASS), desc=f"  {class_name}"):
            torch.manual_seed(args.seed * 10000 + i)
            img = generate_image(cond_encoder, unet, vae, scheduler, meta, device, GUIDANCE)
            img.save(os.path.join(out_dir, f"{i:04d}.png"))


if __name__ == "__main__":
    main()
