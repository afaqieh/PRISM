import os
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from metadata_conditioning import MetadataConditionEncoder
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from PIL import Image
from ..lora_utils import LoRALinear, inject_metadata_into_attention

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
CSV_PATH   = "./data/HAM10000_metadata_train.csv"

M_PER_CLASS = 300
DDIM_STEPS  = 50
SEED        = 42

CLASSES = sorted(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])

GUIDANCE_SCALE_PER_CLASS = {
    "nv":    6.0,
    "mel":   6.0,
    "bkl":   6.0,
    "bcc":   6.0,
    "akiec": 6.0,
    "df":    5.0,
    "vasc":  2.0,
}


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context


def apply_lora_to_unet(unet, r=4):
    lora_layers = []
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["to_q", "to_k", "to_v", "to_out"]):
                parent = unet
                parts  = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                orig = getattr(parent, parts[-1])
                lora = LoRALinear(orig, r=r)
                setattr(parent, parts[-1], lora)
                lora_layers.append(lora)
    return lora_layers


def generate_image(cond_encoder, unet, vae, scheduler, metadata, device, guidance_scale):
    cond   = cond_encoder(metadata)
    uncond = torch.zeros(1, 77, 768, device=device)

    latents = torch.randn(1, 4, 64, 64, device=device)

    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="steps", leave=False):
            set_attention_context(unet, uncond)
            noise_uncond = unet(latents, t, encoder_hidden_states=uncond).sample
            set_attention_context(unet, cond)
            noise_cond   = unet(latents, t, encoder_hidden_states=cond).sample
            noise_pred   = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents      = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        image   = vae.decode(latents).sample

    image = (image.clamp(-1, 1) + 1) / 2
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--lora_path",      required=True)
    args = parser.parse_args()
    OUT_ROOT  = args.generated_root
    LORA_PATH = args.lora_path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading train split CSV from {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run create_split.py first to generate the train/val CSV files.")
        return

    df = pd.read_csv(CSV_PATH).dropna(
        subset=["dx", "localization", "sex", "age"]
    ).reset_index(drop=True)
    df = df[df["dx"].isin(CLASSES)].reset_index(drop=True)

    dx2idx   = {k: i for i, k in enumerate(sorted(df["dx"].unique()))}
    site2idx = {k: i for i, k in enumerate(sorted(df["localization"].unique()))}
    sex2idx  = {k: i for i, k in enumerate(sorted(df["sex"].unique()))}

    print(f"  Train images available: {len(df)}")
    for cls in CLASSES:
        print(f"    {cls:<8s}: {(df['dx']==cls).sum()}")

    df_by_class = {cls: df[df["dx"] == cls].reset_index(drop=True) for cls in CLASSES}
    print(f"\nMetadata sampled from real train rows per class (training distribution)")
    print(f"Generating M={M_PER_CLASS} images per class (equal budget for all classes)")
    print(f"Per-class guidance scales: {GUIDANCE_SCALE_PER_CLASS}")

    print("\nLoading VAE & UNet...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_layers = apply_lora_to_unet(unet, r=16)
    for layer in lora_layers:
        layer.to(device)
    inject_metadata_into_attention(unet, device)

    ckpt  = torch.load(LORA_PATH, map_location="cpu", weights_only=False)
    state = ckpt["cond_encoder"]
    num_dx    = state["embeddings.dx.weight"].shape[0]
    num_sites = state["embeddings.site.weight"].shape[0]
    num_sex   = state["embeddings.sex.weight"].shape[0]

    cond_encoder = MetadataConditionEncoder.for_ham10000(
        num_dx=num_dx, num_sites=num_sites, num_sex=num_sex,
        hidden_dim=256, final_dim=768
    )
    cond_encoder.load_state_dict(state)
    cond_encoder.to(device).eval()

    for layer, layer_state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(layer_state)

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    for dx_name in CLASSES:
        class_out_dir = os.path.join(OUT_ROOT, dx_name)
        os.makedirs(class_out_dir, exist_ok=True)

        guidance    = GUIDANCE_SCALE_PER_CLASS[dx_name]
        class_rows  = df_by_class[dx_name]
        sampled_rows = class_rows.sample(
            n=M_PER_CLASS, replace=True, random_state=SEED
        ).reset_index(drop=True)

        print(f"\nGenerating {M_PER_CLASS} images for '{dx_name}' "
              f"(train-distribution metadata, guidance={guidance})...")

        for i, row in enumerate(tqdm(sampled_rows.itertuples(), 
                                     total=M_PER_CLASS, desc=f"[{dx_name}]")):
            meta = {
                "dx":   torch.tensor([dx2idx[dx_name]], device=device),
                "site": torch.tensor([site2idx[row.localization]], device=device),
                "sex":  torch.tensor([sex2idx[row.sex]], device=device),
                "age":  torch.tensor(
                    [float(row.age) / 100.0], dtype=torch.float32, device=device
                ),
            }

            img = generate_image(
                cond_encoder, unet, vae, scheduler, meta,
                device=device, guidance_scale=guidance
            )
            img.save(os.path.join(class_out_dir, f"{i:04d}.png"))

        print(f"  Saved {M_PER_CLASS} images to {class_out_dir}")

    print(f"\nAll classes done.")
    print(f"Generated images saved to: {OUT_ROOT}/")
    print(f"Each class has exactly {M_PER_CLASS} images — equal budget for all.")


if __name__ == "__main__":
    main()