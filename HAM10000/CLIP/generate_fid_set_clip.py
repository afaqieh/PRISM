import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from PromptBuilder import create_prompt
from ...lora_utils import LoRALinear, apply_lora_to_unet

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
CSV_PATH    = "../data/HAM10000_metadata_train.csv" 

M_PER_CLASS = 100   
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

def encode_prompt(prompt, tokenizer, text_encoder, device):
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device)
        )[0]  # (1, 77, 768)
    return encoder_hidden_states


def encode_uncond(tokenizer, text_encoder, device):
    return encode_prompt("", tokenizer, text_encoder, device)


def generate_image(unet, vae, scheduler, cond_emb, uncond_emb,
                   device, guidance_scale):
    latents = torch.randn(1, 4, 64, 64, device=device)

    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="steps", leave=False):
            noise_uncond = unet(
                latents, t, encoder_hidden_states=uncond_emb
            ).sample
            noise_cond = unet(
                latents, t, encoder_hidden_states=cond_emb
            ).sample
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents    = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        image   = vae.decode(latents).sample

    image = (image.clamp(-1, 1) + 1) / 2
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path",      required=True,
                        help="Path to trained CLIP LoRA checkpoint (.pth)")
    parser.add_argument("--generated_root", required=True,
                        help="Output folder for generated images")
    args = parser.parse_args()

    LORA_PATH = args.lora_path
    OUT_ROOT  = args.generated_root

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading train split CSV from {CSV_PATH} ...")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run create_split.py first.")
        return

    df = pd.read_csv(CSV_PATH).dropna(
        subset=["dx", "localization", "sex", "age"]
    ).reset_index(drop=True)
    df = df[df["dx"].isin(CLASSES)].reset_index(drop=True)

    print(f"  Train images available: {len(df)}")
    for cls in CLASSES:
        print(f"    {cls:<8s}: {(df['dx']==cls).sum()}")

    df_by_class = {cls: df[df["dx"] == cls].reset_index(drop=True)
                   for cls in CLASSES}

    print(f"\nGenerating M={M_PER_CLASS} images per class (equal budget)")
    print(f"Metadata sampled from real train rows per class")
    print(f"Per-class guidance scales: {GUIDANCE_SCALE_PER_CLASS}")

    print("\nLoading VAE & UNet...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME, subfolder="unet"
    ).to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_layers = apply_lora_to_unet(unet, r=16)
    for layer in lora_layers:
        layer.to(device)

    print(f"Loading LoRA checkpoint from {LORA_PATH} ...")
    ckpt = torch.load(LORA_PATH, map_location="cpu", weights_only=False)
    for layer, layer_state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(layer_state)
    print(f"  LoRA rank: {ckpt.get('lora_rank', 16)}")

    unet.eval()

    print("Loading CLIP tokenizer and text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder"
    ).to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    uncond_emb = encode_uncond(tokenizer, text_encoder, device)

    for dx_name in CLASSES:
        class_out_dir = os.path.join(OUT_ROOT, dx_name)
        os.makedirs(class_out_dir, exist_ok=True)

        guidance    = GUIDANCE_SCALE_PER_CLASS[dx_name]
        class_rows  = df_by_class[dx_name]
        sampled_rows = class_rows.sample(
            n=M_PER_CLASS, replace=True, random_state=SEED
        ).reset_index(drop=True)

        print(f"\nGenerating {M_PER_CLASS} images for '{dx_name}' "
              f"(CLIP text conditioning, guidance={guidance})...")

        for i, row in enumerate(tqdm(
                sampled_rows.itertuples(), total=M_PER_CLASS, desc=f"[{dx_name}]")):

            prompt = create_prompt('ham', row._asdict(), inference=True)
            cond_emb = encode_prompt(prompt, tokenizer, text_encoder, device)

            img = generate_image(
                unet, vae, scheduler, cond_emb, uncond_emb,
                device=device, guidance_scale=guidance
            )
            img.save(os.path.join(class_out_dir, f"{i:04d}.png"))

        print(f"  Saved {M_PER_CLASS} images to {class_out_dir}")

    print(f"\nAll classes done.")
    print(f"Generated images saved to: {OUT_ROOT}/")
    print(f"Each class has exactly {M_PER_CLASS} images — equal budget for all.")
    print(f"\nNext steps:")
    print(f"  python Evaluate_Metrics.py --generated_root {OUT_ROOT} --dataset ham10000")
    print(f"  python downstream_fewshot_clean.py --generated_root {OUT_ROOT}")


if __name__ == "__main__":
    main()
