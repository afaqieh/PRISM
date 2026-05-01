import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PromptBuilder_cub import create_prompt_cub
from ...lora_utils import LoRALinear, apply_lora_to_unet

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
CSV_PATH    = "./data/cub_train.csv"
M_PER_CLASS = 100
DDIM_STEPS  = 50
SEED        = 42
GUIDANCE    = 6.0


def encode_prompt(prompt, tokenizer, text_encoder, device):
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]
    return hidden


def generate_image(unet, vae, scheduler, cond_hidden, uncond_hidden, device, guidance):
    latents = torch.randn(1, 4, 64, 64, device=device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            n_u = unet(latents, t, encoder_hidden_states=uncond_hidden).sample
            n_c = unet(latents, t, encoder_hidden_states=cond_hidden).sample
        latents = scheduler.step(
            n_u + guidance * (n_c - n_u), t, latents
        ).prev_sample
    with torch.no_grad():
        img = vae.decode(latents / 0.18215).sample
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype("uint8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path",      required=True,
                        help="Path to CLIP LoRA checkpoint (lora_prompt_cub.pth)")
    parser.add_argument("--generated_root", required=True,
                        help="Output root directory for generated images")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.generated_root, exist_ok=True)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)

    print("Loading CLIP tokenizer + text encoder...")
    tokenizer    = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").to(device)
    text_encoder.eval()

    print(f"Loading LoRA checkpoint: {args.lora_path}")
    ckpt        = torch.load(args.lora_path, map_location=device, weights_only=False)
    lora_rank   = ckpt["lora_rank"]
    species2idx = ckpt["species2idx"]
    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer in lora_layers:
        layer.to(device)
    for layer, state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state)
    unet.eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    uncond_hidden = encode_prompt("", tokenizer, text_encoder, device)

    df = pd.read_csv(CSV_PATH)
    df_by_species = {
        name: df[df["species_name"] == name].reset_index(drop=True)
        for name in species2idx.keys()
    }

    print(f"\nMetadata sampled from real train rows per species (same as metadata version)")
    print(f"Generating {M_PER_CLASS} images per species "
          f"({len(species2idx)} species, {M_PER_CLASS * len(species2idx)} total)\n")

    for species_name, species_idx in sorted(species2idx.items(), key=lambda x: x[1]):
        out_folder = os.path.join(
            args.generated_root, species_name.replace(" ", "_"))
        os.makedirs(out_folder, exist_ok=True)

        class_rows   = df_by_species[species_name]
        sampled_rows = class_rows.sample(
            n=M_PER_CLASS, replace=True, random_state=SEED
        ).reset_index(drop=True)

        print(f"[{species_name}] ({len(class_rows)} train rows sample {M_PER_CLASS})")

        for i, row in tqdm(sampled_rows.iterrows(),
                           total=M_PER_CLASS, desc=f"  {species_name}"):
            prompt      = create_prompt_cub(row, inference=True)
            cond_hidden = encode_prompt(prompt, tokenizer, text_encoder, device)

            img = generate_image(
                unet, vae, scheduler, cond_hidden, uncond_hidden, device, GUIDANCE)
            Image.fromarray(img).save(
                os.path.join(out_folder, f"{i:04d}.png"))

    print(f"\nDone. Images saved to: {args.generated_root}")


if __name__ == "__main__":
    main()
