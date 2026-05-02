import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
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
CSV_PATH    = "./data/plantvillage_train.csv"
M_PER_CLASS = 100
DDIM_STEPS  = 50
SEED        = 42
GUIDANCE    = 6.0

BIOCLIP_HUB  = "hf-hub:imageomics/bioclip"
BIOCLIP_DIM  = 512
SD_CROSS_DIM = 768

class BioCLIPProjection(nn.Module):
    def __init__(self, in_dim=BIOCLIP_DIM, out_dim=SD_CROSS_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.norm(self.proj(x))


def encode_prompt_bioclip(prompt, bioclip_model, tokenizer, projection, device):
    tokens = tokenizer([prompt]).to(device)
    with torch.no_grad():
        x = bioclip_model.token_embedding(tokens)
        x = x + bioclip_model.positional_embedding.to(x.dtype)
        x = x.permute(1, 0, 2)
        x = bioclip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = bioclip_model.ln_final(x)
        hidden = projection(x)
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
                        help="Path to BioCLIP LoRA checkpoint (lora_bioclip_plantvillage.pth)")
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

    print(f"Loading BioCLIP from {BIOCLIP_HUB}...")
    bioclip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_HUB)
    bioclip_model = bioclip_model.to(device).eval()
    for p in bioclip_model.parameters():
        p.requires_grad = False
    tokenizer = open_clip.get_tokenizer(BIOCLIP_HUB)

    global BIOCLIP_DIM
    actual_dim = bioclip_model.token_embedding.embedding_dim
    if actual_dim != BIOCLIP_DIM:
        print(f"  WARNING: BioCLIP embedding dim is {actual_dim}, expected {BIOCLIP_DIM}. Updating.")
        BIOCLIP_DIM = actual_dim

    print(f"Loading LoRA checkpoint: {args.lora_path}")
    ckpt       = torch.load(args.lora_path, map_location=device, weights_only=False)
    lora_rank  = ckpt["lora_rank"]
    class2idx  = ckpt["class2idx"]

    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer in lora_layers:
        layer.to(device)
    for layer, state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state)
    unet.eval()

    projection = BioCLIPProjection(in_dim=BIOCLIP_DIM, out_dim=SD_CROSS_DIM).to(device)
    projection.load_state_dict(ckpt["projection"])
    projection.eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    uncond_hidden = encode_prompt_bioclip("", bioclip_model, tokenizer, projection, device)

    df = pd.read_csv(CSV_PATH)
    df_by_class = {
        name: df[df["class_name"] == name].reset_index(drop=True)
        for name in class2idx.keys()
    }

    print(f"\nGenerating {M_PER_CLASS} images per class "
          f"({len(class2idx)} classes, {M_PER_CLASS * len(class2idx)} total)\n")

    for class_name, class_idx in sorted(class2idx.items(), key=lambda x: x[1]):
        out_folder = os.path.join(args.generated_root, class_name)
        os.makedirs(out_folder, exist_ok=True)

        class_rows   = df_by_class[class_name]
        sampled_rows = class_rows.sample(
            n=M_PER_CLASS, replace=True, random_state=SEED
        ).reset_index(drop=True)

        print(f"[{class_name}] ({len(class_rows)} train rows sample {M_PER_CLASS})")

        for i, row in tqdm(sampled_rows.iterrows(),
                           total=M_PER_CLASS, desc=f"  {class_name}"):
            prompt      = create_prompt_bioclip_plantvillage(row, inference=True)
            cond_hidden = encode_prompt_bioclip(
                prompt, bioclip_model, tokenizer, projection, device)

            img = generate_image(
                unet, vae, scheduler, cond_hidden, uncond_hidden, device, GUIDANCE)
            Image.fromarray(img).save(
                os.path.join(out_folder, f"{i:04d}.png"))

    print(f"\nDone. Images saved to: {args.generated_root}")


if __name__ == "__main__":
    main()
