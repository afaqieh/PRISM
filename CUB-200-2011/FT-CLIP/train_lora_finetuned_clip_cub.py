import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset_cub_prompt import CUBDatasetPrompt
from PromptBuilder_cub import create_prompt_cub
from ...lora_utils import LoRALinear, apply_lora_to_unet, apply_lora_to_clip

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV   = "./data/cub_train.csv"
OUT_DIR     = "results/cub_stage1_clip_finetuned"
STEPS       = 50000
BATCH_SIZE  = 4
LR_UNET     = 3e-5   
LR_CLIP     = 1e-5 
LORA_RANK   = 16
CFG_DROPOUT = 0.15


def encode_prompts(prompts, tokenizer, text_encoder, device):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids      = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    encoder_hidden_states = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )[0]
    return encoder_hidden_states


def save_sample(unet, vae, tokenizer, text_encoder, out_dir, device, step, species2idx, df):
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)

    folder = os.path.join(out_dir, f"samples_step_{step}")
    os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        uncond_inputs = tokenizer(
            [""],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_hidden = text_encoder(
            input_ids=uncond_inputs.input_ids.to(device),
            attention_mask=uncond_inputs.attention_mask.to(device),
        )[0]

        for species_name in sorted(species2idx.keys()):
            row    = df[df["species_name"] == species_name].sample(n=1, random_state=step).iloc[0]
            prompt = create_prompt_cub(row, inference=True)

            cond_inputs = tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            cond_hidden = text_encoder(
                input_ids=cond_inputs.input_ids.to(device),
                attention_mask=cond_inputs.attention_mask.to(device),
            )[0]

            latents = torch.randn(1, 4, 64, 64, device=device)
            for t in scheduler.timesteps:
                n_u = unet(latents, t, encoder_hidden_states=uncond_hidden).sample
                n_c = unet(latents, t, encoder_hidden_states=cond_hidden).sample
                latents = scheduler.step(
                    n_u + 7.5 * (n_c - n_u), t, latents
                ).prev_sample
            img = vae.decode(latents / 0.18215).sample

            img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")
            safe_name = species_name.replace(" ", "_")
            Image.fromarray(img).save(os.path.join(folder, f"{safe_name}.png"))

    print(f"  Samples saved {folder}")


def train():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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

    print("Loading CLIP tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")

    print("Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder"
    ).to(device)

    for p in text_encoder.parameters():
        p.requires_grad = False

    print("Building dataset...")
    dataset = CUBDatasetPrompt(TRAIN_CSV, vae, mode="prompt", device=device)

    class_counts = dataset.df["species_name"].value_counts().to_dict()
    weights      = dataset.df["species_name"].map(lambda x: 1.0 / class_counts[x]).tolist()
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(dataset), replacement=True)
    dataloader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    print("Applying LoRA to UNet...")
    unet_lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for layer in unet_lora_layers:
        layer.to(device)
    print(f"  {len(unet_lora_layers)} LoRA layers applied to UNet")

    print("Applying LoRA to CLIP text encoder...")
    clip_lora_layers = apply_lora_to_clip(text_encoder, r=LORA_RANK)
    for layer in clip_lora_layers:
        layer.to(device)
    print(f"  {len(clip_lora_layers)} LoRA layers applied to CLIP text encoder")

    optimizer_unet = optim.AdamW(
        [p for layer in unet_lora_layers for p in layer.parameters()],
        lr=LR_UNET,
    )
    optimizer_clip = optim.AdamW(
        [p for layer in clip_lora_layers for p in layer.parameters()],
        lr=LR_CLIP,
    )

    print(f"Training for {STEPS} steps...")
    print(f"  UNet LoRA lr={LR_UNET}  |  CLIP LoRA lr={LR_CLIP}")
    step = 0

    while step < STEPS:
        for latents, noise, noisy_latents, t, prompts in dataloader:
            if step >= STEPS:
                break

            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)

            prompts = list(prompts)

            drop_mask = torch.rand(len(prompts)) < CFG_DROPOUT
            prompts   = ["" if drop_mask[i] else prompts[i] for i in range(len(prompts))]

            encoder_hidden_states = encode_prompts(prompts, tokenizer, text_encoder, device)

            noise_pred = unet(noisy_latents, t,
                              encoder_hidden_states=encoder_hidden_states).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer_unet.zero_grad()
            optimizer_clip.zero_grad()
            loss.backward()
            optimizer_unet.step()
            optimizer_clip.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{STEPS}  loss={loss.item():.6f}", flush=True)

            if step % 500 == 0:
                save_sample(unet, vae, tokenizer, text_encoder,
                            OUT_DIR, device, step, dataset.species2idx, dataset.df)

    print("Saving checkpoint...")
    torch.save(
        {
            "unet_lora_layers": [l.state_dict() for l in unet_lora_layers],
            "clip_lora_layers": [l.state_dict() for l in clip_lora_layers],
            "lora_rank":        LORA_RANK,
            "steps":            step,
            "model_name":       MODEL_NAME,
            "conditioning":     "finetuned_clip_prompt",
            "species2idx":      dataset.species2idx,
        },
        os.path.join(OUT_DIR, "lora_prompt_cub.pth"),
    )
    print(f"Done {OUT_DIR}/lora_prompt_cub.pth")


if __name__ == "__main__":
    train()
