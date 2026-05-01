import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTokenizer, CLIPTextModel
from ...lora_utils import apply_lora_to_unet, LoRALinear, apply_lora_to_clip
from generate_sample import save_sample_prompt
from datetime import datetime
from torch.utils.data import WeightedRandomSampler

def train(
    dataset,
    csv="../data/HAM10000_metadata_train.csv",
    img_dir="../images/HAM10000_images",
    outdir="./results/clip_finetuned_stage1",
    batch_size=4,
    lr_unet=3e-5,
    lr_clip=1e-5,
    steps=40000,
    r=16,
    prompt="Dermoscopy image of a dermatofibroma on the back of a 65-year-old male."
    ):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(outdir, exist_ok=True)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="vae"
    ).to(device)
    vae.eval()

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="unet"
    ).to(device)

    for p in unet.parameters():
        p.requires_grad = False

    print("Building dataset...")
    if dataset == 'ham':
        from dataset_ham10000_prompt import HAM10000Dataset
        dataset = HAM10000Dataset(
            csv_path=csv,
            img_dir=img_dir,
            vae=vae,
            device=device,
            mode='prompt'
        )

        class_counts = dataset.df["dx"].value_counts().to_dict()
        weights = dataset.df["dx"].map(lambda x: 1.0 / class_counts[x]).tolist()
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(dataset),
            replacement=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler
        )

    print("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer"
    )

    print("Loading text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="text_encoder"
    ).to(device)

    for p in text_encoder.parameters():
        p.requires_grad = False

    print("Applying LoRA to CLIP text encoder...")
    clip_lora_layers = apply_lora_to_clip(text_encoder, r=r)
    for layer in clip_lora_layers:
        layer.to(device)
    print(f"  {len(clip_lora_layers)} LoRA layers applied to CLIP text encoder")
    print("Applying LoRA to UNet...")
    unet_lora_layers = apply_lora_to_unet(unet, r=r)
    for layer in unet_lora_layers:
        layer.to(device)
    print(f"  {len(unet_lora_layers)} LoRA layers applied to UNet")

    optimizer_unet = optim.AdamW(
        [p for layer in unet_lora_layers for p in layer.parameters()],
        lr=lr_unet
    )
    optimizer_clip = optim.AdamW(
        [p for layer in clip_lora_layers for p in layer.parameters()],
        lr=lr_clip
    )

    print("Training...")
    step = 0

    while step < steps:
        for latents, noise, noisy_latents, t, prompts in tqdm(dataloader):
            if step >= steps:
                break

            latents = latents.to(device)
            noise = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t = t.squeeze().to(device)

            prompts = list(prompts)

            drop_mask = torch.rand(len(prompts)) < 0.15
            prompts = ["" if drop_mask[i] else prompts[i] for i in range(len(prompts))]

            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            )

            input_ids      = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            
            encoder_hidden_states = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )[0]

            noise_pred = unet(
                noisy_latents,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer_unet.zero_grad()
            optimizer_clip.zero_grad()
            loss.backward()
            optimizer_unet.step()
            optimizer_clip.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{steps}  Loss: {loss.item():.6f}", flush=True)

            if step % 500 == 0:
                filename = save_sample_prompt(
                    unet, vae, tokenizer, text_encoder, outdir, device, step, prompt)

    print("Saving weights...")
    ckpt_path = os.path.join(outdir, "lora_prompt_condition.pth")

    torch.save(
        {
            "unet_lora_layers": [layer.state_dict() for layer in unet_lora_layers],
            "clip_lora_layers": [layer.state_dict() for layer in clip_lora_layers],
            "lora_rank":        r,
            "steps":            step,
            "model_name":       "runwayml/stable-diffusion-v1-5",
            "conditioning":     "finetuned_clip",
        },
        ckpt_path
    )

    print("Training complete.")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    train(dataset='ham',
          csv='../data/HAM10000_metadata_train.csv',
          img_dir="../images/HAM10000_images",
          outdir='results/clip_finetuned_stage1')
