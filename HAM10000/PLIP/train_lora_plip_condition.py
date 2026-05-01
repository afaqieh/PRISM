import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import AutoTokenizer, CLIPModel
from ...lora_utils import apply_lora_to_unet
from generate_sample import save_sample_prompt
from datetime import datetime
from torch.utils.data import WeightedRandomSampler

def train(
    dataset,
    csv="../data/HAM10000_metadata_train.csv",
    img_dir="../images/HAM10000_images",
    outdir="./results/plip_stage1",
    batch_size=4,
    lr=3e-5,
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

    print("Loading PLIP tokenizer and text encoder...")
    tokenizer = AutoTokenizer.from_pretrained("vinid/plip")
    plip_model   = CLIPModel.from_pretrained("vinid/plip").to(device)
    text_encoder = plip_model.text_model
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    plip_proj = nn.Linear(512, 768).to(device)

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=r)
    for layer in lora_layers:
        layer.to(device)

    optimizer = optim.AdamW(
        [p for layer in lora_layers for p in layer.parameters()] +
        list(plip_proj.parameters()),
        lr=lr
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
                max_length=77,
                return_tensors="pt"
            )

            input_ids      = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            with torch.no_grad():
                plip_out = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )[0] 

            encoder_hidden_states = plip_proj(plip_out)

            noise_pred = unet(
                noisy_latents,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{steps}  Loss: {loss.item():.6f}")

            if step % 500 == 0:
                pass

    print("Saving weights...")
    ckpt_path = os.path.join(outdir, "lora_prompt_condition.pth")

    torch.save(
        {
            "lora_layers":  [layer.state_dict() for layer in lora_layers],
            "plip_proj":    plip_proj.state_dict(),
            "lora_rank":    r,
            "steps":        step,
            "model_name":   "runwayml/stable-diffusion-v1-5",
            "conditioning": "plip",
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
          outdir='results/plip_stage1')
