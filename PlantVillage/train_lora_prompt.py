import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from torch.utils.data import DataLoader, WeightedRandomSampler
from PlantVillage.dataset_prompt import PlantVillageDatasetPrompt
from PlantVillage.PromptBuilder_plantvillage import create_prompt_plantvillage
from ..lora_utils import LoRALinear, apply_lora_to_unet

MODEL_NAME  = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV   = "./data/plantvillage_train.csv"
OUT_DIR     = "results/stage1_clip"
STEPS       = 20000
BATCH_SIZE  = 4
LR          = 3e-5
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
    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
    return encoder_hidden_states


def save_sample(unet, vae, tokenizer, text_encoder, out_dir, device, step, class2idx, df):
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)

    folder = os.path.join(out_dir, f"samples_step_{step}")
    os.makedirs(folder, exist_ok=True)

    uncond_hidden = encode_prompts([""], tokenizer, text_encoder, device)

    for class_name in sorted(class2idx.keys()):
        row         = df[df["class_name"] == class_name].sample(n=1, random_state=step).iloc[0]
        prompt      = create_prompt_plantvillage(row, inference=True)
        cond_hidden = encode_prompts([prompt], tokenizer, text_encoder, device)
        latents     = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                n_u = unet(latents, t, encoder_hidden_states=uncond_hidden).sample
                n_c = unet(latents, t, encoder_hidden_states=cond_hidden).sample
                latents = scheduler.step(
                    n_u + 7.5 * (n_c - n_u), t, latents
                ).prev_sample
            img = vae.decode(latents / 0.18215).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")
        safe_name = class_name.replace(" ", "_")
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
        MODEL_NAME, subfolder="text_encoder").to(device)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    print("Building dataset...")
    dataset = PlantVillageDatasetPrompt(TRAIN_CSV, vae, mode="prompt", device=device)

    class_counts = dataset.df["class_name"].value_counts().to_dict()
    weights      = dataset.df["class_name"].map(lambda x: 1.0 / class_counts[x]).tolist()
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights), num_samples=len(dataset), replacement=True)
    dataloader   = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for layer in lora_layers:
        layer.to(device)

    optimizer = optim.AdamW(
        [p for layer in lora_layers for p in layer.parameters()],
        lr=LR,
    )

    print(f"Training for {STEPS} steps...")
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{STEPS}  loss={loss.item():.6f}")

            if step % 500 == 0:
                save_sample(unet, vae, tokenizer, text_encoder,
                            OUT_DIR, device, step, dataset.class2idx, dataset.df)

    print("Saving checkpoint...")
    torch.save(
        {
            "lora_layers":  [l.state_dict() for l in lora_layers],
            "lora_rank":    LORA_RANK,
            "steps":        step,
            "model_name":   MODEL_NAME,
            "conditioning": "clip_prompt",
            "class2idx":    dataset.class2idx,
        },
        os.path.join(OUT_DIR, "lora_prompt_plantvillage.pth"),
    )
    print(f"Done {OUT_DIR}/lora_prompt_plantvillage.pth")


if __name__ == "__main__":
    train()
