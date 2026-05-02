from torch.utils.data import WeightedRandomSampler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from HAM10000.dataset import HAM10000Dataset
from metadata_conditioning import MetadataConditionEncoder
from ...lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

DX_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def save_sample(unet, vae, cond_encoder, output_dir, device, step):

    meta = {
        "dx": torch.tensor([5], device=device),
        "site": torch.tensor([0], device=device),
        "sex": torch.tensor([0], device=device),
        "age": torch.tensor([0.45], device=device),
    }

    cond = cond_encoder(meta)

    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = cond

    uncond = torch.zeros(1, 77, 768, device=device)

    model_name = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(50)

    latents = torch.randn(1, 4, 64, 64, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            for _, module in unet.named_modules():
                if hasattr(module, "attn2"):
                    module.attn2.metadata_context = uncond
            noise_uncond = unet(latents, t, encoder_hidden_states=uncond).sample

            for _, module in unet.named_modules():
                if hasattr(module, "attn2"):
                    module.attn2.metadata_context = cond
            noise_cond = unet(latents, t, encoder_hidden_states=cond).sample

            noise_pred = noise_uncond + 7.5 * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        img = vae.decode(latents).sample

    img = ((img.clamp(-1, 1) + 1) / 2)[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")

    filename = os.path.join(output_dir, f"sample_step_{step}.png")
    Image.fromarray(img).save(filename)
    print(f"Saved general sample {filename}")

def save_all_classes(unet, vae, cond_encoder, output_dir, device, step):

    num_classes = len(DX_LABELS)

    folder = os.path.join(output_dir, "samples_by_class", f"step_{step}")
    os.makedirs(folder, exist_ok=True)

    uncond = torch.zeros(1, 77, 768, device=device)

    model_name = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(50)

    for cls in range(num_classes):

        meta = {
            "dx": torch.tensor([cls], device=device),
            "site": torch.tensor([0], device=device),
            "sex": torch.tensor([0], device=device),
            "age": torch.tensor([0.50], device=device),
        }

        cond = cond_encoder(meta)

        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                for _, module in unet.named_modules():
                    if hasattr(module, "attn2"):
                        module.attn2.metadata_context = uncond
                noise_uncond = unet(latents, t, encoder_hidden_states=uncond).sample

                for _, module in unet.named_modules():
                    if hasattr(module, "attn2"):
                        module.attn2.metadata_context = cond
                noise_cond = unet(latents, t, encoder_hidden_states=cond).sample

                noise_pred = noise_uncond + 7.5 * (noise_cond - noise_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / 0.18215
            img = vae.decode(latents).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")

        filename = os.path.join(folder, f"{DX_LABELS[cls]}.png")
        Image.fromarray(img).save(filename)

    print(f"Saved per-class samples {folder}")

def train(
    csv="./data/HAM10000_metadata_train.csv",
    img_dir="/storage/homefs/af24h089/MSGAI/Final/ham_lora",
    outdir="results/metadata_stage1",
    batch_size=4,
    lr=3e-5,
    steps=20000,
    r=16,
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
    dataset = HAM10000Dataset(csv, img_dir, vae, device=device)

    class_counts = dataset.df["dx"].value_counts().to_dict()
    weights      = dataset.df["dx"].map(lambda x: 1.0 / class_counts[x]).tolist()
    sampler      = WeightedRandomSampler(
        weights     = torch.DoubleTensor(weights),
        num_samples = len(dataset),
        replacement = True,
    )   
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler
    )   


    print("Building metadata encoder...")
    print("Building metadata encoder...")
    cond_encoder = MetadataConditionEncoder.for_ham10000(
        num_dx=len(dataset.dx2idx),
        num_sites=len(dataset.site2idx),
        num_sex=len(dataset.sex2idx),
        hidden_dim=256,
        final_dim=768
    ).to(device)

    print("Injecting metadata into attention...")
    inject_metadata_into_attention(unet, device)

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=r)
    for layer in lora_layers:
        layer.to(device)

    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) +
        [p for layer in lora_layers for p in layer.parameters()],
        lr=lr
    )

    print("Training...")
    step = 0

    while step < steps:
        for latents, noise, noisy_latents, t, meta in dataloader:
            if step >= steps:
                break

            latents = latents.to(device)
            noise = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t = t.squeeze().to(device)

            meta_batch = {
                "dx": meta["dx"].to(device),
                "site": meta["site"].to(device),
                "sex": meta["sex"].to(device),
                "age": meta["age"].to(device),
            }

            cond = cond_encoder(meta_batch)

            dropout_mask = (torch.rand(cond.shape[0], 1, 1, device=device) < 0.15)
            cond = torch.where(dropout_mask, torch.zeros_like(cond), cond)

            for _, module in unet.named_modules():
                if hasattr(module, "attn2"):
                    module.attn2.metadata_context = cond

            noise_pred = unet(noisy_latents, t, encoder_hidden_states=cond).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if step % 50 == 0:
                print(f"Step {step}/{steps}  Loss: {loss.item():.6f}")

            if step % 500 == 0:
                save_sample(unet, vae, cond_encoder, outdir, device, step)
                save_all_classes(unet, vae, cond_encoder, outdir, device, step)

    print("Saving weights...")
    torch.save(
        {
            "lora_layers": [l.state_dict() for l in lora_layers],
            "cond_encoder": cond_encoder.state_dict(),
        },
        os.path.join(outdir, "lora_numeric_condition.pth")
    )

    print("Training complete.")



if __name__ == "__main__":
    train()

