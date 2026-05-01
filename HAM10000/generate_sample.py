import os
import torch
from PIL import Image
from diffusers import DDPMScheduler

DX_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def save_sample_metadata(unet, vae, cond_encoder, output_dir, device, step):

    meta = {
        "dx": torch.tensor([5], device=device),
        "site": torch.tensor([0], device=device),
        "sex": torch.tensor([0], device=device),
        "age": torch.tensor([0.45], device=device),
    }

    cond = cond_encoder(meta)
    cond = cond.unsqueeze(1).repeat(1, 77, 1)

    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = cond

    model_name = "runwayml/stable-diffusion-v1-5"
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(70)

    latents = torch.randn(1, 4, 64, 64, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(latents, t,encoder_hidden_states=cond).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        img = vae.decode(latents).sample

    img = ((img.clamp(-1, 1) + 1) / 2)[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")

    filename = os.path.join(output_dir, f"sample_step_{step}.png")
    Image.fromarray(img).save(filename)
    print(f"Saved general sample → {filename}")
    
    return filename

def save_sample_prompt(unet,vae,tokenizer,text_encoder,output_dir,device,step,prompt):
    os.makedirs(output_dir, exist_ok=True)
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )

    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

    model_name = "runwayml/stable-diffusion-v1-5"
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(70)

    latents = torch.randn(1, 4, 64, 64, device=device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        img = vae.decode(latents).sample

    img = ((img.clamp(-1, 1) + 1) / 2)[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype("uint8")

    filename = os.path.join(output_dir, f"sample_step_{step}.png")
    Image.fromarray(img).save(filename)

    print(f"Saved prompt sample → {filename}")
    print(f"Prompt: {prompt}")

    return filename


def save_all_classes(unet, vae, cond_encoder, output_dir, device, step):

    num_classes = len(DX_LABELS)

    folder = os.path.join(output_dir, "samples_by_class", f"step_{step}")
    os.makedirs(folder, exist_ok=True)

    model_name = "runwayml/stable-diffusion-v1-5"
    scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    scheduler.set_timesteps(70)

    for cls in range(num_classes):

        meta = {
            "dx": torch.tensor([cls], device=device),
            "site": torch.tensor([0], device=device),
            "sex": torch.tensor([0], device=device),
            "age": torch.tensor([0.50], device=device),
        }

        cond = cond_encoder(meta)
        cond = cond.unsqueeze(1).repeat(1, 77, 1)

        for _, module in unet.named_modules():
            if hasattr(module, "attn2"):
                module.attn2.metadata_context = cond

        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = unet(latents, t,encoder_hidden_states=cond).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / 0.18215
            img = vae.decode(latents).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")

        filename = os.path.join(folder, f"{DX_LABELS[cls]}.png")
        Image.fromarray(img).save(filename)

    print(f"Saved per-class samples → {folder}")