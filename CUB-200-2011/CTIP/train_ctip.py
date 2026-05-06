import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from diffusers import (AutoencoderKL, UNet2DConditionModel,
                       DDPMScheduler, DDIMScheduler)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from lora_utils import apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV  = "./data/cub_train.csv"
OUT_DIR    = "results/ctip_baseline"

PRETRAIN_STEPS = 5_000
PRETRAIN_LR    = 1e-4
PRETRAIN_BATCH = 32
EMBED_DIM      = 256
TEMPERATURE    = 0.01

TRAIN_STEPS  = 30_000
TRAIN_LR     = 3e-5
TRAIN_BATCH  = 4
LORA_RANK    = 16
CFG_DROPOUT  = 0.15

NUM_SPECIES         = 15
NUM_THROAT_COLORS   = 15
NUM_FOREHEAD_COLORS = 15
NUM_BELLY_COLORS    = 15
NUM_NAPE_COLORS     = 15

TABULAR_INPUT_DIM   = (NUM_SPECIES + NUM_THROAT_COLORS +
                       NUM_FOREHEAD_COLORS + NUM_BELLY_COLORS + NUM_NAPE_COLORS)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int = TABULAR_INPUT_DIM,
                 embed_dim: int = EMBED_DIM):
        super().__init__()
        hidden = embed_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_onehot(meta: dict, device: torch.device) -> torch.Tensor:
    species        = F.one_hot(meta["species"],        num_classes=NUM_SPECIES).float()
    throat_color   = F.one_hot(meta["throat_color"],   num_classes=NUM_THROAT_COLORS).float()
    forehead_color = F.one_hot(meta["forehead_color"], num_classes=NUM_FOREHEAD_COLORS).float()
    belly_color    = F.one_hot(meta["belly_color"],    num_classes=NUM_BELLY_COLORS).float()
    nape_color     = F.one_hot(meta["nape_color"],     num_classes=NUM_NAPE_COLORS).float()
    return torch.cat([species, throat_color, forehead_color,
                      belly_color, nape_color], dim=-1).to(device)


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0)
            feat_dim = self.backbone.num_features
        except ImportError:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            vit.heads = nn.Identity()
            self.backbone = vit
            feat_dim = 768
        self.proj = nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))

class CTIPConditioner(nn.Module):
    def __init__(self, tabular_encoder: TabularEncoder,
                 embed_dim: int = EMBED_DIM,
                 sd_dim: int = 768,
                 seq_len: int = 77):
        super().__init__()
        self.tabular_encoder = tabular_encoder
        self.proj = nn.Linear(embed_dim, sd_dim)
        self.seq_len = seq_len

    def forward(self, meta: dict) -> torch.Tensor:
        device = next(self.parameters()).device
        x   = build_onehot(meta, device)
        emb = self.tabular_encoder(x)
        emb = self.proj(emb)
        return emb.unsqueeze(1).expand(-1, self.seq_len, -1)


_vit_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_sd_tf = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class PretrainDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)
        self.df = df
        species_list     = sorted(df["species_name"].unique())
        self.species2idx = {s: i for i, s in enumerate(species_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = _vit_tf(Image.open(row["full_path"]).convert("RGB"))
        meta = {
            "species":        torch.tensor(self.species2idx[row["species_name"]], dtype=torch.long),
            "throat_color":   torch.tensor(int(row["throat_color"]),   dtype=torch.long),
            "forehead_color": torch.tensor(int(row["forehead_color"]), dtype=torch.long),
            "belly_color":    torch.tensor(int(row["belly_color"]),     dtype=torch.long),
            "nape_color":     torch.tensor(int(row["nape_color"]),      dtype=torch.long),
        }
        return img, meta


class DiffusionDataset(Dataset):
    def __init__(self, csv_path: str, vae: AutoencoderKL, device: str):
        df = pd.read_csv(csv_path)
        df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)
        self.df        = df
        self.vae       = vae
        self.device    = device
        self.scheduler = DDPMScheduler.from_pretrained(
            MODEL_NAME, subfolder="scheduler")
        species_list     = sorted(df["species_name"].unique())
        self.species2idx = {s: i for i, s in enumerate(species_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = _sd_tf(Image.open(row["full_path"]).convert("RGB"))
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample() * 0.18215
            latents = latents.squeeze(0).detach()

        t             = torch.randint(0, self.scheduler.num_train_timesteps,
                                      (1,), device=self.device, dtype=torch.long)
        noise         = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t)

        meta = {
            "species":        torch.tensor(self.species2idx[row["species_name"]], dtype=torch.long),
            "throat_color":   torch.tensor(int(row["throat_color"]),   dtype=torch.long),
            "forehead_color": torch.tensor(int(row["forehead_color"]), dtype=torch.long),
            "belly_color":    torch.tensor(int(row["belly_color"]),     dtype=torch.long),
            "nape_color":     torch.tensor(int(row["nape_color"]),      dtype=torch.long),
        }
        return latents, noise, noisy_latents, t, meta


def clip_loss(tab_emb: torch.Tensor, img_emb: torch.Tensor,
              temperature: float = TEMPERATURE) -> torch.Tensor:
    tab_emb = F.normalize(tab_emb, dim=-1)
    img_emb = F.normalize(img_emb, dim=-1)
    logits  = (tab_emb @ img_emb.T) / temperature
    labels  = torch.arange(len(logits), device=logits.device)
    loss_i  = F.cross_entropy(logits,   labels)
    loss_t  = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def pretrain(device: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Stage A — CTIP Contrastive Pre-training")

    dataset = PretrainDataset(TRAIN_CSV)
    loader  = DataLoader(dataset, batch_size=PRETRAIN_BATCH,
                         shuffle=True, num_workers=4, drop_last=True)

    tab_enc = TabularEncoder().to(device)
    img_enc = ImageEncoder().to(device)

    optimizer = optim.Adam(
        list(tab_enc.parameters()) + list(img_enc.parameters()),
        lr=PRETRAIN_LR, betas=(0.0, 0.9))

    step  = 0
    pbar  = tqdm(total=PRETRAIN_STEPS, desc="Pretrain")
    while step < PRETRAIN_STEPS:
        for imgs, meta in loader:
            if step >= PRETRAIN_STEPS:
                break

            imgs = imgs.to(device)
            meta = {k: v.to(device) for k, v in meta.items()}

            x       = build_onehot(meta, device)
            tab_emb = tab_enc(x)
            img_emb = img_enc(imgs)
            loss    = clip_loss(tab_emb, img_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % 100 == 0:
                print(f"  Step {step}/{PRETRAIN_STEPS}  loss={loss.item():.4f}")

    pbar.close()

    ckpt_path = os.path.join(OUT_DIR, "ctip_pretrained.pth")
    torch.save({
        "tab_enc": tab_enc.state_dict(),
        "img_enc": img_enc.state_dict(),
    }, ckpt_path)
    print(f"Pre-training done. Saved {ckpt_path}\n")
    return tab_enc

def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context


def save_samples(unet, vae, conditioner, species2idx, out_dir, device, step):
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)
    uncond = torch.zeros(1, 77, 768, device=device)

    folder = os.path.join(out_dir, f"samples_step_{step}")
    os.makedirs(folder, exist_ok=True)

    for species_name, species_idx in sorted(species2idx.items(), key=lambda x: x[1]):
        meta = {
            "species":        torch.tensor([species_idx], device=device),
            "throat_color":   torch.tensor([0], device=device),
            "forehead_color": torch.tensor([0], device=device),
            "belly_color":    torch.tensor([0], device=device),
            "nape_color":     torch.tensor([0], device=device),
        }
        cond    = conditioner(meta)
        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                set_attention_context(unet, uncond)
                n_u = unet(latents, t, encoder_hidden_states=uncond).sample
                set_attention_context(unet, cond)
                n_c = unet(latents, t, encoder_hidden_states=cond).sample
                noise_pred = n_u + 7.5 * (n_c - n_u)
                latents    = scheduler.step(noise_pred, t, latents).prev_sample
            img = vae.decode(latents / 0.18215).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(folder, f"{species_name.replace(' ', '_')}.png"))

    print(f"Samples saved {folder}")


def train_ldm(tab_enc: TabularEncoder, device: str):
    print("Stage B — LDM Fine-tuning (CTIP conditioning)")

    conditioner = CTIPConditioner(tab_enc).to(device)
    for p in conditioner.tabular_encoder.parameters():
        p.requires_grad = False
    print("  TabularEncoder frozen. Training: projection + LoRA only.")

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    inject_metadata_into_attention(unet, device)
    lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for l in lora_layers:
        l.to(device)

    dataset = DiffusionDataset(TRAIN_CSV, vae, device)
    class_counts = dataset.df["species_name"].value_counts().to_dict()
    weights      = dataset.df["species_name"].map(lambda x: 1.0 / class_counts[x]).tolist()
    loader       = DataLoader(
        dataset, batch_size=TRAIN_BATCH,
        sampler=WeightedRandomSampler(
            torch.DoubleTensor(weights), len(dataset), replacement=True))

    trainable = (list(conditioner.proj.parameters()) +
                 [p for l in lora_layers for p in l.parameters()])
    optimizer = optim.AdamW(trainable, lr=TRAIN_LR)

    step = 0
    pbar = tqdm(total=TRAIN_STEPS, desc="LDM train")
    while step < TRAIN_STEPS:
        for latents, noise, noisy_latents, t, meta in loader:
            if step >= TRAIN_STEPS:
                break

            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)
            meta          = {k: v.to(device) for k, v in meta.items()}

            cond = conditioner(meta)

            dropout_mask = (torch.rand(cond.shape[0], 1, 1, device=device) < CFG_DROPOUT)
            cond = torch.where(dropout_mask, torch.zeros_like(cond), cond)

            set_attention_context(unet, cond)
            noise_pred = unet(noisy_latents, t,
                              encoder_hidden_states=cond).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % 50 == 0:
                print(f"  Step {step}/{TRAIN_STEPS}  loss={loss.item():.6f}")

            if step % 500 == 0:
                save_samples(unet, vae, conditioner,
                             dataset.species2idx, OUT_DIR, device, step)

            if step % 1000 == 0:
                ckpt_path = os.path.join(OUT_DIR, f"ctip_ldm_step{step}.pth")
                torch.save({
                    "lora_layers":  [l.state_dict() for l in lora_layers],
                    "conditioner":  conditioner.state_dict(),
                    "species2idx":  dataset.species2idx,
                    "lora_rank":    LORA_RANK,
                    "step":         step,
                }, ckpt_path)
                print(f"Checkpoint {ckpt_path}")

    pbar.close()

    final_path = os.path.join(OUT_DIR, "ctip_ldm_final.pth")
    torch.save({
        "lora_layers": [l.state_dict() for l in lora_layers],
        "conditioner": conditioner.state_dict(),
        "species2idx": dataset.species2idx,
        "lora_rank":   LORA_RANK,
        "step":        step,
    }, final_path)
    print(f"\nDone. Final checkpoint{final_path}")
    return final_path

def generate_all(ckpt_path: str, device: str,
                 out_dir: str = "results/ctip_generated",
                 n_per_class: int = 100,
                 cfg: float = 6.0,
                 steps: int = 50):
    print(f"Generating {n_per_class} images per species {out_dir}")

    ckpt        = torch.load(ckpt_path, map_location=device, weights_only=False)
    species2idx = ckpt["species2idx"]
    lora_rank   = ckpt.get("lora_rank", LORA_RANK)

    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    inject_metadata_into_attention(unet, device)
    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer, state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state)
        layer.to(device)

    tab_enc     = TabularEncoder().to(device)
    conditioner = CTIPConditioner(tab_enc).to(device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(steps)
    uncond = torch.zeros(1, 77, 768, device=device)

    for species_name, species_idx in tqdm(
            sorted(species2idx.items(), key=lambda x: x[1]), desc="Generating"):
        folder = os.path.join(out_dir, species_name.replace(" ", "_"))
        os.makedirs(folder, exist_ok=True)

        meta = {
            "species":        torch.tensor([species_idx], device=device),
            "throat_color":   torch.tensor([0], device=device),
            "forehead_color": torch.tensor([0], device=device),
            "belly_color":    torch.tensor([0], device=device),
            "nape_color":     torch.tensor([0], device=device),
        }
        cond = conditioner(meta)

        for i in tqdm(range(n_per_class), desc=species_name, leave=False):
            latents = torch.randn(1, 4, 64, 64, device=device)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    set_attention_context(unet, uncond)
                    n_u = unet(latents, t, encoder_hidden_states=uncond).sample
                    set_attention_context(unet, cond)
                    n_c = unet(latents, t, encoder_hidden_states=cond).sample
                    latents = scheduler.step(
                        n_u + cfg * (n_c - n_u), t, latents).prev_sample
                img = vae.decode(latents / 0.18215).sample
            img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype("uint8")
            Image.fromarray(img).save(os.path.join(folder, f"{i:04d}.png"))

    print(f"Generation complete {out_dir}")

def main():
    assert torch.cuda.is_available(), "CUDA required."
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    pretrain_ckpt = os.path.join(OUT_DIR, "ctip_pretrained.pth")
    if os.path.exists(pretrain_ckpt):
        print(f"Found pre-trained checkpoint at {pretrain_ckpt}, skipping Stage A.")
        tab_enc = TabularEncoder().to(device)
        tab_enc.load_state_dict(torch.load(pretrain_ckpt,
                                           map_location=device)["tab_enc"])
    else:
        tab_enc = pretrain(device)

    final_ckpt = train_ldm(tab_enc, device)
    generate_all(final_ckpt, device,
                 out_dir="results/ctip_generated",
                 n_per_class=100,
                 cfg=6.0)


if __name__ == "__main__":
    main()
