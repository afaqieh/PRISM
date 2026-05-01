import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from metadata_conditioning import MetadataConditionEncoder
from ..lora_utils import LoRALinear, apply_lora_to_unet

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DDIM_STEPS        = 50
GUIDANCE          = 6.0
SEED              = 0
CAPTURE_FRAC      = 0.2

TOKEN_NAMES = ["species", "throat\ncolor", "forehead\ncolor",
               "belly\ncolor", "nape\ncolor", "summary"]
N_TOKENS    = 6


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context
            
class CaptureState:
    def __init__(self):
        self.active = False
        self.maps   = []

    def clear(self):
        self.maps.clear()


class CaptureAttnProcessor:
    def __init__(self, attn_module, state: CaptureState):
        self.attn_module = attn_module
        self.state       = state

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        encoder_hidden_states = self.attn_module.metadata_context

        residual   = hidden_states
        input_ndim = hidden_states.ndim

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.state.active:
            BH, S, T = attention_probs.shape
            n_heads  = attn.heads
            B_batch  = max(1, BH // n_heads)
            attn_avg = (
                attention_probs
                .reshape(B_batch, n_heads, S, T)
                .mean(dim=(0, 1))
                .detach()
                .float()
                .cpu()
            )
            hw = int(math.isqrt(S))
            if hw * hw == S:
                n_cap = min(N_TOKENS, T)
                maps  = attn_avg[:, :n_cap].reshape(hw, hw, n_cap)
                self.state.maps.append(maps)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = (
                hidden_states
                .transpose(-1, -2)
                .reshape(B, C, H, W)
            )

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)

        return hidden_states


def inject_capture_attention(unet, device, state: CaptureState):
    n_patched = 0
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = torch.zeros(1, 77, 768, device=device)
            module.attn2.processor = CaptureAttnProcessor(module.attn2, state)
            n_patched += 1
    return n_patched

def aggregate_attention(state: CaptureState, out_size: int = 64):
    if not state.maps:
        return np.zeros((out_size, out_size, N_TOKENS), dtype=np.float32)

    upsampled = []
    for maps in state.maps:
        h, w, nc = maps.shape
        m = maps.permute(2, 0, 1).unsqueeze(0)
        m = F.interpolate(
            m.float(), size=(out_size, out_size),
            mode="bilinear", align_corners=False
        )
        upsampled.append(m[0])

    stacked = torch.stack(upsampled, dim=0)
    avg     = stacked.mean(dim=0)
    result  = avg.permute(1, 2, 0).numpy()
    return result.astype(np.float32)

def generate_with_attention(
    cond_encoder, unet, vae, scheduler, metadata,
    state: CaptureState, device, guidance
):
    cond   = cond_encoder(metadata)
    uncond = torch.zeros(1, 77, 768, device=device)

    torch.manual_seed(SEED)
    latents = torch.randn(1, 4, 64, 64, device=device)

    state.clear()

    total_steps  = len(scheduler.timesteps)
    capture_from = int(total_steps * (1.0 - CAPTURE_FRAC))

    with torch.no_grad():
        for step_idx, t in enumerate(scheduler.timesteps):
            set_attention_context(unet, uncond)
            state.active = False
            n_u = unet(latents, t, encoder_hidden_states=uncond).sample

            set_attention_context(unet, cond)
            state.active = (step_idx >= capture_from)
            n_c = unet(latents, t, encoder_hidden_states=cond).sample
            state.active = False

            noise_pred = n_u + guidance * (n_c - n_u)
            latents    = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / 0.18215
        image   = vae.decode(latents).sample

    img_np  = (image.clamp(-1, 1) + 1) / 2
    img_np  = img_np[0].permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray((img_np * 255).astype("uint8"))

    attn_maps = aggregate_attention(state, out_size=64)
    return pil_img, attn_maps

def _norm_map(m):
    lo, hi = m.min(), m.max()
    if hi > lo:
        return (m - lo) / (hi - lo)
    return np.zeros_like(m)


def render_overlay(pil_img, attn_map, img_res=128):
    gray = np.array(
        pil_img.convert("L").resize((img_res, img_res), Image.BILINEAR)
    ) / 255.0

    attn_resized = np.array(
        Image.fromarray((_norm_map(attn_map) * 255).astype(np.uint8))
        .resize((img_res, img_res), Image.BILINEAR)
    ) / 255.0

    cmap   = plt.get_cmap("jet")
    heat   = cmap(attn_resized)[:, :, :3]

    gray3  = np.stack([gray] * 3, axis=-1)
    blend  = (0.45 * gray3 + 0.55 * heat).clip(0, 1)
    return (blend * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_path",  required=True,
                    help="Path to trained lora_cub.pth checkpoint")
    ap.add_argument("--csv_path",   default="./data/cub_train.csv",
                    help="Train-split CSV (used to sample representative metadata)")
    ap.add_argument("--out_path",   default="results/attention_maps_cub.png",
                    help="Output figure path (.png)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.lora_path}")
    ckpt          = torch.load(args.lora_path, map_location="cpu", weights_only=False)
    species2idx   = ckpt["species2idx"]
    lora_rank     = ckpt["lora_rank"]
    field_configs = ckpt["field_configs"]
    idx2species   = {v: k for k, v in species2idx.items()}
    n_species     = len(species2idx)
    print(f"  {n_species} species: {sorted(species2idx.keys())}")

    df = pd.read_csv(args.csv_path).reset_index(drop=True)
    print(f"  Train rows: {len(df)}")

    print("Loading VAE ...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    print("Loading UNet ...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer in lora_layers:
        layer.to(device)

    capture_state = CaptureState()
    n_patched = inject_capture_attention(unet, device, capture_state)
    print(f"  Patched {n_patched} cross-attention modules")

    for layer, state_dict in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state_dict)

    cond_encoder = MetadataConditionEncoder(field_configs, hidden_dim=256, final_dim=768)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.to(device).eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)

    species_list = sorted(species2idx.items(), key=lambda x: x[1])
    results      = []

    for sp_name, sp_idx in species_list:
        print(f"\n[{sp_idx + 1}/{n_species}] {sp_name}")

        rows = df[df["species_name"] == sp_name].reset_index(drop=True)
        if len(rows) == 0:
            print(f"  WARNING: no train rows — using zero attributes")
            meta = {
                "species":        torch.tensor([sp_idx], device=device),
                "throat_color":   torch.tensor([0],      device=device),
                "forehead_color": torch.tensor([0],      device=device),
                "belly_color":    torch.tensor([0],      device=device),
                "nape_color":     torch.tensor([0],      device=device),
            }
        else:
            row = rows.iloc[0]
            meta = {
                "species":        torch.tensor([sp_idx],                   device=device),
                "throat_color":   torch.tensor([int(row.throat_color)],    device=device),
                "forehead_color": torch.tensor([int(row.forehead_color)],  device=device),
                "belly_color":    torch.tensor([int(row.belly_color)],     device=device),
                "nape_color":     torch.tensor([int(row.nape_color)],      device=device),
            }
            print(f"  throat={int(row.throat_color)}  forehead={int(row.forehead_color)}"
                  f"  belly={int(row.belly_color)}  nape={int(row.nape_color)}")

        pil_img, attn_maps = generate_with_attention(
            cond_encoder, unet, vae, scheduler,
            meta, capture_state, device, GUIDANCE,
        )
        n_cond_maps  = len(capture_state.maps)
        n_late_steps = int(DDIM_STEPS * CAPTURE_FRAC)
        print(f"  Captured {n_cond_maps} layer-step maps "
              f"(last {n_late_steps}/{DDIM_STEPS} steps × layers)  "
              f"attn shape: {attn_maps.shape}")
        results.append((sp_name, pil_img, attn_maps))

    n_cols     = 1 + N_TOKENS
    col_titles = ["Generated"] + TOKEN_NAMES

    print(f"\nPlotting {n_species} × {n_cols} figure ...")
    
    save_dir = os.path.dirname(args.out_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    CELL_PX  = 2.3
    fig_w    = n_cols  * CELL_PX + 1.4
    fig_h    = (n_species + 1) * CELL_PX

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    gs  = gridspec.GridSpec(
        n_species + 1, n_cols,
        figure  = fig,
        hspace  = 0.04,
        wspace  = 0.04,
        top     = 0.97,
        bottom  = 0.01,
        left    = 0.11,
        right   = 0.99,
    )

    for col, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, title,
                ha="center", va="center",
                fontsize=8, fontweight="bold",
                transform=ax.transAxes)
        ax.set_axis_off()

    IMG_RES = 128

    for row_idx, (sp_name, pil_img, attn_maps) in enumerate(results):
        r = row_idx + 1

        ax_gen = fig.add_subplot(gs[r, 0])
        ax_gen.imshow(pil_img)
        short = sp_name.replace("Warbler", "W.").replace(" ", "\n")
        ax_gen.set_ylabel(short, fontsize=6.5, rotation=0,
                          labelpad=50, va="center", ha="right")
        ax_gen.set_xticks([])
        ax_gen.set_yticks([])

        for tok in range(N_TOKENS):
            ax = fig.add_subplot(gs[r, tok + 1])
            overlay = render_overlay(pil_img, attn_maps[:, :, tok], img_res=IMG_RES)
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full figure {args.out_path}")

    row_dir = os.path.splitext(args.out_path)[0] + "_rows"
    os.makedirs(row_dir, exist_ok=True)

    for sp_name, pil_img, attn_maps in results:
        fig2, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.6, 2.8))

        axes[0].imshow(pil_img)
        axes[0].set_title("Generated", fontsize=9, fontweight="bold")
        axes[0].axis("off")

        for tok in range(N_TOKENS):
            ax      = axes[tok + 1]
            overlay = render_overlay(pil_img, attn_maps[:, :, tok], img_res=256)
            ax.imshow(overlay)
            ax.set_title(TOKEN_NAMES[tok].replace("\n", " "), fontsize=8)
            ax.axis("off")

        plt.suptitle(sp_name, fontsize=10, fontweight="bold", y=1.02)
        plt.tight_layout()

        fname = sp_name.replace(" ", "_") + "_attn.png"
        fig2.savefig(os.path.join(row_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"Individual rows saved {row_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
