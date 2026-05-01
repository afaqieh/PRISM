import os
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from metadata_conditioning import MetadataConditionEncoder
from tqdm import tqdm
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_closing
from ..lora_utils import LoRALinear, apply_lora_to_unet

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

TOK_SPECIES  = 0
TOK_THROAT   = 1
TOK_FOREHEAD = 2
TOK_BELLY    = 3
TOK_NAPE     = 4
TOK_SUMMARY  = 5
N_TOKENS     = 6

ATTN_RES        = 64
NOISE_T_DEFAULT = 500

img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


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

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
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
                hidden_states.transpose(1, 2)).transpose(1, 2)

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
                .detach().float().cpu()
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
            hidden_states = (hidden_states.transpose(-1, -2)
                             .reshape(B, C, H, W))
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


def aggregate_attention(state: CaptureState, out_size: int = ATTN_RES):
    if not state.maps:
        return np.zeros((out_size, out_size, N_TOKENS), dtype=np.float32)
    upsampled = []
    for maps in state.maps:
        h, w, nc = maps.shape
        m = maps.permute(2, 0, 1).unsqueeze(0).float()
        m = F.interpolate(m, size=(out_size, out_size),
                          mode="bilinear", align_corners=False)
        upsampled.append(m[0])
    stacked = torch.stack(upsampled, dim=0)
    avg     = stacked.mean(dim=0)
    return avg.permute(1, 2, 0).numpy().astype(np.float32)

def get_seg_path(full_img_path, cub_root):
    images_marker = os.path.join("CUB_200_2011", "images") + os.sep
    idx = full_img_path.find(images_marker)
    if idx == -1:
        idx2 = full_img_path.rfind(os.sep + "images" + os.sep)
        if idx2 == -1:
            return None
        rel = full_img_path[idx2 + len(os.sep + "images" + os.sep):]
    else:
        rel = full_img_path[idx + len(images_marker):]

    seg_rel = os.path.splitext(rel)[0] + ".png"
    return os.path.join(cub_root, "segmentations", seg_rel)


def load_segmentation(seg_path, size=ATTN_RES):
    if seg_path is None or not os.path.exists(seg_path):
        return None
    seg = Image.open(seg_path).convert("L").resize((size, size), Image.NEAREST)
    arr = (np.array(seg) > 127).astype(np.float32)
    return arr

def norm_map(m):
    lo, hi = m.min(), m.max()
    if hi > lo:
        return (m - lo) / (hi - lo)
    return np.zeros_like(m)


def binarize_otsu(attn_map, closing_size=3):
    norm = norm_map(attn_map)
    try:
        thresh = threshold_otsu(norm)
    except Exception:
        thresh = 0.5
    binary = norm >= thresh
    if closing_size > 0:
        struct = np.ones((closing_size, closing_size), dtype=bool)
        binary = binary_closing(binary, structure=struct)
    return binary.astype(np.float32)


def compute_iou(attn_map, seg_mask):
    pred         = binarize_otsu(attn_map)
    intersection = (pred * seg_mask).sum()
    union        = np.clip(pred + seg_mask, 0, 1).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)

def render_overlay(pil_img, attn_map, res=128):
    gray  = np.array(pil_img.convert("L").resize((res, res), Image.BILINEAR)) / 255.0
    attn_r = np.array(
        Image.fromarray((norm_map(attn_map) * 255).astype(np.uint8))
        .resize((res, res), Image.BILINEAR)
    ) / 255.0
    heat  = plt.get_cmap("jet")(attn_r)[:, :, :3]
    gray3 = np.stack([gray] * 3, axis=-1)
    blend = (0.45 * gray3 + 0.55 * heat).clip(0, 1)
    return (blend * 255).astype(np.uint8)


def render_seg_overlay(pil_img, seg_mask, res=128):
    gray  = np.array(pil_img.convert("L").resize((res, res), Image.BILINEAR)) / 255.0
    gray3 = np.stack([gray] * 3, axis=-1)
    seg_r = np.array(
        Image.fromarray((seg_mask * 255).astype(np.uint8))
        .resize((res, res), Image.NEAREST)
    ) / 255.0
    overlay = gray3.copy()
    overlay[..., 1] = np.clip(gray3[..., 1] + 0.4 * seg_r, 0, 1)
    return (overlay * 255).astype(np.uint8)


def render_pred_seg_overlay(pil_img, attn_map, res=128):
    gray  = np.array(pil_img.convert("L").resize((res, res), Image.BILINEAR)) / 255.0
    gray3 = np.stack([gray] * 3, axis=-1)
    pred  = binarize_otsu(attn_map)
    pred_r = np.array(
        Image.fromarray((pred * 255).astype(np.uint8))
        .resize((res, res), Image.NEAREST)
    ) / 255.0
    overlay = gray3.copy()
    overlay[..., 0] = np.clip(gray3[..., 0] + 0.5 * pred_r, 0, 1)
    return (overlay * 255).astype(np.uint8)


def save_row_figure(pil_img, seg_mask, attn_maps, iou_val, out_path, res=128):
    summary_attn = attn_maps[:, :, TOK_SUMMARY]
    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.2, 2.5))

    axes[0].imshow(np.array(pil_img.resize((res, res))))
    axes[0].set_title("Real image", fontsize=7, fontweight="bold")
    axes[0].axis("off")

    if seg_mask is not None:
        axes[1].imshow(render_seg_overlay(pil_img, seg_mask, res=res))
        axes[1].set_title("Seg mask (GT)", fontsize=7)
    else:
        axes[1].imshow(np.zeros((res, res, 3), dtype=np.uint8))
        axes[1].set_title("Seg mask (missing)", fontsize=7, color="red")
    axes[1].axis("off")

    axes[2].imshow(render_pred_seg_overlay(pil_img, summary_attn, res=res))
    axes[2].set_title(f"Pred seg\n(IoU={iou_val:.3f})", fontsize=7)
    axes[2].axis("off")

    axes[3].imshow(render_overlay(pil_img, summary_attn, res=res))
    axes[3].set_title("Summary token\nheatmap", fontsize=7)
    axes[3].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

def load_model(lora_path, device):
    print(f"Loading checkpoint: {lora_path}")
    ckpt          = torch.load(lora_path, map_location="cpu", weights_only=False)
    lora_rank     = ckpt.get("lora_rank", 16)
    field_configs = ckpt["field_configs"]
    species2idx   = ckpt["species2idx"]

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer in lora_layers:
        layer.to(device)

    capture_state = CaptureState()
    n_patched = inject_capture_attention(unet, device, capture_state)
    print(f"  Patched {n_patched} cross-attention modules")

    for layer, sd in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(sd)

    cond_encoder = MetadataConditionEncoder(field_configs, hidden_dim=256, final_dim=768)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.to(device).eval()

    scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    return vae, unet, cond_encoder, scheduler, capture_state, species2idx

def run_forward_pass(vae, unet, cond_encoder, scheduler, capture_state,
                     pil_img, meta, noise_t, device):
    img_tensor = img_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.mean * 0.18215

    noise = torch.randn_like(latent)
    t     = torch.tensor([noise_t], device=device, dtype=torch.long)
    noisy = scheduler.add_noise(latent, noise, t)

    cond = cond_encoder(meta)

    capture_state.clear()
    with torch.no_grad():
        set_attention_context(unet, cond)
        capture_state.active = True
        unet(noisy, t, encoder_hidden_states=cond)
        capture_state.active = False

    return aggregate_attention(capture_state, out_size=ATTN_RES)

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate summary-token segmentation IoU on CUB val images.")
    ap.add_argument("--lora_path",   required=True,
                    help="Trained lora_cub.pth checkpoint")
    ap.add_argument("--csv_path",    default="data/cub_val.csv",
                    help="Val split CSV (default: data/cub_val.csv)")
    ap.add_argument("--cub_root",    required=True,
                    help="Root directory containing CUB_200_2011/ and segmentations/")
    ap.add_argument("--n_per_class", type=int, default=1,
                    help="Number of val images per species (default: 1, -1 = all)")
    ap.add_argument("--noise_t",     type=int, default=NOISE_T_DEFAULT,
                    help=f"Noise timestep (default: {NOISE_T_DEFAULT})")
    ap.add_argument("--output_dir",  default="results/attention_localization",
                    help="Directory to save results")
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"N per class: {args.n_per_class}")
    print(f"Noise t    : {args.noise_t}")
    vae, unet, cond_encoder, scheduler, capture_state, species2idx = \
        load_model(args.lora_path, device)

    print(f"\nLoading val CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path).reset_index(drop=True)
    df = df[df["species_name"].isin(species2idx)].reset_index(drop=True)
    print(f"  Val rows: {len(df)} across {df['species_name'].nunique()} species")
    
    rng = np.random.default_rng(args.seed)
    sampled_rows = []
    for sp_name in sorted(species2idx.keys()):
        sp_rows = df[df["species_name"] == sp_name]
        if len(sp_rows) == 0:
            print(f"  WARNING: no val rows for {sp_name}")
            continue
        if args.n_per_class == -1:
            chosen = sp_rows
        else:
            n = min(args.n_per_class, len(sp_rows))
            chosen = sp_rows.sample(n=n, random_state=int(rng.integers(0, 9999)))
        sampled_rows.append(chosen)

    df_eval = pd.concat(sampled_rows).reset_index(drop=True)
    print(f"\nEvaluating {len(df_eval)} images")

    records = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Images"):
        sp_name  = row["species_name"]
        sp_idx   = species2idx[sp_name]
        img_path = row["full_path"]
        img_id   = int(row["image_id"])

        if not os.path.exists(img_path):
            print(f"  SKIP (image not found): {img_path}")
            continue

        pil_img = Image.open(img_path).convert("RGB")

        meta = {
            "species":        torch.tensor([sp_idx],                     device=device),
            "throat_color":   torch.tensor([int(row["throat_color"])],   device=device),
            "forehead_color": torch.tensor([int(row["forehead_color"])], device=device),
            "belly_color":    torch.tensor([int(row["belly_color"])],    device=device),
            "nape_color":     torch.tensor([int(row["nape_color"])],     device=device),
        }

        attn_maps = run_forward_pass(
            vae, unet, cond_encoder, scheduler, capture_state,
            pil_img, meta, args.noise_t, device
        )

        seg_path = get_seg_path(img_path, args.cub_root)
        seg_mask = load_segmentation(seg_path, size=ATTN_RES)
        iou      = compute_iou(attn_maps[:, :, TOK_SUMMARY], seg_mask) \
                   if seg_mask is not None else float("nan")

        safe_name = sp_name.replace(" ", "_")
        fig_path  = os.path.join(fig_dir, f"{safe_name}_{img_id}.png")
        save_row_figure(pil_img, seg_mask, attn_maps, iou, fig_path)

        records.append({
            "image_id": img_id,
            "species":  sp_name,
            "img_path": img_path,
            "iou":      iou,
        })

    df_scores = pd.DataFrame(records)
    csv_path  = os.path.join(args.output_dir, "scores.csv")
    df_scores.to_csv(csv_path, index=False)
    print(f"\nPer-image scores {csv_path}")

    summary_rows = []
    for sp_name in sorted(species2idx.keys()):
        sp_df = df_scores[df_scores["species"] == sp_name]
        if sp_df.empty:
            continue
        summary_rows.append({
            "species":  sp_name,
            "iou_mean": round(sp_df["iou"].mean(), 4),
            "n_images": len(sp_df),
        })
    summary_rows.append({
        "species":  "OVERALL MEAN",
        "iou_mean": round(df_scores["iou"].mean(), 4),
        "n_images": len(df_scores),
    })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output_dir, "summary_scores.csv")
    df_summary.to_csv(summary_csv, index=False)

    print(f"\n{'='*50}")
    print("SEGMENTATION IoU RESULTS")
    print(f"{'='*50}")
    print(f"  Random baseline IoU ≈ 0.20  (bird area / image area)")
    for _, r in df_summary.iterrows():
        print(f"  {r['species']:<35}  IoU = {r['iou_mean']:.4f}  (n={r['n_images']})")
    print(f"\nSummary CSV {summary_csv}")
    print(f"Figures {fig_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
