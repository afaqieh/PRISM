import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from metadata_conditioning import MetadataConditionEncoder
from ..lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
CSV_PATH   = "./data/awa2_train.csv"


def set_ctx(unet, ctx):
    for _, m in unet.named_modules():
        if hasattr(m, "attn2"):
            m.attn2.metadata_context = ctx

def get_meta(df, class_name, class2idx, attr_names, overrides, device):
    rows = df[df["class_name"] == class_name]
    if len(rows) == 0:
        raise ValueError(
            f"Class '{class_name}' not found. "
            f"Available: {sorted(df['class_name'].unique())}"
        )
    row  = rows.iloc[0]
    meta = {"class": torch.tensor([class2idx[class_name]], device=device)}
    for attr in attr_names:
        val = overrides.get(attr, int(row[attr]))
        meta[attr] = torch.tensor([val], device=device)
    active = [a for a in attr_names if meta[a].item() == 1]
    tag    = f"  {class_name} {overrides if overrides else '(natural)'}: {active}"
    print(tag)
    return meta

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate(unet, vae, scheduler, cond, device, seed, guidance, ddim_steps):
    scheduler.set_timesteps(ddim_steps)
    set_seed(seed)
    uncond  = torch.zeros(1, 77, 768, device=device)
    latents = torch.randn(1, 4, 64, 64, device=device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            set_ctx(unet, uncond)
            nu = unet(latents, t, encoder_hidden_states=uncond).sample
            set_ctx(unet, cond)
            nc = unet(latents, t, encoder_hidden_states=cond).sample
            latents = scheduler.step(
                nu + guidance * (nc - nu), t, latents
            ).prev_sample
        img = vae.decode(latents / 0.18215).sample
    img = (img.clamp(-1, 1) + 1) / 2
    img = img[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((img * 255).astype("uint8"))

def load_font(size, bold=False):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def assemble_figure(cells, row_labels, col_labels, mod_tags, out_path,
                    img_size=256, gap=12, label_w=90, hdr_h=36):
    n_rows  = len(cells)
    n_cols  = 2
    total_w = label_w + n_cols * img_size + (n_cols + 1) * gap
    total_h = hdr_h   + n_rows * img_size + (n_rows + 1) * gap

    C_BG    = (255, 255, 255)
    C_TEXT  = (20,  20,  20)
    C_TAG   = (255, 255, 255)
    C_TAG_BG= (0,   0,   0)

    canvas = Image.new("RGB", (total_w, total_h), C_BG)
    draw   = ImageDraw.Draw(canvas)

    f_hdr   = load_font(13, bold=True)
    f_label = load_font(12, bold=False)
    f_tag   = load_font(10, bold=True)

    for ci, title in enumerate(col_labels):
        x_img = label_w + gap + ci * (img_size + gap)
        bb    = draw.textbbox((0, 0), title, font=f_hdr)
        tw    = bb[2] - bb[0]
        draw.text(
            (x_img + (img_size - tw) // 2, (hdr_h - (bb[3] - bb[1])) // 2),
            title, fill=C_TEXT, font=f_hdr
        )

    for ri, (row_imgs, row_label, tag) in enumerate(zip(cells, row_labels, mod_tags)):
        y_img = hdr_h + gap + ri * (img_size + gap)
        bb  = draw.textbbox((0, 0), row_label, font=f_label)
        tw  = bb[2] - bb[0]
        th  = bb[3] - bb[1]
        draw.text(
            (label_w - tw - 6, y_img + (img_size - th) // 2),
            row_label, fill=C_TEXT, font=f_label
        )

        for ci, img in enumerate(row_imgs):
            x_img_left = label_w + gap + ci * (img_size + gap)
            img_r      = img.resize((img_size, img_size), Image.LANCZOS)
            canvas.paste(img_r, (x_img_left, y_img))
            if ci == 1 and tag:
                pad_tag = 4
                bb_t    = draw.textbbox((0, 0), tag, font=f_tag)
                tw_t    = bb_t[2] - bb_t[0]
                th_t    = bb_t[3] - bb_t[1]
                rx0 = x_img_left + img_size - tw_t - 2 * pad_tag - 6
                ry0 = y_img      + img_size - th_t - 2 * pad_tag - 6
                rx1 = x_img_left + img_size - 6
                ry1 = y_img      + img_size - 6
                draw.rectangle([rx0, ry0, rx1, ry1], fill=C_TAG_BG)
                draw.text((rx0 + pad_tag, ry0 + pad_tag), tag,
                          fill=C_TAG, font=f_tag)

    canvas.save(out_path, dpi=(300, 300))
    print(f"Figure saved {out_path}  ({canvas.width}×{canvas.height}px)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path",  required=True,
                        help="Path to AWA2 Stage 1 checkpoint (.pth)")
    parser.add_argument("--csv",        default=CSV_PATH)
    parser.add_argument("--out_dir",    default="results/attribute_transfer")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed — change to get a different figure")
    parser.add_argument("--guidance",   type=float, default=7.5)
    parser.add_argument("--ddim_steps", type=int,   default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Seed: {args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading checkpoint: {args.lora_path}")
    ckpt          = torch.load(args.lora_path, map_location="cpu", weights_only=False)
    class2idx     = ckpt["class2idx"]
    attr_names    = ckpt["attr_names"]
    field_configs = ckpt["field_configs"]
    lora_rank     = ckpt.get("lora_rank", 16)
    print("Loading models...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    lora_layers = apply_lora_to_unet(unet, r=lora_rank)
    for layer in lora_layers:
        layer.to(device)
    inject_metadata_into_attention(unet, device)
    for layer, state in zip(lora_layers, ckpt["lora_layers"]):
        layer.load_state_dict(state)

    cond_encoder = MetadataConditionEncoder(field_configs, hidden_dim=256, final_dim=768)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    cond_encoder.to(device).eval()

    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    df = pd.read_csv(args.csv)

    pairs = [
        ("horse",  {}, {"horns": 1},            "Horse",  "horns = 1"),
        ("rabbit", {}, {"big": 1, "small": 0}, "Rabbit", "big = 1"),
    ]

    print("\nBuilding conditioning vectors...")
    conds = []
    for class_name, nat_ov, mod_ov, _, _ in pairs:
        meta_nat = get_meta(df, class_name, class2idx, attr_names, nat_ov, device)
        meta_mod = get_meta(df, class_name, class2idx, attr_names, mod_ov, device)
        with torch.no_grad():
            cond_nat = cond_encoder(meta_nat)
            cond_mod = cond_encoder(meta_mod)
        conds.append((cond_nat, cond_mod))

    print(f"\nGenerating 6 images (seed={args.seed})...")
    cells      = []
    row_labels = []

    for i, ((cond_nat, cond_mod), (_, _, _, row_label, mod_label)) in \
            enumerate(zip(conds, pairs)):

        print(f"\n  [{i*2+1}/6] {pairs[i][0]} natural...", end=" ", flush=True)
        img_nat = generate(unet, vae, scheduler, cond_nat, device,
                           args.seed, args.guidance, args.ddim_steps)
        img_nat.save(os.path.join(args.out_dir,
                                  f"{pairs[i][0]}_natural_seed{args.seed}.png"))
        print("done")

        print(f"  [{i*2+2}/6] {pairs[i][0]} modified ({mod_label.replace(chr(10), ' ')})...",
              end=" ", flush=True)
        img_mod = generate(unet, vae, scheduler, cond_mod, device,
                           args.seed, args.guidance, args.ddim_steps)
        img_mod.save(os.path.join(args.out_dir,
                                  f"{pairs[i][0]}_modified_seed{args.seed}.png"))
        print("done")

        cells.append([img_nat, img_mod])
        row_labels.append(row_label)

    mod_tags  = [p[4] for p in pairs]
    col_labels = ["Natural", "Modified"]
    fig_path   = os.path.join(args.out_dir, f"attribute_transfer_seed{args.seed}.png")
    assemble_figure(cells, row_labels, col_labels, mod_tags, fig_path)
    print(f"\nDone. All outputs {args.out_dir}/")


if __name__ == "__main__":
    main()
