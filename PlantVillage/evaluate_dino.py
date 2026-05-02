import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

IMG_SIZE   = 224
BATCH_SIZE = 64
DEFAULT_DINO_CACHE = "dino_linear_classifier_plantvillage.pth"

classify_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def prettify_class_name(raw_name: str) -> str:
    name = raw_name.replace("___", " | ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")
    return name

class ImagePathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return classify_transform(img)


def make_loader(paths, num_workers):
    return DataLoader(
        ImagePathDataset(paths),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def get_image_paths(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

def load_dino_linear(cache_path, device):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"DINOv2 linear probe not found at '{cache_path}'.\n"
            f"Run: python train_dino_linear_plantvillage.py"
        )

    print(f"Loading DINOv2 linear probe from {cache_path}...")
    ckpt = torch.load(cache_path, map_location=device, weights_only=False)

    dino_model_name = ckpt.get("dino_model_name", "dinov2_vitb14")
    feat_dim        = ckpt.get("feat_dim", 768)
    num_classes     = len(ckpt["classes"])

    backbone = torch.hub.load("facebookresearch/dinov2", dino_model_name)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(device)

    head = nn.Linear(feat_dim, num_classes).to(device)
    head.load_state_dict(ckpt["linear_head_state"])
    head.eval()

    print(f"  Backbone : {dino_model_name}  (feat_dim={feat_dim})")
    print(f"  Classes  : {num_classes}")
    print(f"  Val acc  : {ckpt['val_acc']:.3f}   Bal acc: {ckpt['bal_acc']:.3f}   (epoch {ckpt['epoch']})")

    return backbone, head, ckpt, dino_model_name

def classify(image_paths, backbone, head, device, idx2class, num_workers):
    loader = make_loader(image_paths, num_workers)
    preds  = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch        = batch.to(device)
            feats        = backbone(batch)
            pred_indices = head(feats).argmax(dim=1).cpu().numpy()
            preds.extend([idx2class[i] for i in pred_indices])
    return np.array(preds)


def class_consistency(predicted, target_class):
    return (predicted == target_class).sum() / len(predicted) if len(predicted) > 0 else 0.0

def plot_bar(results, display_names, title, out_path):
    classes = sorted(results.keys())
    labels  = [display_names.get(c, prettify_class_name(c)) for c in classes]
    vals    = [results[c] for c in classes]
    mean    = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.55), 6))
    bars = ax.bar(labels, vals, color="#2CA02C", alpha=0.85)
    ax.axhline(1.0,  color="black",  linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(mean, color="#E05C00", linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Mean = {mean:.3f}")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Class consistency (fraction correct)")
    ax.set_title(title)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion(all_target, all_predicted, classes, display_names, title, out_path):
    n          = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    matrix     = np.zeros((n, n), dtype=int)

    for target, pred in zip(all_target, all_predicted):
        if target in cls_to_idx and pred in cls_to_idx:
            matrix[cls_to_idx[target], cls_to_idx[pred]] += 1

    row_sums    = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    short = [display_names.get(c, prettify_class_name(c)) for c in classes]
    fig_size = max(12, n * 0.35)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Target class (used for generation)")
    ax.set_title(title + "\nRow-normalised: diagonal = class consistency")
    for i in range(n):
        for j in range(n):
            text_color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{norm_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color=text_color,
                    fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, label="Fraction of generated images")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_root", required=True,
                        help="Root folder — subfolders named after each class containing generated images")
    parser.add_argument("--dino_cache",     default=DEFAULT_DINO_CACHE,
                        help=f"DINOv2 linear probe checkpoint (default: {DEFAULT_DINO_CACHE})")
    parser.add_argument("--workers",        type=int, default=4)
    args = parser.parse_args()

    out_dir = os.path.join(args.generated_root, "metrics_dino")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    print(f"Generated root : {args.generated_root}")
    print(f"Output dir     : {out_dir}\n")

    backbone, head, ckpt, dino_model_name = load_dino_linear(args.dino_cache, device)

    classes       = ckpt["classes"]
    class2idx     = ckpt["class2idx"]
    idx2class     = {v: k for k, v in class2idx.items()}
    display_names = ckpt.get("display_names",
                             {c: prettify_class_name(c) for c in classes})

    print(f"\nEvaluating {len(classes)} classes\n")
    consistency_results = {}
    all_target          = []
    all_predicted       = []
    summary_rows        = []

    for class_name in classes:
        gen_dir = os.path.join(args.generated_root, class_name)
        if not os.path.exists(gen_dir):
            print(f"[{display_names.get(class_name, class_name)}] Skipping — folder not found: {gen_dir}")
            continue

        gen_paths = get_image_paths(gen_dir)
        if len(gen_paths) < 5:
            print(f"[{display_names.get(class_name, class_name)}] Skipping — too few images ({len(gen_paths)})")
            continue

        disp = display_names.get(class_name, prettify_class_name(class_name))
        print(f"[{disp}] {len(gen_paths)} images...")

        preds       = classify(gen_paths, backbone, head, device, idx2class, args.workers)
        consistency = class_consistency(preds, class_name)

        consistency_results[class_name] = consistency
        all_target.extend([class_name] * len(preds))
        all_predicted.extend(preds.tolist())

        print(f"  consistency = {consistency:.3f}")

        summary_rows.append({
            "class":        class_name,
            "display_name": disp,
            "n_generated":  len(gen_paths),
            "consistency":  round(consistency, 3),
        })

    if not summary_rows:
        print("\nNo classes processed — check that generated_root has class subfolders.")
        return

    csv_path = os.path.join(out_dir, "class_consistency_dino.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nCSV {csv_path}")

    mean_cons = np.mean([r["consistency"] for r in summary_rows])

    plot_bar(
        consistency_results, display_names,
        title=(f"Class consistency — DINOv2 linear probe ({dino_model_name}) — PlantVillage\n"
               f"Val acc on real images: {ckpt['val_acc']:.3f}  "
               f"|  Mean consistency: {mean_cons:.3f}"),
        out_path=os.path.join(out_dir, "class_consistency_dino.png"),
    )

    plot_confusion(
        all_target, all_predicted,
        classes, display_names,
        title=f"DINOv2 linear probe ({dino_model_name}) — Generated vs Predicted — PlantVillage",
        out_path=os.path.join(out_dir, "confusion_matrix_dino.png"),
    )

    print(f"\n{'='*60}")
    print(f"  CLASS CONSISTENCY — DINOv2 ({dino_model_name})")
    print(f"  Val acc on real images: {ckpt['val_acc']:.3f}  "
          f"Bal acc: {ckpt['bal_acc']:.3f}")
    print(f"{'='*60}")
    print(f"{'Class':<40} {'N':>5}  {'Consist':>8}")
    print(f"{'-'*60}")
    for row in summary_rows:
        print(f"{row['display_name']:<40} {row['n_generated']:>5}  {row['consistency']:>8.3f}")
    print(f"{'-'*60}")
    print(f"{'Mean':<40} {'':>5}  {mean_cons:>8.3f}")
    print(f"{'='*60}")
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
