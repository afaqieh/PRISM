import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

IMG_SIZE   = 224
BATCH_SIZE = 64

DEFAULTS = {
    "dino_cache":         "dino_linear_classifier_cub.pth",
}

classify_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def make_loader(paths, num_workers=4):
    dataset = ImagePathDataset(paths, classify_transform)
    return DataLoader(
        dataset,
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
        print(f"\nERROR: DINOv2 linear probe weights not found at '{cache_path}'")
        print("Please run: python train_dino_linear_cub.py")
        exit(1)

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

    print(f"  Backbone: {dino_model_name}  Feature dim: {feat_dim}")
    print(f"  Val accuracy: {ckpt['val_acc']:.3f}  "
          f"Bal accuracy: {ckpt['bal_acc']:.3f}  "
          f"(epoch {ckpt['epoch']})")

    return backbone, head, ckpt["class2idx"], ckpt["val_acc"], dino_model_name


def classify_with_dino(image_paths, backbone, head, device, class2idx, num_workers=4):
    idx2class = {v: k for k, v in class2idx.items()}
    loader    = make_loader(image_paths, num_workers)
    preds     = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch        = batch.to(device)
            feats        = backbone(batch)
            pred_indices = head(feats).argmax(dim=1).cpu().numpy()
            preds.extend([idx2class[i] for i in pred_indices])

    return np.array(preds)

def compute_class_consistency(predicted_labels, target_class):
    correct = (predicted_labels == target_class).sum()
    total   = len(predicted_labels)
    return correct / total if total > 0 else 0.0

def plot_consistency_single(results, title, out_path, color="#2CA02C"):
    classes = sorted(results.keys())
    short   = [c.replace(" ", "_") for c in classes]
    vals    = [results[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(short, vals, color=color, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Class consistency (fraction correct)")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(all_target, all_predicted, classes, title, out_path):
    short      = [c.replace(" ", "_") for c in classes]
    n          = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    matrix     = np.zeros((n, n), dtype=int)

    for target, pred in zip(all_target, all_predicted):
        if target in cls_to_idx and pred in cls_to_idx:
            matrix[cls_to_idx[target], cls_to_idx[pred]] += 1

    row_sums    = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("Target species (used for generation)")
    ax.set_title(title + "\nRow-normalised: diagonal = class consistency")
    for i in range(n):
        for j in range(n):
            text_color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{norm_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color=text_color,
                    fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, label="Fraction of generated images")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_root",   required=True,
                        help="Root folder with subfolders per species containing generated .png files")
    parser.add_argument("--dino_cache",       default=None,
                        help=f"DINOv2 linear probe checkpoint (default: {DEFAULTS['dino_cache']})")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dino_cache        = args.dino_cache        or DEFAULTS["dino_cache"]

    out_dir = os.path.join(args.generated_root, "metrics_dino")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:           {device}")
    print(f"Generated root:   {args.generated_root}")
    print(f"Output dir:       {out_dir}\n")

    backbone, head, dino_class2idx, dino_val_acc, dino_model_name = \
        load_dino_linear(dino_cache, device)

    classes = sorted(dino_class2idx.keys())
    print(f"\nEvaluating {len(classes)} species: {classes}\n")

    dino_results   = {}

    all_target         = []
    all_dino_preds     = []
    summary_rows       = []

    for species_name in classes:
        gen_dir = os.path.join(args.generated_root,
                               species_name.replace(" ", "_"))
        if not os.path.exists(gen_dir):
            print(f"[{species_name}] Skipping — folder not found: {gen_dir}")
            continue

        gen_paths = get_image_paths(gen_dir)
        if len(gen_paths) < 5:
            print(f"[{species_name}] Skipping — too few images ({len(gen_paths)})")
            continue

        short = species_name.replace(" ", "_")
        print(f"[{short}] {len(gen_paths)} generated images")

        print(f" DINOv2 linear probe...")
        dino_preds  = classify_with_dino(
            gen_paths, backbone, head, device, dino_class2idx, args.workers)
        dino_cons   = compute_class_consistency(dino_preds, species_name)
        dino_results[species_name] = dino_cons

        all_target.extend([species_name] * len(dino_preds))
        all_dino_preds.extend(dino_preds.tolist())

        tag = (f"  DINOv2={dino_cons:.3f}")
        print(tag)

        summary_rows.append({
            "class":             species_name,
            "n_generated":       len(gen_paths),
            "consistency_dino":  round(dino_cons, 3)
        })

    if not summary_rows:
        print("\nNo species processed — check that generated_root has species subfolders.")
        return

    csv_path = os.path.join(out_dir, "class_consistency_dino.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nCSV saved {csv_path}")

    print("\nGenerating plots...")

    plot_consistency_single(
        dino_results,
        title=(f"Class consistency — DINOv2 linear probe ({dino_model_name})\n"
               f"val_acc={dino_val_acc:.3f} on real images  "
               f"| Fraction of generated images classified as their target species"),
        out_path=os.path.join(out_dir, "class_consistency_dino.png"),
        color="#2CA02C",
    )

    plot_confusion_matrix(
        all_target, all_dino_preds, classes,
        title=f"DINOv2 linear probe — Generated vs Predicted — CUB warblers",
        out_path=os.path.join(out_dir, "confusion_matrix_dino.png"),
    )

    print(f"\n{'='*65}")
    print(f"  CLASS CONSISTENCY — DINOv2 linear probe ({dino_model_name})")
    print(f"{'='*65}")

    header = f"{'Species':<30} {'DINOv2':>8}"
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        line = f"{row['class']:<30} {row['consistency_dino']:>8.3f}"
        print(line)

    mean_dino = np.mean([r["consistency_dino"] for r in summary_rows])
    print(f"\n{'Mean':>30} {mean_dino:>8.3f}")

    print(f"\nAll outputs in: {out_dir}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
