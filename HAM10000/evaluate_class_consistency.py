import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


IMG_SIZE = 224
BATCH_SIZE = 64

DEFAULT_MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"

DEFAULT_CLASSES = [
    "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"
]

classify_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


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
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ])


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_convnext(checkpoint_path, model_name, device):
    print(f"Loading ConvNeXt model: {model_name}")

    ckpt = None
    classes = DEFAULT_CLASSES
    class2idx = {c: i for i, c in enumerate(classes)}

    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(ckpt, dict) and "classes" in ckpt:
            classes = ckpt["classes"]
            class2idx = ckpt.get("class2idx", {c: i for i, c in enumerate(classes)})

    num_classes = len(classes)

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )

    if checkpoint_path is not None:
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        state_dict = clean_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"Loaded HAM10000 checkpoint: {checkpoint_path}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        if isinstance(ckpt, dict):
            if "val_acc" in ckpt:
                print(f"Val acc : {ckpt['val_acc']:.4f}")
            if "bal_acc" in ckpt:
                print(f"Bal acc : {ckpt['bal_acc']:.4f}")
    else:
        print("WARNING: using ImageNet-pretrained weights only.")
        print("This is NOT a HAM10000 classifier unless you fine-tuned it.")

    model = model.to(device)
    model.eval()

    idx2class = {v: k for k, v in class2idx.items()}

    print(f"Classes: {classes}")

    return model, classes, class2idx, idx2class, ckpt


def classify(image_paths, model, device, idx2class, num_workers):
    loader = make_loader(image_paths, num_workers)
    preds = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            logits = model(batch)
            pred_indices = logits.argmax(dim=1).cpu().numpy()
            preds.extend([idx2class[i] for i in pred_indices])

    return np.array(preds)


def class_consistency(predicted, target_class):
    return (predicted == target_class).sum() / len(predicted) if len(predicted) > 0 else 0.0


def plot_bar(results, title, out_path):
    classes = sorted(results.keys())
    vals = [results[c] for c in classes]
    mean = np.mean(vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, vals)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(mean, linestyle="--", linewidth=1.2,
               alpha=0.8, label=f"Mean = {mean:.3f}")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Class consistency")
    ax.set_title(title)
    ax.legend(fontsize=10)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confusion(all_target, all_predicted, classes, title, out_path):
    n = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)

    for target, pred in zip(all_target, all_predicted):
        if target in cls_to_idx and pred in cls_to_idx:
            matrix[cls_to_idx[target], cls_to_idx[pred]] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Target class")
    ax.set_title(title + "\nRow-normalised: diagonal = class consistency")

    for i in range(n):
        for j in range(n):
            text_color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(
                j, i, f"{norm_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    plt.colorbar(im, ax=ax, label="Fraction of generated images")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--checkpoint", default="convnext_classifier_ham_melaware.pth",
                        help="Fine-tuned HAM10000 ConvNeXt checkpoint")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = os.path.join(args.generated_root, "metrics_convnext")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device         : {device}")
    print(f"Generated root : {args.generated_root}")
    print(f"Output dir     : {out_dir}")
    print()

    model, classes, class2idx, idx2class, ckpt = load_convnext(
        args.checkpoint,
        args.model_name,
        device,
    )

    consistency_results = {}
    all_target = []
    all_predicted = []
    summary_rows = []

    print(f"\nEvaluating {len(classes)} classes: {classes}\n")

    for dx_name in classes:
        gen_dir = os.path.join(args.generated_root, dx_name)

        if not os.path.exists(gen_dir):
            print(f"[{dx_name}] Skipping, folder not found: {gen_dir}")
            continue

        gen_paths = get_image_paths(gen_dir)

        if len(gen_paths) < 5:
            print(f"[{dx_name}] Skipping, too few images: {len(gen_paths)}")
            continue

        print(f"[{dx_name}] {len(gen_paths)} images...")

        preds = classify(
            gen_paths,
            model,
            device,
            idx2class,
            args.workers,
        )

        consistency = class_consistency(preds, dx_name)

        consistency_results[dx_name] = consistency
        all_target.extend([dx_name] * len(preds))
        all_predicted.extend(preds.tolist())

        print(f"consistency = {consistency:.3f}")

        summary_rows.append({
            "class": dx_name,
            "n_generated": len(gen_paths),
            "consistency": round(consistency, 3),
        })

    if not summary_rows:
        print("\nNo classes processed. Check generated_root structure.")
        return

    csv_path = os.path.join(out_dir, "class_consistency_convnext.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"\nCSV {csv_path}")

    mean_cons = np.mean([r["consistency"] for r in summary_rows])

    val_text = ""
    if isinstance(ckpt, dict):
        if "val_acc" in ckpt:
            val_text += f" | Val acc: {ckpt['val_acc']:.3f}"
        if "bal_acc" in ckpt:
            val_text += f" | Bal acc: {ckpt['bal_acc']:.3f}"

    plot_bar(
        consistency_results,
        title=(
            f"Class consistency — ConvNeXt — HAM10000\n"
            f"{args.model_name}{val_text} | Mean consistency: {mean_cons:.3f}"
        ),
        out_path=os.path.join(out_dir, "class_consistency_convnext.png"),
    )

    plot_confusion(
        all_target,
        all_predicted,
        classes,
        title=f"ConvNeXt — Generated vs Predicted — HAM10000",
        out_path=os.path.join(out_dir, "confusion_matrix_convnext.png"),
    )

    print(f"\n{'=' * 55}")
    print(f"CLASS CONSISTENCY — ConvNeXt")
    print(f"Model: {args.model_name}")
    print(f"{'=' * 55}")
    print(f"{'Class':<10} {'N':>6}  {'Consistency':>12}")
    print(f"{'-' * 32}")

    for row in summary_rows:
        print(f"{row['class']:<10} {row['n_generated']:>6}  {row['consistency']:>12.3f}")

    print(f"{'-' * 32}")
    print(f"{'Mean':<10} {'':>6}  {mean_cons:>12.3f}")
    print(f"{'=' * 55}")
    print(f"\nAll outputs in: {out_dir}/")


if __name__ == "__main__":
    main()