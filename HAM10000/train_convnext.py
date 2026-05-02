import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


TRAIN_CSV = "./data/HAM10000_metadata_train.csv"
VAL_CSV   = "./data/HAM10000_metadata_val.csv"
IMG_DIR   = "ham_lora"
SAVE_PATH = "convnext_classifier_ham_melaware.pth"

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 60
LR = 3e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 12
SEED = 42

MEL_BOOST = 1.5
MEL_LOSS_WEIGHT = 1.3
MEL_SCORE_WEIGHT = 0.2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, class2idx, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.class2idx = class2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.class2idx[row["dx"]]
        return image, label


def load_df(csv_path, img_dir):
    df = pd.read_csv(csv_path).dropna(subset=["dx"]).reset_index(drop=True)
    df["filepath"] = df["image_id"].astype(str).apply(
        lambda x: os.path.join(img_dir, x + ".jpg")
    )
    return df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_acc = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    per_class_acc = []
    for c in range(num_classes):
        mask = all_labels == c
        acc = (all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0
        per_class_acc.append(acc)

    return val_acc, bal_acc, per_class_acc, all_labels, all_preds


def plot_confusion(labels, preds, classes, save_path, bal_acc, val_acc, epoch, score, mel_acc):
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"ConvNeXt HAM10000 validation confusion matrix\n"
        f"score={score:.3f} | bal_acc={bal_acc:.3f} | mel_acc={mel_acc:.3f} | "
        f"val_acc={val_acc:.3f} | epoch={epoch}"
    )

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if norm_cm[i, j] > 0.6 else "black"
            ax.text(
                j, i, f"{norm_cm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if i == j else "normal",
            )

    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()

    out_path = save_path.replace(".pth", "_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--val_csv", default=VAL_CSV)
    parser.add_argument("--img_dir", default=IMG_DIR)
    parser.add_argument("--save_path", default=SAVE_PATH)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mel_boost", type=float, default=MEL_BOOST)
    parser.add_argument("--mel_loss_weight", type=float, default=MEL_LOSS_WEIGHT)
    parser.add_argument("--mel_score_weight", type=float, default=MEL_SCORE_WEIGHT)
    args = parser.parse_args()

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")

    train_df = load_df(args.train_csv, args.img_dir)
    val_df = load_df(args.val_csv, args.img_dir)

    classes = sorted(train_df["dx"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    if "mel" not in class2idx:
        raise ValueError("Class 'mel' not found in training classes.")

    mel_idx = class2idx["mel"]

    print(f"Classes: {classes}")
    print(f"Train images: {len(train_df)}")
    print(f"Val images:   {len(val_df)}")

    class_counts = train_df["dx"].value_counts().to_dict()
    print(f"Class counts: {class_counts}")
    print(f"Melanoma sampling boost: {args.mel_boost}")
    print(f"Melanoma loss weight:    {args.mel_loss_weight}")
    print(f"Melanoma score weight:   {args.mel_score_weight}")

    sample_weights = train_df["dx"].map(
        lambda x: (1.0 / class_counts[x]) * (args.mel_boost if x == "mel" else 1.0)
    ).values

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataset = HAM10000Dataset(train_df, args.img_dir, class2idx, train_transform)
    val_dataset = HAM10000Dataset(val_df, args.img_dir, class2idx, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    print(f"\nLoading model: {args.model_name}")

    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.15,
        drop_path_rate=0.10,
    ).to(device)

    class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    class_weights[mel_idx] = args.mel_loss_weight

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = 0.0
    best_bal_acc = 0.0
    best_val_acc = 0.0
    best_mel_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=False,
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        train_acc = train_correct / train_total
        train_loss = train_loss / train_total

        val_acc, bal_acc, per_class_acc, labels_np, preds_np = evaluate(
            model, val_loader, device, num_classes
        )

        mel_acc = per_class_acc[mel_idx]
        score = (1.0 - args.mel_score_weight) * bal_acc + args.mel_score_weight * mel_acc

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f} | "
            f"bal_acc={bal_acc:.3f} | "
            f"mel_acc={mel_acc:.3f} | "
            f"score={score:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        print(
            "  per-class: "
            + "  ".join(f"{classes[i]}={per_class_acc[i]:.2f}" for i in range(num_classes))
        )

        if score > best_score:
            best_score = score
            best_bal_acc = bal_acc
            best_val_acc = val_acc
            best_mel_acc = mel_acc
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model_name,
                "classes": classes,
                "class2idx": class2idx,
                "val_acc": float(val_acc),
                "bal_acc": float(bal_acc),
                "mel_acc": float(mel_acc),
                "score": float(score),
                "epoch": best_epoch,
                "img_size": IMG_SIZE,
                "mel_boost": float(args.mel_boost),
                "mel_loss_weight": float(args.mel_loss_weight),
                "mel_score_weight": float(args.mel_score_weight),
            }, args.save_path)

            print(
                f"  New best checkpoint saved: "
                f"score={score:.3f}, bal_acc={bal_acc:.3f}, "
                f"mel_acc={mel_acc:.3f}, val_acc={val_acc:.3f}"
            )

            plot_confusion(
                labels_np, preds_np, classes, args.save_path,
                bal_acc, val_acc, best_epoch, score, mel_acc
            )

        else:
            patience_counter += 1
            print(f"  No score improvement: {patience_counter}/{PATIENCE}")

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    print("\nTraining complete.")
    print(f"Best epoch:              {best_epoch}")
    print(f"Best score:              {best_score:.3f}")
    print(f"Best val accuracy:       {best_val_acc:.3f}")
    print(f"Best balanced accuracy:  {best_bal_acc:.3f}")
    print(f"Best melanoma accuracy:  {best_mel_acc:.3f}")
    print(f"Saved to: {args.save_path}")


if __name__ == "__main__":
    main()