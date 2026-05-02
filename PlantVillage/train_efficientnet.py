import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from PIL import Image
from tqdm import tqdm

TRAIN_CSV  = "./data/plantvillage_train.csv"
VAL_CSV    = "./data/plantvillage_val.csv"
SAVE_PATH  = "efficientnet_classifier_plantvillage.pth"
EPOCHS     = 50
LR         = 1e-4
BATCH_SIZE = 32
PATIENCE   = 5
IMG_SIZE   = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.15, contrast=0.10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class PlantVillageClassifierDataset(Dataset):
    def __init__(self, df, class2idx, transform):
        self.df        = df.reset_index(drop=True)
        self.class2idx = class2idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["full_path"]).convert("RGB")
        img   = self.transform(img)
        label = self.class2idx[row["class_name"]]
        return img, label


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    def load_df(csv_path):
        df = pd.read_csv(csv_path).reset_index(drop=True)
        return df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    train_df = load_df(TRAIN_CSV)
    val_df   = load_df(VAL_CSV)

    classes     = sorted(train_df["class_name"].unique())
    class2idx   = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")

    class_counts = train_df["class_name"].value_counts().to_dict()
    weights      = train_df["class_name"].map(lambda x: 1.0 / class_counts[x]).tolist()
    sampler      = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(train_df),
        replacement=True,
    )

    train_dataset = PlantVillageClassifierDataset(train_df, class2idx, train_transform)
    val_dataset   = PlantVillageClassifierDataset(val_df,   class2idx, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False,  num_workers=4)

    class_weights = torch.tensor(
        [1.0 / class_counts[c] for c in classes], dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1}/{EPOCHS} train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(imgs)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total   += len(imgs)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_acc = train_correct / train_total
        val_acc   = (np.array(all_preds) == np.array(all_labels)).mean()
        bal_acc   = balanced_accuracy_score(all_labels, all_preds)
        scheduler.step()

        per_cls = []
        for c in range(num_classes):
            mask = np.array(all_labels) == c
            per_cls.append(
                (np.array(all_preds)[mask] == c).mean() if mask.sum() > 0 else 0.0)

        print(f"Epoch {epoch+1:2d}/{EPOCHS}  "
              f"loss={train_loss/train_total:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  bal_acc={bal_acc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print("  per-class: " +
              "  ".join(f"{classes[i].split('_')[-1]}={per_cls[i]:.2f}"
                        for i in range(num_classes)))

        if bal_acc > best_val_acc:
            best_val_acc     = bal_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes":     classes,
                "class2idx":   class2idx,
                "val_acc":     val_acc,
                "bal_acc":     bal_acc,
                "epoch":       epoch + 1,
            }, SAVE_PATH)
            print(f"  ✓ New best bal_acc={bal_acc:.3f} — saved to {SAVE_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best balanced accuracy: {best_val_acc:.3f}")
    print(f"Weights saved to: {SAVE_PATH}")

    print("\nGenerating confusion matrix on val set using best checkpoint...")
    ckpt = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    final_preds, final_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    n           = num_classes
    matrix      = np.zeros((n, n), dtype=int)
    for t, p in zip(final_labels, final_preds):
        matrix[t, p] += 1
    row_sums    = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.35), max(10, n * 0.35)))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        "EfficientNet-B0 evaluator — confusion matrix on real PlantVillage val set\n"
        f"Row-normalised  |  bal_acc={ckpt['bal_acc']:.3f}  "
        f"val_acc={ckpt['val_acc']:.3f}  epoch={ckpt['epoch']}"
    )
    for i in range(n):
        for j in range(n):
            text_color = "white" if norm_matrix[i, j] > 0.6 else "black"
            ax.text(j, i, f"{norm_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color=text_color,
                    fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, label="Fraction of images")
    plt.tight_layout()
    plot_path = SAVE_PATH.replace(".pth", "_confusion_matrix.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix {plot_path}")
    print(f"\nNow run evaluate_metrics_plantvillage.py — it will load these weights automatically.")


if __name__ == "__main__":
    main()
