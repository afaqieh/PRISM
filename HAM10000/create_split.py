import os
import numpy as np
import pandas as pd

CSV_PATH  = "/storage/homefs/af24h089/MSGAI/Final/data/HAM10000_metadata.csv"
VAL_FRAC  = 0.2
SEED      = 42
CLASSES   = sorted(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])

def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(CSV_PATH).dropna(
        subset=["dx", "localization", "sex", "age"]
    ).reset_index(drop=True)
    df = df[df["dx"].isin(CLASSES)].reset_index(drop=True)

    print(f"Total images: {len(df)}")

    train_indices, val_indices = [], []

    for cls in CLASSES:
        cls_indices = df[df["dx"] == cls].index.tolist()
        cls_indices = list(rng.permutation(cls_indices))
        n_val       = max(1, int(len(cls_indices) * VAL_FRAC))
        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df   = df.loc[val_indices].reset_index(drop=True)

    train_path = "data/HAM10000_metadata_train.csv"
    val_path   = "data/HAM10000_metadata_val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,    index=False)

    print(f"\nSplit created (seed={SEED}, val_frac={VAL_FRAC})")
    print(f"  Train: {len(train_df)} images {train_path}")
    print(f"  Val:   {len(val_df)} images   {val_path}")
    print(f"\nPer-class breakdown:")
    print(f"{'Class':<10} {'Total':>8} {'Train':>8} {'Val':>8}")
    for cls in CLASSES:
        n_total = (df["dx"] == cls).sum()
        n_train = (train_df["dx"] == cls).sum()
        n_val   = (val_df["dx"]   == cls).sum()
        print(f"{cls:<10} {n_total:>8} {n_train:>8} {n_val:>8}")

if __name__ == "__main__":
    main()
