import os
import argparse
import random
import numpy as np
import pandas as pd


SELECTED_ATTRS = [
    "black", "white", "blue", "brown", "gray", "orange", "red", "yellow",
    "patches", "spots", "stripes", "furry", "hairless", "toughskin",
    "big", "small", "bulbous", "lean",
    "flippers", "hands", "hooves", "pads", "paws",
    "longleg", "longneck", "tail", "horns", "claws", "tusks",
    "bipedal", "quadrapedal",
]


def load_awa2_attributes(awa2_root: str):
    classes_path    = os.path.join(awa2_root, "classes.txt")
    predicates_path = os.path.join(awa2_root, "predicates.txt")
    matrix_path     = os.path.join(awa2_root, "predicate-matrix-binary.txt")

    classes = []
    with open(classes_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            classes.append(parts[1].strip())

    predicates = []
    with open(predicates_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            predicates.append(parts[1].strip())

    matrix = np.loadtxt(matrix_path, dtype=int)

    return classes, predicates, matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awa2_root",  default="./Animals_with_Attributes2",
                    help="Path to AWA2 root (contains JPEGImages/, classes.txt, etc.)")
    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out_dir",    type=str,   default="./data")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    images_root = os.path.join(args.awa2_root, "JPEGImages")
    classes, predicates, matrix = load_awa2_attributes(args.awa2_root)

    pred_lower  = [p.lower() for p in predicates]
    attr_indices = []
    for attr in SELECTED_ATTRS:
        if attr.lower() not in pred_lower:
            raise ValueError(f"Attribute '{attr}' not found in predicates.txt. "
                             f"Available: {predicates}")
        attr_indices.append(pred_lower.index(attr.lower()))

    print(f"Selected attributes: {SELECTED_ATTRS}")
    print(f"Attribute column indices: {attr_indices}")

    rows_train, rows_val = [], []

    for cls_idx, cls_name in enumerate(classes):
        folder_name = cls_name.replace(" ", "+")
        cls_dir     = os.path.join(images_root, folder_name)

        if not os.path.isdir(cls_dir):
            folder_name = cls_name.replace(" ", "_")
            cls_dir     = os.path.join(images_root, folder_name)

        if not os.path.isdir(cls_dir):
            print(f"  WARNING: folder not found for class '{cls_name}', skipping.")
            continue

        images = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(images) == 0:
            print(f"  WARNING: no images found for class '{cls_name}', skipping.")
            continue

        attr_vals = [int(matrix[cls_idx, j]) for j in attr_indices]

        random.shuffle(images)
        n_val   = max(1, int(len(images) * args.val_frac))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            for img_path in img_list:
                row = {"full_path": img_path, "class_name": cls_name}
                for attr_name, val in zip(SELECTED_ATTRS, attr_vals):
                    row[attr_name] = val
                if split == "train":
                    rows_train.append(row)
                else:
                    rows_val.append(row)

        print(f"  {cls_name:<30}: {len(train_imgs)} train / {len(val_imgs)} val | "
              f"attrs={attr_vals}")

    df_train = pd.DataFrame(rows_train)
    df_val   = pd.DataFrame(rows_val)

    train_path = os.path.join(args.out_dir, "awa2_train.csv")
    val_path   = os.path.join(args.out_dir, "awa2_val.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path,   index=False)

    print(f"\nSaved {len(df_train)} train rows -> {train_path}")
    print(f"Saved {len(df_val)} val rows   -> {val_path}")
    print(f"\nClasses: {df_train['class_name'].nunique()}")
    print(f"Columns: {list(df_train.columns)}")


if __name__ == "__main__":
    main()
