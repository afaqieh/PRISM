import os
import numpy as np
import pandas as pd

DATASET_DIR = "./PlantVillage"
VAL_SPLIT   = 0.2
SEED        = 42

CONDITION_MAP = {
    "Pepper__bell___Bacterial_spot":                ("Pepper", "Bacterial_spot"),
    "Pepper__bell___healthy":                       ("Pepper", "healthy"),
    "Potato___Early_blight":                        ("Potato", "Early_blight"),
    "Potato___Late_blight":                         ("Potato", "Late_blight"),
    "Potato___healthy":                             ("Potato", "healthy"),
    "Tomato_Bacterial_spot":                        ("Tomato", "Bacterial_spot"),
    "Tomato_Early_blight":                          ("Tomato", "Early_blight"),
    "Tomato_Late_blight":                           ("Tomato", "Late_blight"),
    "Tomato_Leaf_Mold":                             ("Tomato", "Leaf_Mold"),
    "Tomato_Septoria_leaf_spot":                    ("Tomato", "Septoria_leaf_spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite":  ("Tomato", "Spider_mites"),
    "Tomato__Target_Spot":                          ("Tomato", "Target_Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":        ("Tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato__Tomato_mosaic_virus":                  ("Tomato", "mosaic_virus"),
    "Tomato_healthy":                               ("Tomato", "healthy"),
}


def is_image_file(fname: str) -> bool:
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def prettify_class_name(raw_name: str) -> str:
    name = raw_name.replace("___", " | ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")
    return name


def build_dataframe_from_folders(dataset_dir: str) -> pd.DataFrame:
    rows = []
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    def contains_images(folder):
        return any(
            is_image_file(f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        )

    for root, dirs, files in os.walk(dataset_dir):
        if contains_images(root):
            class_name = os.path.basename(root)
            if class_name not in CONDITION_MAP:
                print(f"  WARNING: unknown class folder '{class_name}' — skipping")
                continue
            plant, condition = CONDITION_MAP[class_name]
            for fname in files:
                if is_image_file(fname):
                    rows.append({
                        "full_path":    os.path.join(root, fname),
                        "class_name":   class_name,
                        "display_name": prettify_class_name(class_name),
                        "plant":        plant,
                        "condition":    condition,
                    })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No image files found.")
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, label_col: str, val_split: float, seed: int):
    train_parts, val_parts = [], []
    for cls in sorted(df[label_col].unique()):
        cls_df = df[df[label_col] == cls].sample(
            frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(round(len(cls_df) * val_split)))
        val_parts.append(cls_df.iloc[:n_val])
        train_parts.append(cls_df.iloc[n_val:])
    train_df = pd.concat(train_parts).sample(
        frac=1.0, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat(val_parts).sample(
        frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def main():
    os.makedirs("data", exist_ok=True)

    print(f"Building dataframe from {DATASET_DIR} ...")
    full_df = build_dataframe_from_folders(DATASET_DIR)
    print(f"Total images: {len(full_df)}")

    print(f"\nClasses ({full_df['class_name'].nunique()}):")
    for cls in sorted(full_df["class_name"].unique()):
        plant, condition = CONDITION_MAP[cls]
        n = (full_df["class_name"] == cls).sum()
        print(f"  [{n:5d}]  {cls:<45}  plant={plant}  condition={condition}")

    print(f"\nCreating {int((1-VAL_SPLIT)*100)}/{int(VAL_SPLIT*100)} stratified split (seed={SEED})...")
    train_df, val_df = stratified_split(full_df, "class_name", VAL_SPLIT, SEED)

    train_df.to_csv("data/plantvillage_train.csv", index=False)
    val_df.to_csv(  "data/plantvillage_val.csv",   index=False)

    print(f"\nSplit complete:")
    print(f"  Train: {len(train_df)} images data/plantvillage_train.csv")
    print(f"  Val:   {len(val_df)} images  data/plantvillage_val.csv")

    print(f"\nPer-class breakdown:")
    print(f"  {'Class':<45} {'Total':>6} {'Train':>6} {'Val':>6}")
    for cls in sorted(full_df["class_name"].unique()):
        total = (full_df["class_name"] == cls).sum()
        train = (train_df["class_name"] == cls).sum()
        val   = (val_df["class_name"] == cls).sum()
        print(f"  {cls:<45} {total:>6} {train:>6} {val:>6}")

    print(f"\nMetadata vocabulary:")
    plants     = sorted(full_df["plant"].unique())
    conditions = sorted(full_df["condition"].unique())
    print(f"  plant     ({len(plants)} options): {plants}")
    print(f"  condition ({len(conditions)} options): {conditions}")


if __name__ == "__main__":
    main()
