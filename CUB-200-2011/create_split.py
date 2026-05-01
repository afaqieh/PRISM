import os
import numpy as np
import pandas as pd
from collections import defaultdict

CUB_ROOT = "CUB-200-2011/CUB_200_2011"
VAL_FRAC = 0.2
SEED     = 42

SELECTED_SPECIES_IDS = [183, 184, 104, 159, 169, 163, 179, 162, 175, 161, 166, 176, 170, 168, 182]

ATTR_GROUPS = {
    "throat_color":   (121, 135),
    "forehead_color": (153, 167),
    "belly_color":    (198, 212),
    "nape_color":     (183, 197),
}


def load_dominant_attributes(attr_file, attr_groups, image_ids_set):
    print(f"  Parsing attribute file (this takes ~30s for 3.6M lines)...")

    data = defaultdict(lambda: defaultdict(list))

    with open(attr_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            img_id    = int(parts[0])
            attr_id   = int(parts[1])
            is_present = int(parts[2])
            certainty  = int(parts[3])

            if img_id not in image_ids_set:
                continue

            for group_name, (lo, hi) in attr_groups.items():
                if lo <= attr_id <= hi:
                    idx_in_group = attr_id - lo
                    data[img_id][group_name].append((idx_in_group, certainty, is_present))

    result = {}
    for img_id in image_ids_set:
        result[img_id] = {}
        for group_name in attr_groups:
            entries = data[img_id].get(group_name, [])
            present = [(idx, cert) for idx, cert, pres in entries if pres == 1]
            if present:
                dominant = max(present, key=lambda x: x[1])[0]
            else:
                dominant = 0
            result[img_id][group_name] = dominant

    return result


def main():
    rng = np.random.default_rng(SEED)

    os.makedirs("data", exist_ok=True)

    classes_df = pd.read_csv(
        os.path.join(CUB_ROOT, "classes.txt"),
        sep=" ", header=None, names=["class_id", "class_name"]
    )
    selected_set = set(SELECTED_SPECIES_IDS)
    classes_df   = classes_df[classes_df["class_id"].isin(selected_set)]
    id_to_name   = dict(zip(classes_df["class_id"], classes_df["class_name"]))

    print(f"Selected {len(classes_df)} species:")
    for cid, name in sorted(id_to_name.items()):
        clean = name.split(".", 1)[1].replace("_", " ")
        id_to_name[cid] = clean
        print(f"  [{cid}] {clean}")

    images_df = pd.read_csv(
        os.path.join(CUB_ROOT, "images.txt"),
        sep=" ", header=None, names=["image_id", "filepath"]
    )
    labels_df = pd.read_csv(
        os.path.join(CUB_ROOT, "image_class_labels.txt"),
        sep=" ", header=None, names=["image_id", "class_id"]
    )
    df = images_df.merge(labels_df, on="image_id")
    df = df[df["class_id"].isin(selected_set)].reset_index(drop=True)
    df["species_name"] = df["class_id"].map(id_to_name)
    df["full_path"]    = df["filepath"].apply(
        lambda p: os.path.join(CUB_ROOT, "images", p)
    )
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"\nTotal images found: {len(df)}")

    print("\nLoading attributes...")
    image_ids_set = set(df["image_id"].tolist())
    attr_file     = os.path.join(CUB_ROOT, "attributes", "image_attribute_labels.txt")
    dominant_attrs = load_dominant_attributes(attr_file, ATTR_GROUPS, image_ids_set)

    for group in ATTR_GROUPS:
        df[group] = df["image_id"].map(lambda iid: dominant_attrs[iid][group])

    print("\nCreating 80/20 stratified split...")
    train_rows, val_rows = [], []

    for species in sorted(df["species_name"].unique()):
        rows  = df[df["species_name"] == species]
        idxs  = list(rng.permutation(rows.index.tolist()))
        n_val = max(1, int(len(idxs) * VAL_FRAC))
        val_rows.extend(idxs[:n_val])
        train_rows.extend(idxs[n_val:])

    train_df = df.loc[train_rows].reset_index(drop=True)
    val_df   = df.loc[val_rows].reset_index(drop=True)

    train_df.to_csv("data/cub_train.csv", index=False)
    val_df.to_csv("data/cub_val.csv",   index=False)

    print(f"\nSplit created (seed={SEED}, val_frac={VAL_FRAC})")
    print(f"  Train: {len(train_df)} images data/cub_train.csv")
    print(f"  Val:   {len(val_df)} images   data/cub_val.csv")
    print(f"\nPer-species breakdown:")
    print(f"{'Species':<35} {'Total':>6} {'Train':>6} {'Val':>6}")
    for species in sorted(df["species_name"].unique()):
        total = (df["species_name"] == species).sum()
        train = (train_df["species_name"] == species).sum()
        val   = (val_df["species_name"] == species).sum()
        print(f"  {species:<33} {total:>6} {train:>6} {val:>6}")

    print(f"\nAttribute vocab sizes:")
    for group, (lo, hi) in ATTR_GROUPS.items():
        print(f"  {group:<18}: {hi - lo + 1} options (attrs {lo}-{hi})")


if __name__ == "__main__":
    main()
