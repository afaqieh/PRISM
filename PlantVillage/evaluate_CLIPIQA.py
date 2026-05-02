import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


DEFAULT_CSV = "./data/plantvillage_train.csv"
DEFAULT_PLANTVILLAGE_ROOT = "./PlantVillage"
BATCH_SIZE = 32


iqa_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class PathDataset(data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_class_to_real_folder(plantvillage_root: str, class_names: list) -> dict:
    try:
        all_folders = [
            d for d in os.listdir(plantvillage_root)
            if os.path.isdir(os.path.join(plantvillage_root, d))
        ]
    except FileNotFoundError:
        raise FileNotFoundError(f"PlantVillage root not found: {plantvillage_root}")

    folder_set = set(all_folders)
    mapping = {}

    for cls in class_names:
        if cls in folder_set:
            mapping[cls] = os.path.join(plantvillage_root, cls)
        else:
            print(f"WARNING: No folder found for class '{cls}'")

    print(f"Matched {len(mapping)}/{len(class_names)} classes.")
    return mapping


def get_image_paths(folder: str) -> list:
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])


def build_clipiqa(device: str):
    try:
        import pyiqa
    except ImportError:
        raise ImportError("pyiqa is not installed. Run: pip install pyiqa")

    metric = pyiqa.create_metric("clipiqa", device=device)
    print("Loaded CLIPIQA")
    return metric


def compute_clipiqa(image_paths: list, clipiqa_metric, device: str, desc: str = "") -> float:
    values = []

    loader = data.DataLoader(
        PathDataset(image_paths, iqa_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    for batch in tqdm(loader, desc=f"CLIPIQA {desc}", leave=False):
        batch = batch.to(device)

        with torch.no_grad():
            scores = clipiqa_metric(batch)

        values.extend(scores.detach().cpu().numpy().reshape(-1).tolist())

    return float(np.mean(values)) if values else float("nan")


def build_results_table(results: dict) -> pd.DataFrame:
    rows = []

    for cls in sorted(results.keys()):
        real_score = results[cls]["clipiqa_real"]
        gen_score = results[cls]["clipiqa_gen"]
        delta = gen_score - real_score

        rows.append({
            "class": cls.replace("_", " "),
            "clipiqa_real": round(real_score, 4),
            "clipiqa_gen": round(gen_score, 4),
            "clipiqa_delta": round(delta, 4),
        })

    df = pd.DataFrame(rows)

    mean_row = {
        "class": "MEAN",
        "clipiqa_real": round(df["clipiqa_real"].mean(), 4),
        "clipiqa_gen": round(df["clipiqa_gen"].mean(), 4),
        "clipiqa_delta": round(df["clipiqa_delta"].mean(), 4),
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df


def print_table(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("CLIPIQA REAL vs GENERATED COMPARISON — PLANTVILLAGE")
    print("=" * 75)
    print(f"{'Class':<40} {'Real':>10} {'Generated':>12} {'Delta':>10}")
    print("-" * 75)

    for _, row in df.iterrows():
        if row["class"] == "MEAN":
            print("-" * 75)

        print(
            f"{row['class']:<40} "
            f"{row['clipiqa_real']:>10.4f} "
            f"{row['clipiqa_gen']:>12.4f} "
            f"{row['clipiqa_delta']:>+10.4f}"
        )

    print("=" * 75)
    print("Delta = generated CLIPIQA - real CLIPIQA")
    print("Positive delta means generated images scored higher.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIPIQA on real and generated PlantVillage images."
    )

    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--plantvillage_root", default=DEFAULT_PLANTVILLAGE_ROOT)
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--max_real", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=None)

    args = parser.parse_args()

    out_dir = os.path.join(args.generated_root, "clipiqa_metrics")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading class list from {args.csv_path}...")
    df_csv = pd.read_csv(args.csv_path)
    class_list = sorted(df_csv["class_name"].unique().tolist())

    print(f"Found {len(class_list)} classes:")
    for cls in class_list:
        print(f"  {cls}")
    print()

    print("Mapping classes to PlantVillage folders...")
    class_to_real_folder = build_class_to_real_folder(
        args.plantvillage_root,
        class_list
    )

    print("Initialising CLIPIQA...")
    clipiqa_metric = build_clipiqa(device)

    results = {}

    for cls in class_list:
        print(f"\n{'=' * 60}")
        print(cls.replace("_", " "))
        print(f"{'=' * 60}")

        if cls not in class_to_real_folder:
            print("SKIP: no real folder found.")
            continue

        real_folder = class_to_real_folder[cls]
        real_paths = get_image_paths(real_folder)

        if args.max_real:
            real_paths = real_paths[:args.max_real]

        gen_folder = os.path.join(args.generated_root, cls)

        if not os.path.exists(gen_folder):
            print(f"SKIP: generated folder not found: {gen_folder}")
            continue

        gen_paths = get_image_paths(gen_folder)

        if args.max_gen:
            gen_paths = gen_paths[:args.max_gen]

        if not real_paths:
            print("SKIP: no real images found.")
            continue

        if not gen_paths:
            print("SKIP: no generated images found.")
            continue

        print(f"Real images:      {len(real_paths)}")
        print(f"Generated images: {len(gen_paths)}")

        real_score = compute_clipiqa(
            real_paths,
            clipiqa_metric,
            device,
            desc=f"{cls} real"
        )

        gen_score = compute_clipiqa(
            gen_paths,
            clipiqa_metric,
            device,
            desc=f"{cls} generated"
        )

        print(f"CLIPIQA real:      {real_score:.4f}")
        print(f"CLIPIQA generated: {gen_score:.4f}")
        print(f"Delta:             {gen_score - real_score:+.4f}")

        results[cls] = {
            "clipiqa_real": real_score,
            "clipiqa_gen": gen_score,
        }

    if not results:
        print("\nERROR: No classes were successfully evaluated.")
        return

    df = build_results_table(results)
    print_table(df)

    csv_path = os.path.join(out_dir, "clipiqa_real_vs_generated.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()