import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
from tqdm import tqdm


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
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


def get_image_paths(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])


def build_class_to_real_paths(csv_path, images_root):
    df = pd.read_csv(csv_path)

    df["full_path"] = df["image_id"].apply(
        lambda x: os.path.join(images_root, f"{x}.jpg")
    )

    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    print(f"Found {len(df)} real images across {df['dx'].nunique()} classes")

    class_to_paths = {}

    for cls in CLASSES:
        paths = df[df["dx"] == cls]["full_path"].tolist()
        class_to_paths[cls] = paths
        print(f"  {cls:<8}: {len(paths)} real images")

    return class_to_paths


def build_clipiqa(device):
    try:
        import pyiqa
    except ImportError:
        raise ImportError("pyiqa is not installed. Run: pip install pyiqa")

    metric = pyiqa.create_metric("clipiqa", device=device)
    print("Loaded CLIPIQA")
    return metric


def compute_clipiqa(image_paths, clipiqa_metric, device, desc=""):
    values = []

    loader = data.DataLoader(
        PathDataset(image_paths, iqa_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    for batch in tqdm(loader, leave=False, desc=f"CLIPIQA {desc}"):
        batch = batch.to(device)

        with torch.no_grad():
            scores = clipiqa_metric(batch)

        values.extend(scores.detach().cpu().numpy().reshape(-1).tolist())

    return float(np.mean(values)) if values else float("nan")


def build_comparison_table(results):
    rows = []

    for cls in CLASSES:
        if cls not in results:
            continue

        real_score = results[cls]["clipiqa_real"]
        gen_score = results[cls]["clipiqa_gen"]
        delta = gen_score - real_score

        rows.append({
            "class": cls,
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


def print_table(df):
    print("\n" + "=" * 65)
    print("CLIPIQA REAL vs GENERATED COMPARISON — HAM10000")
    print("=" * 65)
    print(f"{'Class':<10} {'Real':>10} {'Generated':>12} {'Delta':>10}")
    print("-" * 65)

    for _, row in df.iterrows():
        if row["class"] == "MEAN":
            print("-" * 65)

        print(
            f"{row['class']:<10} "
            f"{row['clipiqa_real']:>10.4f} "
            f"{row['clipiqa_gen']:>12.4f} "
            f"{row['clipiqa_delta']:>+10.4f}"
        )

    print("=" * 65)
    print("Delta = generated CLIPIQA - real CLIPIQA")
    print("Positive delta means generated images scored higher.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIPIQA on real and generated HAM10000 images."
    )

    parser.add_argument("--generated_root", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--max_real", type=int, default=None)
    parser.add_argument("--max_gen", type=int, default=None)

    args = parser.parse_args()

    out_dir = os.path.join(args.generated_root, "clipiqa_metrics")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Generated root: {args.generated_root}\n")

    print("Loading real image paths from CSV...")
    class_to_real_paths = build_class_to_real_paths(
        args.csv_path,
        args.images_root
    )

    print("\nInitialising CLIPIQA...")
    clipiqa_metric = build_clipiqa(device)

    results = {}

    for cls in CLASSES:
        print(f"\n{'=' * 50}")
        print(cls)
        print(f"{'=' * 50}")

        real_paths = class_to_real_paths.get(cls, [])

        if not real_paths:
            print(f"SKIP: no real images found for {cls}")
            continue

        if args.max_real:
            real_paths = real_paths[:args.max_real]

        gen_folder = os.path.join(args.generated_root, cls)

        if not os.path.exists(gen_folder):
            print(f"SKIP: generated folder not found: {gen_folder}")
            continue

        gen_paths = get_image_paths(gen_folder)

        if args.max_gen:
            gen_paths = gen_paths[:args.max_gen]

        if not gen_paths:
            print(f"SKIP: no generated images found for {cls}")
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
        print("\nERROR: No classes evaluated. Check your folder structure.")
        return

    df = build_comparison_table(results)
    print_table(df)

    csv_out = os.path.join(out_dir, "clipiqa_real_vs_generated.csv")
    df.to_csv(csv_out, index=False)

    print(f"Saved CSV to: {csv_out}")


if __name__ == "__main__":
    main()