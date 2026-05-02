import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.data import Dataset, DataLoader
from scipy.linalg import sqrtm

BATCH_SIZE = 64
WORKERS    = 4

inception_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return inception_tf(Image.open(self.paths[i]).convert("RGB"))


def make_loader(paths):
    return DataLoader(PathDataset(paths), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=WORKERS, pin_memory=True)


def get_image_paths(folder):
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])


def extract_features(paths, model, device):
    feats = []
    for batch in tqdm(make_loader(paths), desc="  Extracting", leave=False):
        with torch.no_grad():
            feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def compute_fid(feat_real, feat_gen):
    mu_r = feat_real.mean(0)
    mu_g = feat_gen.mean(0)
    sg_r = np.cov(feat_real, rowvar=False) + np.eye(feat_real.shape[1]) * 1e-6
    sg_g = np.cov(feat_gen,  rowvar=False) + np.eye(feat_gen.shape[1])  * 1e-6
    diff = mu_r - mu_g
    cov, _ = sqrtm(sg_r @ sg_g, disp=False)
    if np.iscomplexobj(cov):
        cov = cov.real
    return float(diff @ diff + np.trace(sg_r + sg_g - 2 * cov))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_root", required=True,
                        help="Root folder with one subfolder per dx class")
    parser.add_argument("--csv_path", default="./data/HAM10000_metadata_train.csv",
                        help="Train split CSV (real images)")
    parser.add_argument("--img_dir",  default="/storage/homefs/af24h089/MSGAI/Final/ham_lora",
                        help="Directory containing HAM10000 .jpg images")
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["dx"]).reset_index(drop=True)
    df["filepath"] = df["image_id"].astype(str).apply(
        lambda f: os.path.join(args.img_dir, f + ".jpg")
    )
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
    real_paths = df["filepath"].tolist()
    print(f"Real images  : {len(real_paths)}")
    print(f"Classes      : {sorted(df['dx'].unique())}")

    gen_paths = []
    for entry in sorted(os.listdir(args.generated_root)):
        sp_dir = os.path.join(args.generated_root, entry)
        if os.path.isdir(sp_dir):
            paths = get_image_paths(sp_dir)
            gen_paths.extend(paths)
    print(f"Generated    : {len(gen_paths)}")

    if len(gen_paths) == 0:
        print("ERROR: no generated images found. Check --generated_root.")
        return

    print("\nLoading Inception v3...")
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    print("Extracting real features...")
    feat_real = extract_features(real_paths, model, device)
    print(f"  shape: {feat_real.shape}")

    print("Extracting generated features...")
    feat_gen = extract_features(gen_paths, model, device)
    print(f"  shape: {feat_gen.shape}")

    print("\nComputing FID...")
    fid = compute_fid(feat_real, feat_gen)
    print(f"\n{'='*40}")
    print(f"  Global FID = {fid:.2f}")
    print(f"{'='*40}")
    print(f"  Real images     : {len(real_paths)}")
    print(f"  Generated images: {len(gen_paths)}")
    print(f"  Generated root  : {args.generated_root}")


if __name__ == "__main__":
    main()
