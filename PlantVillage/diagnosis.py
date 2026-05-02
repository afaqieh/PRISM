import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--csv",            default="data/plantvillage_train.csv")
parser.add_argument("--convnext_model", default="convnext_large.fb_in22k")
parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size",     type=int, default=32)
parser.add_argument("--out_dir",        default="data/diagnostics_plantvillage")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

SHORT_NAMES = {
    "Pepper__bell___Bacterial_spot":                "Pep/BactSpot",
    "Pepper__bell___healthy":                       "Pep/Healthy",
    "Potato___Early_blight":                        "Pot/EarlyBlt",
    "Potato___Late_blight":                         "Pot/LateBlt",
    "Potato___healthy":                             "Pot/Healthy",
    "Tomato_Bacterial_spot":                        "Tom/BactSpot",
    "Tomato_Early_blight":                          "Tom/EarlyBlt",
    "Tomato_Late_blight":                           "Tom/LateBlt",
    "Tomato_Leaf_Mold":                             "Tom/LeafMold",
    "Tomato_Septoria_leaf_spot":                    "Tom/Septoria",
    "Tomato_Spider_mites_Two_spotted_spider_mite":  "Tom/SpiderMit",
    "Tomato__Target_Spot":                          "Tom/TargetSpt",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":        "Tom/YellowCrl",
    "Tomato__Tomato_mosaic_virus":                  "Tom/Mosaic",
    "Tomato_healthy":                               "Tom/Healthy",
}

df = pd.read_csv(args.csv)
df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

class_list  = sorted(df["class_name"].unique())
class2idx   = {c: i for i, c in enumerate(class_list)}
df["label"] = df["class_name"].map(class2idx)
K           = len(class_list)

print(f"Loaded {len(df)} images across {K} classes")
for c in class_list:
    print(f"  {SHORT_NAMES.get(c, c):<14}: {(df['class_name'] == c).sum()} images")

paths     = df["full_path"].tolist()
labels_np = df["label"].values
short_names = [SHORT_NAMES.get(c, c) for c in class_list]
colors      = plt.cm.tab20(np.linspace(0, 1, K))

print(f"\nLoading ConvNeXt-Large ({args.convnext_model}) from timm...")
try:
    import timm
except ImportError:
    raise ImportError("timm not installed — run: pip install timm")

model    = timm.create_model(args.convnext_model, pretrained=True, num_classes=0)
model.eval().to(args.device)
for p in model.parameters():
    p.requires_grad = False

data_cfg  = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_cfg, is_training=False)

print("Extracting ConvNeXt-Large features...")
all_feats = []
for i in tqdm(range(0, len(paths), args.batch_size)):
    bp   = paths[i : i + args.batch_size]
    imgs = []
    for p in bp:
        try:
            imgs.append(transform(Image.open(p).convert("RGB")))
        except Exception:
            imgs.append(torch.zeros(3, 224, 224))
    batch = torch.stack(imgs).to(args.device)
    with torch.no_grad():
        feats = model(batch)
        feats = F.normalize(feats, dim=-1)
    all_feats.append(feats.cpu())

all_feats = torch.cat(all_feats, dim=0)
print(f"Feature dim: {all_feats.shape[1]}")

cnx_protos = []
for idx in range(K):
    mask  = labels_np == idx
    proto = F.normalize(all_feats[mask].mean(0), dim=0)
    cnx_protos.append(proto)
cnx_protos = torch.stack(cnx_protos)

cnx_sim        = (cnx_protos @ cnx_protos.T).numpy()
cnx_silhouette = silhouette_score(all_feats.numpy(), labels_np)
print(f"\nConvNeXt-Large silhouette: {cnx_silhouette:.4f}")

cnx_pairs = []
for i in range(K):
    for j in range(i + 1, K):
        cnx_pairs.append((cnx_sim[i, j], class_list[i], class_list[j]))
cnx_pairs.sort(reverse=True)

report_path = os.path.join(args.out_dir, "separability_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  SEPARABILITY DIAGNOSTIC — PlantVillage 15 Classes\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"ConvNeXt model : {args.convnext_model}\n")
    f.write(f"Total images   : {len(df)}\n")
    f.write(f"Classes        : {K}\n\n")

    f.write(f"Silhouette score (closer to 1.0 = tighter clusters):\n")
    f.write(f"  ConvNeXt-Large : {cnx_silhouette:.4f}\n\n")

    f.write("Top 10 hardest ConvNeXt pairs (most visually confusable):\n")
    for sim, a, b in cnx_pairs[:10]:
        f.write(f"  {sim:.4f}  {SHORT_NAMES.get(a,a)}  ↔  {SHORT_NAMES.get(b,b)}\n")

    f.write("\nPer-class intra-class std (higher = more spread within class):\n")
    for idx, cls in enumerate(class_list):
        mask  = labels_np == idx
        std   = all_feats[mask].std(0).mean().item()
        count = mask.sum()
        f.write(f"  {SHORT_NAMES.get(cls,cls):<14}  n={count:>4}  std={std:.4f}\n")

print(f"Report saved {report_path}")
print("\nTop 10 hardest pairs:")
for sim, a, b in cnx_pairs[:10]:
    print(f"  {sim:.4f}  {SHORT_NAMES.get(a,a):<14}  ↔  {SHORT_NAMES.get(b,b)}")

mask_diag = np.eye(K, dtype=bool)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cnx_sim, ax=ax,
    xticklabels=short_names, yticklabels=short_names,
    annot=True, fmt=".2f", cmap="coolwarm",
    vmin=0.5, vmax=1.0,
    mask=mask_diag,
    linewidths=0.3,
)
ax.set_title(
    f"ConvNeXt-Large Prototype Cosine Similarity — PlantVillage\n"
    f"(silhouette={cnx_silhouette:.3f})",
    fontsize=13,
)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
out = os.path.join(args.out_dir, "convnext_similarity_matrix.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")

perp = min(30, len(df) // K)
print(f"\nRunning t-SNE (perplexity={perp}) on all {len(df)} images...")
cnx_2d = TSNE(
    n_components=2, perplexity=perp, random_state=42, max_iter=1000
).fit_transform(all_feats.numpy())

fig, ax = plt.subplots(figsize=(13, 10))
for idx, cls in enumerate(class_list):
    mask = labels_np == idx
    ax.scatter(
        cnx_2d[mask, 0], cnx_2d[mask, 1],
        color=colors[idx], alpha=0.55, s=20,
        label=short_names[idx],
    )
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.7)
ax.set_title(
    f"ConvNeXt-Large t-SNE — PlantVillage 15 Classes\n"
    f"(silhouette={cnx_silhouette:.3f})",
    fontsize=13,
)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
out = os.path.join(args.out_dir, "convnext_tsne.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  ConvNeXt-Large silhouette : {cnx_silhouette:.4f}")
print(f"\n  Hardest pair : {SHORT_NAMES.get(cnx_pairs[0][1], cnx_pairs[0][1])}"
      f"  ↔  {SHORT_NAMES.get(cnx_pairs[0][2], cnx_pairs[0][2])}")
print(f"    cosine sim = {cnx_pairs[0][0]:.4f}")
print(f"\n  All outputs in: {args.out_dir}/")
print("=" * 65)
