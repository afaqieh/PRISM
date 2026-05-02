import argparse
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadata_conditioning import MetadataConditionEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--csv",            default="data/cub_train.csv")
parser.add_argument("--convnext_model", default="convnext_large.fb_in22k",
                    help="timm model name — e.g. convnext_large.fb_in22k or convnext_large.fb_in22k_ft_in1k")
parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch_size",     type=int, default=32)
parser.add_argument("--checkpoint",     default="results/cub_stage1/seed_42/lora_cub_seed_42.pth",
                    help="Stage 1 checkpoint (for cond_encoder weights)")
parser.add_argument("--out_dir",        default="data/diagnostics")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

df = pd.read_csv(args.csv)
df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

species_list = sorted(df["species_name"].unique())
species2idx  = {s: i for i, s in enumerate(species_list)}
df["label"]  = df["species_name"].map(species2idx)
K            = len(species_list)

print(f"Loaded {len(df)} images across {K} species")
print(f"Species: {species_list}\n")

paths     = df["full_path"].tolist()
labels_np = df["label"].values
colors    = plt.cm.tab20(np.linspace(0, 1, K))
short_names = [s.replace(" Warbler", "").replace("Yellow-rumped", "Y-rumped")
               for s in species_list]

print(f"Loading ConvNeXt-Large ({args.convnext_model}) from timm...")
try:
    import timm
except ImportError:
    raise ImportError("timm not installed — run: pip install timm")

model    = timm.create_model(args.convnext_model, pretrained=True, num_classes=0)
model.eval().to(args.device)

data_cfg  = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_cfg, is_training=False)

print("Extracting ConvNeXt-Large features...")
all_feats = []
for i in tqdm(range(0, len(paths), args.batch_size)):
    bp   = paths[i : i + args.batch_size]
    imgs = [transform(Image.open(p).convert("RGB")) for p in bp]
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
print(f"ConvNeXt-Large silhouette: {cnx_silhouette:.4f}")

cnx_pairs = []
for i in range(K):
    for j in range(i + 1, K):
        cnx_pairs.append((cnx_sim[i, j], species_list[i], species_list[j]))
cnx_pairs.sort(reverse=True)

print(f"\nLoading MetadataConditionEncoder from {args.checkpoint}...")
ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

cond_encoder = MetadataConditionEncoder(
    ckpt["field_configs"], hidden_dim=256, final_dim=768
).to(args.device)
cond_encoder.load_state_dict(ckpt["cond_encoder"])
cond_encoder.eval()

sp2idx_ckpt = ckpt["species2idx"]
META_COLS   = ["throat_color", "forehead_color", "belly_color", "nape_color"]

print("Computing metadata prototypes (modal attributes per species)...")
meta_rows = []
with torch.no_grad():
    for species in species_list:
        rows        = df[df["species_name"] == species]
        modal_attrs = {col: int(rows[col].mode()[0]) for col in META_COLS}
        meta_batch  = {
            "species":        torch.tensor([sp2idx_ckpt[species]], device=args.device),
            "throat_color":   torch.tensor([modal_attrs["throat_color"]],   device=args.device),
            "forehead_color": torch.tensor([modal_attrs["forehead_color"]], device=args.device),
            "belly_color":    torch.tensor([modal_attrs["belly_color"]],    device=args.device),
            "nape_color":     torch.tensor([modal_attrs["nape_color"]],     device=args.device),
        }
        cond          = cond_encoder(meta_batch)
        summary_token = F.normalize(cond[0, 5, :], dim=0)
        meta_rows.append(summary_token.cpu())

meta_matrix = torch.stack(meta_rows)
meta_unit   = meta_matrix.numpy()
meta_sim    = (meta_matrix @ meta_matrix.T).numpy()

meta_per_sample = np.array([meta_unit[l] for l in labels_np])
meta_silhouette = silhouette_score(meta_per_sample, labels_np)

meta_pairs = []
for i in range(K):
    for j in range(i + 1, K):
        meta_pairs.append((meta_sim[i, j], species_list[i], species_list[j]))
meta_pairs.sort(reverse=True)

print(f"Metadata silhouette: {meta_silhouette:.4f}")

report_path = os.path.join(args.out_dir, "separability_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 65 + "\n")
    f.write("  SEPARABILITY DIAGNOSTIC REPORT — CUB 15 Warblers\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"ConvNeXt model : {args.convnext_model}\n")
    f.write(f"Total images   : {len(df)}\n")
    f.write(f"Species        : {K}\n\n")

    f.write("Silhouette scores (closer to 1.0 = tighter clusters):\n")
    f.write(f"  ConvNeXt-Large  : {cnx_silhouette:.4f}\n")
    f.write(f"  Metadata        : {meta_silhouette:.4f}\n\n")

    f.write("Top 10 hardest ConvNeXt pairs (most visually confusable → best hard negatives):\n")
    for sim, a, b in cnx_pairs[:10]:
        f.write(f"  {sim:.4f}  {a}  ↔  {b}\n")

    f.write("\nTop 10 hardest metadata pairs:\n")
    for sim, a, b in meta_pairs[:10]:
        f.write(f"  {sim:.4f}  {a}  ↔  {b}\n")

    f.write("\nPer-species intra-class std (higher = more spread within class):\n")
    for idx, species in enumerate(species_list):
        mask  = labels_np == idx
        std   = all_feats[mask].std(0).mean().item()
        count = mask.sum()
        f.write(f"  {species:<40} n={count:>3}  std={std:.4f}\n")

print(f"Report saved → {report_path}")

mask_diag = np.eye(K, dtype=bool)

def save_heatmap(sim_matrix, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim_matrix, ax=ax,
        xticklabels=short_names, yticklabels=short_names,
        annot=True, fmt=".2f", cmap="coolwarm",
        vmin=0.5, vmax=1.0,
        mask=mask_diag,
        linewidths=0.3,
    )
    ax.set_title(title, fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")

save_heatmap(
    cnx_sim,
    f"ConvNeXt-Large Prototype Cosine Similarity  (silhouette={cnx_silhouette:.3f})",
    os.path.join(args.out_dir, "convnext_similarity_matrix.png"),
)
save_heatmap(
    meta_sim,
    f"Metadata Attribute Cosine Similarity  (silhouette={meta_silhouette:.3f})",
    os.path.join(args.out_dir, "meta_similarity_matrix.png"),
)

perp = min(30, len(df) // K)

print("\nRunning t-SNE on ConvNeXt-Large features (all images)...")
cnx_2d = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000
              ).fit_transform(all_feats.numpy())

fig, ax = plt.subplots(figsize=(12, 9))
for idx, species in enumerate(species_list):
    mask = labels_np == idx
    ax.scatter(cnx_2d[mask, 0], cnx_2d[mask, 1],
            color=colors[idx],
            alpha=0.7,
            s=50,
            label=species.replace(" Warbler", ""))

ax.legend(
    fontsize=13,
    markerscale=2,
    ncol=2,
    loc="upper right"
)
ax.set_title(f"ConvNeXt-Large t-SNE — 15 Warblers  (silhouette={cnx_silhouette:.3f})", fontsize=13)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout()
out = os.path.join(args.out_dir, "convnext_tsne.png")
plt.savefig(out, dpi=150); plt.close()
print(f"Saved → {out}")

print("Running t-SNE on metadata prototypes (one point per class)...")
meta_2d = TSNE(n_components=2, perplexity=min(5, K - 1), random_state=42, max_iter=2000
               ).fit_transform(meta_unit)

fig, ax = plt.subplots(figsize=(10, 8))
for idx, species in enumerate(species_list):
    ax.scatter(meta_2d[idx, 0], meta_2d[idx, 1], color=colors[idx], s=120, zorder=3)
    ax.annotate(species.replace(" Warbler", ""),
                (meta_2d[idx, 0], meta_2d[idx, 1]),
                fontsize=7, ha="center", va="bottom",
                xytext=(0, 5), textcoords="offset points")
ax.set_title(f"Metadata Attribute t-SNE — 15 Warblers  (silhouette={meta_silhouette:.3f})", fontsize=13)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
plt.tight_layout()
out = os.path.join(args.out_dir, "meta_tsne.png")
plt.savefig(out, dpi=150); plt.close()
print(f"Saved → {out}")

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  ConvNeXt-Large silhouette : {cnx_silhouette:.4f}")
print(f"  Metadata silhouette       : {meta_silhouette:.4f}")
print(f"\n  Hardest ConvNeXt pair : {cnx_pairs[0][1]}  ↔  {cnx_pairs[0][2]}")
print(f"    cosine sim = {cnx_pairs[0][0]:.4f}")
print(f"\n  Hardest metadata pair : {meta_pairs[0][1]}  ↔  {meta_pairs[0][2]}")
print(f"    cosine sim = {meta_pairs[0][0]:.4f}")
print(f"\n  All outputs in: {args.out_dir}/")
print("=" * 65)
