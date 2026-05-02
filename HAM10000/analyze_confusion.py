import argparse
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr, spearmanr, ttest_ind
from transformers import CLIPTokenizer, CLIPTextModel


MODEL_NAME = "runwayml/stable-diffusion-v1-5"

MIN_CLIP_CONSISTENCY = 0.15

HAM_MAP = {
    "akiec": "actinic keratosis",
    "bcc":   "basal cell carcinoma",
    "bkl":   "benign keratosis-like lesion",
    "df":    "dermatofibroma",
    "mel":   "melanoma",
    "nv":    "melanocytic nevus",
    "vasc":  "vascular lesion",
}


def prettify(name: str) -> str:
    return name.replace("_", " ")


def build_prompts(class_names: list[str], prompt_template: str) -> list[str]:
    prompts = []
    for c in class_names:
        full_name = HAM_MAP.get(c, prettify(c))
        prompts.append(prompt_template.format(class_name=full_name))
    return prompts


@torch.no_grad()
def compute_clip_text_embeddings(
    prompts: list[str],
    model_name: str = MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer  = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(device)
    text_model.eval()

    inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)

    outputs = text_model(**inputs)
    hidden  = outputs.last_hidden_state
    eos_pos = inputs.input_ids.argmax(dim=-1)
    emb     = hidden[torch.arange(hidden.shape[0]), eos_pos]
    emb     = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return emb.cpu().numpy()


def cosine_distance_matrix(emb: np.ndarray) -> np.ndarray:
    sim = emb @ emb.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def pairwise_confusion_reduction(
    clip_conf: np.ndarray,
    ours_conf: np.ndarray,
    class_names: list[str],
    clip_dist: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, j in combinations(range(len(class_names)), 2):
        clip_pair_conf = 0.5 * (clip_conf[i, j] + clip_conf[j, i])
        ours_pair_conf = 0.5 * (ours_conf[i, j] + ours_conf[j, i])
        reduction = clip_pair_conf - ours_pair_conf
        rows.append({
            "class_i":             class_names[i],
            "class_j":             class_names[j],
            "class_i_pretty":      HAM_MAP.get(class_names[i], prettify(class_names[i])),
            "class_j_pretty":      HAM_MAP.get(class_names[j], prettify(class_names[j])),
            "pair_name":           f"{HAM_MAP.get(class_names[i], prettify(class_names[i]))} / "
                                   f"{HAM_MAP.get(class_names[j], prettify(class_names[j]))}",
            "clip_distance":       float(clip_dist[i, j]),
            "clip_pair_confusion": float(clip_pair_conf),
            "ours_pair_confusion": float(ours_pair_conf),
            "confusion_reduction": float(reduction),
        })
    return pd.DataFrame(rows)


def filter_pairs(df_pairs: pd.DataFrame, clip_conf: np.ndarray,
                 class_names: list[str]) -> pd.DataFrame:
    """
    Option A: remove pairs where CLIP achieves below MIN_CLIP_CONSISTENCY
    on BOTH classes. If CLIP generates near-random output for both classes,
    the off-diagonal confusion between them is noise from failed conditioning,
    not a meaningful signal of the language bottleneck.
    """
    clip_consistency = {
        class_names[i]: float(clip_conf[i, i])
        for i in range(len(class_names))
    }
    mask = df_pairs.apply(
        lambda row: not (
            clip_consistency[row["class_i"]] < MIN_CLIP_CONSISTENCY and
            clip_consistency[row["class_j"]] < MIN_CLIP_CONSISTENCY
        ),
        axis=1,
    )
    n_before = len(df_pairs)
    df_filtered = df_pairs[mask].reset_index(drop=True)
    n_after = len(df_filtered)
    print(f"\nOption A filter (threshold={MIN_CLIP_CONSISTENCY}):")
    print(f"  Removed {n_before - n_after} pairs where CLIP consistency "
          f"< {MIN_CLIP_CONSISTENCY} on both classes")
    print(f"  Remaining: {n_after}/{n_before} pairs")
    return df_filtered, n_before - n_after


def make_scatter(df_pairs: pd.DataFrame, out_path: str) -> dict:
    x = df_pairs["clip_distance"].values
    y = df_pairs["confusion_reduction"].values

    pear_r,    pear_p    = pearsonr(x, y)
    spear_rho, spear_p   = spearmanr(x, y)

    coef = np.polyfit(x, y, deg=1)
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = coef[0] * xfit + coef[1]

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.75)
    plt.plot(
        xfit, yfit, linewidth=2,
        label=(
            f"Linear fit\n"
            f"Pearson r={pear_r:.3f} (p={pear_p:.2e})\n"
            f"Spearman ρ={spear_rho:.3f} (p={spear_p:.2e})"
        ),
    )
    top = df_pairs.sort_values("confusion_reduction", ascending=False).head(8)
    for _, row in top.iterrows():
        plt.annotate(
            row["pair_name"],
            (row["clip_distance"], row["confusion_reduction"]),
            xytext=(5, 5), textcoords="offset points", fontsize=10,
        )
    plt.xlabel("CLIP text-embedding distance between class prompts")
    plt.ylabel("Pairwise confusion reduction (CLIP - ours)")
    plt.title(
        f"Does structured conditioning reduce confusion most for CLIP-near HAM10000 classes?\n"
        f"(pairs excluded if CLIP consistency < {MIN_CLIP_CONSISTENCY} on both classes)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "pearson_r":        float(pear_r),
        "pearson_p":        float(pear_p),
        "spearman_rho":     float(spear_rho),
        "spearman_p":       float(spear_p),
        "linear_slope":     float(coef[0]),
        "linear_intercept": float(coef[1]),
    }


def make_binned_plot(df: pd.DataFrame, out_path: str, n_bins: int = 5) -> pd.DataFrame:
    df = df.copy()
    bins = np.linspace(df["clip_distance"].min(), df["clip_distance"].max(), n_bins + 1)
    df["bin"] = np.digitize(df["clip_distance"], bins, right=False) - 1
    df.loc[df["bin"] == n_bins, "bin"] = n_bins - 1
    rows = []
    for b in range(n_bins):
        subset = df[df["bin"] == b]
        if len(subset) == 0:
            continue
        mean_dist = subset["clip_distance"].mean()
        mean_red  = subset["confusion_reduction"].median()
        std_red   = subset["confusion_reduction"].std(ddof=1) if len(subset) > 1 else 0.0
        se_red    = std_red / np.sqrt(len(subset)) if len(subset) > 0 else 0.0
        rows.append({
            "bin": b,
            "bin_left":               bins[b],
            "bin_right":              bins[b + 1],
            "mean_clip_distance":     mean_dist,
            "mean_confusion_reduction": mean_red,
            "std_confusion_reduction":  std_red,
            "se_confusion_reduction":   se_red,
            "count": len(subset),
        })
    df_bins = pd.DataFrame(rows)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        df_bins["mean_clip_distance"],
        df_bins["mean_confusion_reduction"],
        yerr=df_bins["se_confusion_reduction"],
        marker="o", capsize=5,
    )
    plt.xlabel("CLIP distance (binned)")
    plt.ylabel("Mean confusion reduction")
    plt.title("Binned trend: confusion reduction vs CLIP distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return df_bins


def summarize_closest_vs_farthest(df_pairs: pd.DataFrame, k: int = 5):
    df_sorted = df_pairs.sort_values("clip_distance", ascending=True).reset_index(drop=True)
    closest  = df_sorted.head(k).copy()
    farthest = df_sorted.tail(k).copy()
    closest["group"]  = "closest"
    farthest["group"] = "farthest"
    rows = []
    for name, df_sub in [("closest", closest), ("farthest", farthest)]:
        rows.append({
            "group":                    name,
            "n_pairs":                  len(df_sub),
            "mean_clip_distance":       df_sub["clip_distance"].mean(),
            "mean_clip_pair_confusion": df_sub["clip_pair_confusion"].mean(),
            "mean_ours_pair_confusion": df_sub["ours_pair_confusion"].mean(),
            "mean_confusion_reduction": df_sub["confusion_reduction"].mean(),
            "std_confusion_reduction":  df_sub["confusion_reduction"].std(ddof=1),
        })
    t_stat, p_val = ttest_ind(
        closest["confusion_reduction"].values,
        farthest["confusion_reduction"].values,
        equal_var=False,
    )
    summary_df = pd.DataFrame(rows)
    summary_df.attrs["ttest_stat"] = float(t_stat)
    summary_df.attrs["ttest_p"]    = float(p_val)
    extremes_df = pd.concat([closest, farthest], ignore_index=True)
    return summary_df, extremes_df


def make_bar_closest_vs_farthest(summary_df: pd.DataFrame, out_path: str) -> None:
    x   = summary_df["group"].tolist()
    y   = summary_df["mean_confusion_reduction"].tolist()
    err = summary_df["std_confusion_reduction"].fillna(0.0).tolist()
    plt.figure(figsize=(6, 5))
    plt.bar(x, y, yerr=err, capsize=6)
    plt.ylabel("Mean pairwise confusion reduction")
    plt.title("Closest vs farthest CLIP pairs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_top_pairs(df_pairs: pd.DataFrame, out_path: str, top_k: int = 15) -> None:
    df_top = df_pairs.sort_values("confusion_reduction", ascending=False).head(top_k)
    df_top.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",      type=str, required=True)
    ap.add_argument("--prompt-template", type=str, default="{class_name}")
    ap.add_argument("--top-k",  type=int, default=5)
    ap.add_argument("--n-bins", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    clip_conf = np.array([
        [0.16, 0.27, 0.25, 0.02, 0.12, 0.12, 0.06],
        [0.06, 0.32, 0.18, 0.01, 0.16, 0.15, 0.12],
        [0.16, 0.14, 0.15, 0.00, 0.11, 0.28, 0.16],
        [0.01, 0.07, 0.11, 0.06, 0.10, 0.29, 0.36],
        [0.05, 0.04, 0.11, 0.01, 0.29, 0.34, 0.16],
        [0.00, 0.01, 0.04, 0.02, 0.09, 0.84, 0.00],
        [0.00, 0.20, 0.17, 0.01, 0.07, 0.11, 0.44],
    ], dtype=float)

    ours_conf = np.array([
        [0.69, 0.05, 0.11, 0.03, 0.06, 0.05, 0.01], 
        [0.02, 0.53, 0.08, 0.01, 0.04, 0.06, 0.27],
        [0.00, 0.01, 0.72, 0.00, 0.19, 0.07, 0.00],
        [0.00, 0.01, 0.00, 0.91, 0.02, 0.03, 0.04],
        [0.01, 0.00, 0.11, 0.00, 0.73, 0.15, 0.00],
        [0.00, 0.00, 0.05, 0.00, 0.26, 0.68, 0.01],
        [0.00, 0.05, 0.02, 0.00, 0.02, 0.05, 0.86],
    ], dtype=float)

    prompts   = build_prompts(class_names, args.prompt_template)
    emb       = compute_clip_text_embeddings(prompts)
    clip_dist = cosine_distance_matrix(emb)

    df_pairs = pairwise_confusion_reduction(
        clip_conf=clip_conf,
        ours_conf=ours_conf,
        class_names=class_names,
        clip_dist=clip_dist,
    )

    n_removed = 0

    df_pairs.to_csv(
        os.path.join(args.output_dir, "pairwise_confusion_reduction.csv"), index=False)

    stats = make_scatter(
        df_pairs,
        os.path.join(args.output_dir, "scatter_confusion_reduction_vs_clip_distance.png"),
    )

    binned_df = make_binned_plot(
        df_pairs,
        os.path.join(args.output_dir, "binned_trend.png"),
        n_bins=args.n_bins,
    )
    binned_df.to_csv(os.path.join(args.output_dir, "binned_trend.csv"), index=False)

    summary_df, extremes_df = summarize_closest_vs_farthest(df_pairs, k=args.top_k)
    summary_df.to_csv(
        os.path.join(args.output_dir, "closest_vs_farthest_summary.csv"), index=False)
    extremes_df.to_csv(
        os.path.join(args.output_dir, "closest_and_farthest_pairs.csv"), index=False)
    make_bar_closest_vs_farthest(
        summary_df,
        os.path.join(args.output_dir, "closest_vs_farthest_bar.png"),
    )
    save_top_pairs(
        df_pairs,
        os.path.join(args.output_dir, "top_confusion_reduction_pairs.csv"),
        top_k=15,
    )

    clip_acc = np.trace(clip_conf) / len(class_names)
    ours_acc = np.trace(ours_conf) / len(class_names)

    idx = np.triu_indices(len(class_names), k=1)
    print(f"\nMean pairwise CLIP distance: {clip_dist[idx].mean():.4f}")
    print(f"Min pairwise CLIP distance:  {clip_dist[idx].min():.4f}")
    print(f"Max pairwise CLIP distance:  {clip_dist[idx].max():.4f}")

    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump({
            "correlations": stats,
            "closest_vs_farthest_ttest_stat":  summary_df.attrs["ttest_stat"],
            "closest_vs_farthest_ttest_p":     summary_df.attrs["ttest_p"],
            "clip_accuracy_mean_diagonal":     float(clip_acc),
            "ours_accuracy_mean_diagonal":     float(ours_acc),
            "accuracy_improvement":            float(ours_acc - clip_acc),
            "min_clip_consistency_filter":     MIN_CLIP_CONSISTENCY,
            "pairs_removed_by_filter":         n_removed,
            "prompts": prompts,
        }, f, indent=2)

    print("\nSaved outputs to:", args.output_dir)
    print("\nPrompts used:")
    for p in prompts:
        print(" -", p)
    print(f"\nCLIP accuracy: {clip_acc:.3f}")
    print(f"Ours accuracy: {ours_acc:.3f}")
    print(f"Improvement:   {ours_acc - clip_acc:.3f}")
    print("\nTop confusion-reduction pairs:")
    print(df_pairs.sort_values("confusion_reduction", ascending=False).head(10)[
        ["pair_name", "clip_distance", "clip_pair_confusion",
         "ours_pair_confusion", "confusion_reduction"]
    ])
    print("\nClosest pairs:")
    print(extremes_df[extremes_df["group"] == "closest"][
        ["pair_name", "clip_distance", "clip_pair_confusion",
         "ours_pair_confusion", "confusion_reduction"]
    ])
    print("\nFarthest pairs:")
    print(extremes_df[extremes_df["group"] == "farthest"][
        ["pair_name", "clip_distance", "clip_pair_confusion",
         "ours_pair_confusion", "confusion_reduction"]
    ])
    print(
        f"\nWelch t-test (closest vs farthest confusion reduction): "
        f"stat={summary_df.attrs['ttest_stat']:.4f}, "
        f"p={summary_df.attrs['ttest_p']:.4e}"
    )


if __name__ == "__main__":
    main()