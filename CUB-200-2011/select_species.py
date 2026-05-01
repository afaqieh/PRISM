import os
import argparse
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel

MODEL_NAME   = "openai/clip-vit-base-patch32"
DEFAULT_N    = 15
CLASSES_FILE = "/Users/abood/Downloads/archive-2/CUB_200_2011/classes.txt"

def load_species(classes_file):
    species = []
    with open(classes_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            idx      = int(parts[0])
            raw_name = parts[1]

            if "." in raw_name:
                raw_name = raw_name.split(".", 1)[1]

            clean_name = raw_name.replace("_", " ")

            prompt = f"a photo of a {clean_name}"
            species.append((idx, clean_name, prompt))
    return species


def encode_texts(prompts, model_name, device):
    print(f"Loading CLIP text encoder ({model_name})...")
    tokenizer    = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
    text_encoder.eval()

    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tok   = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            emb = text_encoder(
                input_ids      = tok.input_ids.to(device),
                attention_mask = tok.attention_mask.to(device),
            ).pooler_output
            embeddings.append(emb.cpu().float())

    embeddings = torch.cat(embeddings, dim=0)

    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings


def find_most_confusable(embeddings, species, n):
    sim_matrix = (embeddings @ embeddings.T).numpy()
    np.fill_diagonal(sim_matrix, -1)

    best_pair  = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    selected   = list(best_pair)
    remaining  = [i for i in range(len(species)) if i not in selected]

    while len(selected) < n and remaining:
        best_score    = -np.inf
        best_candidate = None
        for candidate in remaining:
            score = sim_matrix[candidate, selected].mean()
            if score > best_score:
                best_score     = score
                best_candidate = candidate
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes_file", default=CLASSES_FILE,
                        help="Path to CUB-200-2011 classes.txt")
    parser.add_argument("--n",            type=int, default=DEFAULT_N,
                        help="Number of species to select (default: 15)")
    parser.add_argument("--output",       type=str,
                        default="selected_species.txt",
                        help="Output file for selected species names")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Selecting {args.n} most CLIP-confusable species from CUB-200-2011\n")

    species = load_species(args.classes_file)
    print(f"Loaded {len(species)} species from {args.classes_file}")

    prompts    = [s[2] for s in species]
    names      = [s[1] for s in species]
    indices    = [s[0] for s in species]

    embeddings = encode_texts(prompts, MODEL_NAME, device)
    print(f"Encoded {len(prompts)} species names with CLIP\n")

    selected_indices = find_most_confusable(embeddings, species, args.n)

    sim_matrix = (embeddings @ embeddings.T).numpy()
    np.fill_diagonal(sim_matrix, -1)

    selected_emb   = embeddings[selected_indices]
    selected_emb   = selected_emb / selected_emb.norm(dim=-1, keepdim=True)
    pairwise_sim   = (selected_emb @ selected_emb.T).numpy()
    np.fill_diagonal(pairwise_sim, -1)
    avg_sim        = pairwise_sim[pairwise_sim > -1].mean()

    print(f"{'='*60}")
    print(f"Selected {args.n} most CLIP-confusable species:")
    print(f"Average pairwise CLIP similarity: {avg_sim:.4f}")
    print(f"{'='*60}")
    for rank, idx in enumerate(selected_indices):
        species_name = names[idx]
        species_idx  = indices[idx]

        row_sim = pairwise_sim[rank]
        mean_to_others = row_sim[row_sim > -1].mean()
        print(f"  {rank+1:2d}. [{species_idx:3d}] {species_name:<40s} "
              f"(mean sim to others: {mean_to_others:.4f})")

    print(f"\nPairwise similarity matrix:")
    selected_names = [names[i] for i in selected_indices]
    col_width = max(len(n) for n in selected_names)
    header = " " * 4 + "  ".join(f"{i+1:2d}" for i in range(args.n))
    print(header)
    for i, name in enumerate(selected_names):
        row = f"{i+1:2d}. "
        for j in range(args.n):
            if i == j:
                row += "  — "
            else:
                row += f"{pairwise_sim[i,j]:4.2f}"
            row += "  "
        print(row)

    with open(args.output, "w") as f:
        f.write(f"# Selected {args.n} most CLIP-confusable CUB-200-2011 species\n")
        f.write(f"# Average pairwise CLIP text similarity: {avg_sim:.4f}\n")
        f.write(f"# Format: <original_idx> <species_name>\n\n")
        for idx in selected_indices:
            f.write(f"{indices[idx]:3d} {names[idx]}\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
