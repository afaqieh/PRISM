import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from ..lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_cub import CUBDataset
from metadata_conditioning import MetadataConditionEncoder, cub_field_configs

MODEL_NAME   = "runwayml/stable-diffusion-v1-5"
CSV_PATH     = "./data/cub_train.csv"
LORA_CKPT    = "results/cub_stage1_continue_30k/lora_cub_continue_5k.pth"
DINO_MODEL   = "dinov2_vitb14"
OUT_DIR      = "results/cub_stage2_meta_contrastive"
PROTO_CACHE  = "dino_prototypes_cub.pt"

STEPS      = 5000
BATCH_SIZE = 4
LR_G       = 5e-6
LORA_RANK  = 16

DISC_INTERVAL  = 10
N_REWARD_STEPS = 6
TOTAL_STEPS    = 50
GUIDANCE_SCALE = 6.0

LAMBDA_DIFF      = 1.0
LAMBDA_SEP_START = 0.02
LAMBDA_SEP_END   = 0.40
WARMUP_STEPS     = 1000

MARGIN   = 0.10
META_TAU = 0.1

CONFUSION_WINDOW = 30
CONFUSION_MIN_P  = 0.15

LOG_INTERVAL    = 500
CKPT_INTERVAL   = 1000
SAMPLE_INTERVAL = 500

SPECIES_LABELS = [
    "American Pipit",
    "Black and white Warbler",
    "Blue winged Warbler",
    "Canada Warbler",
    "Cape May Warbler",
    "Golden winged Warbler",
    "Kentucky Warbler",
    "Louisiana Waterthrush",
    "Magnolia Warbler",
    "Mourning Warbler",
    "Northern Waterthrush",
    "Pine Warbler",
    "Prairie Warbler",
    "Tennessee Warbler",
    "Yellow Warbler",
]
NUM_SPECIES = len(SPECIES_LABELS)

def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context

_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225])

_DINO_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_DINO_MEAN.tolist(), std=_DINO_STD.tolist()),
])


def load_dino(model_name: str, device: str):
    print(f"  Loading frozen {model_name}...")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    feat_dim = model.embed_dim
    print(f"  DINOv2 loaded — feature dim: {feat_dim}D")
    return model, feat_dim


def preprocess_for_dino(img_tensor, device):
    x    = (img_tensor + 1.0) / 2.0
    x    = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = _DINO_MEAN.view(1, 3, 1, 1).to(device)
    std  = _DINO_STD.view(1, 3, 1, 1).to(device)
    return (x - mean) / std


def compute_or_load_dino_prototypes(csv_path, species_labels, dino_model,
                                    device, cache_path=PROTO_CACHE):
    if os.path.exists(cache_path):
        print(f"  Loading DINOv2 prototypes from cache: {cache_path}")
        protos = torch.load(cache_path, map_location=device, weights_only=False)
        print(f"  Loaded {len(protos)} species prototypes.")
        return protos

    print("  Computing DINOv2 prototypes from real training images...")
    df = pd.read_csv(csv_path)
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    protos = {}
    dino_model.eval()
    for species in tqdm(species_labels, desc="  Building DINOv2 prototypes"):
        sp_df = df[df["species_name"] == species]
        feats = []
        for _, row in sp_df.iterrows():
            try:
                img = Image.open(row["full_path"]).convert("RGB")
                x   = _DINO_TRANSFORM(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = dino_model(x)
                    f = F.normalize(f, dim=-1)
                feats.append(f.squeeze(0).cpu())
            except Exception:
                continue
        if feats:
            proto = torch.stack(feats).mean(0)
            protos[species] = F.normalize(proto, dim=0).to(device)
            print(f"    {species:<35}: {len(feats)} imgs", flush=True)
        else:
            print(f"    WARNING: no images for {species}", flush=True)

    torch.save(protos, cache_path)
    print(f"  Prototypes saved {cache_path}")
    return protos

def build_metadata_sim_matrix(cond_encoder, species_labels, species2idx,
                               df, device):
    META_COLS = ["throat_color", "forehead_color", "belly_color", "nape_color"]
    K         = len(species_labels)
    embeddings = []

    cond_encoder.eval()
    with torch.no_grad():
        for species in species_labels:
            rows = df[df["species_name"] == species]
            modal_attrs = {col: int(rows[col].mode()[0]) for col in META_COLS}

            meta_batch = {
                "species":        torch.tensor([species2idx[species]], device=device),
                "throat_color":   torch.tensor([modal_attrs["throat_color"]],   device=device),
                "forehead_color": torch.tensor([modal_attrs["forehead_color"]], device=device),
                "belly_color":    torch.tensor([modal_attrs["belly_color"]],    device=device),
                "nape_color":     torch.tensor([modal_attrs["nape_color"]],     device=device),
            }
            cond    = cond_encoder(meta_batch)
            summary = cond[0, 5, :]
            summary = F.normalize(summary, dim=0)
            embeddings.append(summary.cpu())

    E      = torch.stack(embeddings)
    sim    = (E @ E.T)
    return sim


def get_neg_weights(sim_matrix, class_idx, device):
    K    = sim_matrix.shape[0]
    sims = sim_matrix[class_idx].clone()
    sims[class_idx] = -1e9

    neg_idx = [j for j in range(K) if j != class_idx]
    neg_sim = sims[neg_idx].to(device)

    weights = F.softmax(neg_sim / META_TAU, dim=0)
    return weights, neg_idx


class ConfusionTracker:
    def __init__(self, num_classes, window=30, min_p=0.15):
        self.num_classes      = num_classes
        self.window           = window
        self.min_p            = min_p
        self.confusion_counts = np.zeros((num_classes, num_classes), dtype=np.float32)
        self.total_counts     = np.zeros(num_classes, dtype=np.float32)
        self._buffers         = [[] for _ in range(num_classes)]

    def update(self, class_idx, fake_features, prototypes, species_labels):
        correct_sim = F.cosine_similarity(
            fake_features, prototypes[species_labels[class_idx]].unsqueeze(0)
        ).item()
        confused_vec = np.zeros(self.num_classes, dtype=np.float32)
        for j, sp in enumerate(species_labels):
            if j == class_idx:
                continue
            wrong_sim = F.cosine_similarity(
                fake_features, prototypes[sp].unsqueeze(0)
            ).item()
            confused_vec[j] = float(wrong_sim > correct_sim)

        buf = self._buffers[class_idx]
        buf.append(confused_vec)
        if len(buf) > self.window:
            oldest = buf.pop(0)
            self.confusion_counts[class_idx] -= oldest
            self.total_counts[class_idx]     -= 1
        self.confusion_counts[class_idx] += confused_vec
        self.total_counts[class_idx]     += 1

    def confusion_rates(self, class_idx):
        total = max(self.total_counts[class_idx], 1.0)
        return np.clip(self.confusion_counts[class_idx] / total, 0.0, 1.0)

    def sample_class(self, rng):
        if self.total_counts.sum() > 0:
            mean_conf   = np.array([self.confusion_rates(i).mean()
                                    for i in range(self.num_classes)])
            mean_conf  += self.min_p
            class_probs = mean_conf / mean_conf.sum()
            return int(rng.choice(self.num_classes, p=class_probs))
        return int(rng.integers(self.num_classes))

    def print_heatmap(self, species_labels):
        short  = [s.split()[-1][:6] for s in species_labels]
        header = "          " + "  ".join(f"{s:>6}" for s in short)
        print("\n  Confusion rate heatmap:")
        print(header)
        for i, sp in enumerate(species_labels):
            rates = self.confusion_rates(i)
            row   = f"  {short[i]:<8}  " + "  ".join(
                f"{rates[j]*100:5.1f}" if j != i else "  --- " for j in range(len(species_labels))
            )
            print(row)
        print("")

def ddim_step_grad(latents, noise_pred, t, scheduler):
    alphas       = scheduler.alphas_cumprod.to(latents.device)
    alpha_t      = alphas[t]
    step_ratio   = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    t_prev       = max(int(t) - step_ratio, 0)
    alpha_t_prev = alphas[t_prev]
    pred_x0      = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
    pred_x0      = pred_x0.clamp(-1, 1)
    return alpha_t_prev.sqrt() * pred_x0 + (1 - alpha_t_prev).sqrt() * noise_pred


def generate_with_grad(unet, vae, cond_encoder, meta_single, device, scheduler_ref):
    cond     = cond_encoder(meta_single)
    uncond   = torch.zeros(1, 77, 768, device=device)
    latents  = torch.randn(1, 4, 64, 64, device=device)
    n_warmup = TOTAL_STEPS - N_REWARD_STEPS

    with torch.no_grad():
        for t in scheduler_ref.timesteps[:n_warmup]:
            set_attention_context(unet, uncond)
            noise_u    = unet(latents, t, encoder_hidden_states=uncond).sample
            set_attention_context(unet, cond)
            noise_c    = unet(latents, t, encoder_hidden_states=cond).sample
            noise_pred = noise_u + GUIDANCE_SCALE * (noise_c - noise_u)
            latents    = ddim_step_grad(latents, noise_pred, t, scheduler_ref)

    latents = latents.detach().requires_grad_(True)

    for t in scheduler_ref.timesteps[n_warmup:]:
        with torch.no_grad():
            set_attention_context(unet, uncond)
            noise_u = unet(latents, t, encoder_hidden_states=uncond).sample.detach()
        set_attention_context(unet, cond)
        noise_c    = unet(latents, t, encoder_hidden_states=cond).sample
        noise_pred = noise_u + GUIDANCE_SCALE * (noise_c - noise_u)
        latents    = ddim_step_grad(latents, noise_pred, t, scheduler_ref)

    image = vae.decode(latents / 0.18215).sample
    return F.interpolate(image.clamp(-1, 1), size=(256, 256),
                         mode="bilinear", align_corners=False)


def get_lambda_sep(step):
    if step >= WARMUP_STEPS:
        return LAMBDA_SEP_END
    t = step / WARMUP_STEPS
    return LAMBDA_SEP_START + t * (LAMBDA_SEP_END - LAMBDA_SEP_START)


def save_all_species(unet, vae, cond_encoder, species2idx, output_dir, device, step):
    folder = os.path.join(output_dir, "samples_by_species", f"step_{step}")
    os.makedirs(folder, exist_ok=True)
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)
    uncond = torch.zeros(1, 77, 768, device=device)

    for species_name, species_idx in sorted(species2idx.items(), key=lambda x: x[1]):
        meta = {
            "species":        torch.tensor([species_idx], device=device),
            "throat_color":   torch.tensor([0], device=device),
            "forehead_color": torch.tensor([0], device=device),
            "belly_color":    torch.tensor([0], device=device),
            "nape_color":     torch.tensor([0], device=device),
        }
        cond    = cond_encoder(meta)
        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                set_attention_context(unet, uncond)
                noise_u    = unet(latents, t, encoder_hidden_states=uncond).sample
                set_attention_context(unet, cond)
                noise_c    = unet(latents, t, encoder_hidden_states=cond).sample
                noise_pred = noise_u + 7.5 * (noise_c - noise_u)
                latents    = scheduler.step(noise_pred, t, latents).prev_sample
            img = vae.decode(latents / 0.18215).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")
        Image.fromarray(img).save(
            os.path.join(folder, f"{species_name.replace(' ', '_')}.png"))
    print(f"  Saved per-species samples {folder}", flush=True)

def train():
    assert torch.cuda.is_available(), "CUDA not available."
    device = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(seed=42)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for layer in lora_layers:
        layer.to(device)
    inject_metadata_into_attention(unet, device)

    print("Loading DINOv2...")
    dino_model, feat_dim = load_dino(DINO_MODEL, device)

    print("Building CUB dataset...")
    diff_dataset = CUBDataset(CSV_PATH, vae=vae, device=device)
    species2idx  = diff_dataset.species2idx

    assert sorted(species2idx.keys()) == SPECIES_LABELS, \
        f"Species mismatch: {sorted(species2idx.keys())}"

    class_counts = diff_dataset.df["species_name"].value_counts().to_dict()
    diff_loader  = DataLoader(
        diff_dataset, batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            torch.DoubleTensor([1.0 / class_counts[s]
                                for s in diff_dataset.df["species_name"]]),
            len(diff_dataset), replacement=True),
    )

    print(f"Loading Stage 1 checkpoint from {LORA_CKPT}...")
    stage1_ckpt     = torch.load(LORA_CKPT, map_location=device, weights_only=False)
    species_emb_dim = stage1_ckpt.get("species_emb_dim", 32)
    field_configs   = cub_field_configs(diff_dataset, species_emb_dim=species_emb_dim)
    cond_encoder    = MetadataConditionEncoder(
        field_configs, hidden_dim=256, final_dim=768).to(device)
    for layer, state in zip(lora_layers, stage1_ckpt["lora_layers"]):
        layer.load_state_dict(state)
    
    cond_encoder.load_state_dict(stage1_ckpt["cond_encoder"])
    cond_encoder.eval()

    for p in cond_encoder.parameters():
        p.requires_grad = False

    print("  Stage 1 weights loaded.")
    print("  MetadataConditionEncoder frozen. Training LoRA only.")

    print("Computing / loading DINOv2 class prototypes...")
    class_prototypes = compute_or_load_dino_prototypes(
        CSV_PATH, SPECIES_LABELS, dino_model, device, PROTO_CACHE)

    print("Building metadata similarity matrix from trained encoder...")
    full_df    = pd.read_csv(CSV_PATH)
    full_df    = full_df[full_df["full_path"].apply(os.path.exists)].reset_index(drop=True)
    meta_sim   = build_metadata_sim_matrix(
        cond_encoder, SPECIES_LABELS, species2idx, full_df, device)

    K = NUM_SPECIES
    print("\n  Top 5 hardest metadata pairs (will receive highest contrastive weight):")
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((meta_sim[i, j].item(), SPECIES_LABELS[i], SPECIES_LABELS[j]))
    pairs.sort(reverse=True)
    for sim_val, a, b in pairs[:5]:
        print(f"    {sim_val:.4f}  {a}  ↔  {b}")
    print("")

    df_rows_by_species = {sp: full_df[full_df["species_name"] == sp]
                          for sp in SPECIES_LABELS}

    confusion_tracker = ConfusionTracker(NUM_SPECIES, CONFUSION_WINDOW, CONFUSION_MIN_P)

    gen_params = [p for l in lora_layers for p in l.parameters() if p.requires_grad]
    optimizer_G = optim.AdamW(gen_params, lr=LR_G)

    reward_scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    reward_scheduler.set_timesteps(TOTAL_STEPS)

    print(f"\nStarting Stage 2 — Metadata-Weighted DINOv2 Contrastive Loss")
    print(f"  Feature backbone  : {DINO_MODEL} ({feat_dim}D, frozen)")
    print(f"  Neg. weighting    : metadata summary-token cosine similarity (τ={META_TAU})")
    print(f"  lambda_diff       = {LAMBDA_DIFF}")
    print(f"  lambda_sep        : {LAMBDA_SEP_START} {LAMBDA_SEP_END} over {WARMUP_STEPS} steps")
    print(f"  margin δ          = {MARGIN}")
    print(f"  Contrastive every : {DISC_INTERVAL} steps\n")

    step         = 0
    sep_loss_log = {sp: [] for sp in SPECIES_LABELS}
    pbar         = tqdm(total=STEPS, desc="Stage 2 Meta-Contrastive")

    while step < STEPS:
        for batch in diff_loader:
            if step >= STEPS:
                break

            latents, noise, noisy_latents, t, meta = batch
            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)
            meta_batch    = {k: v.to(device) for k, v in meta.items()}

            cond         = cond_encoder(meta_batch)
            dropout_mask = (torch.rand(cond.shape[0], 1, 1, device=device) < 0.15)
            cond_dropped = torch.where(dropout_mask, torch.zeros_like(cond), cond)
            set_attention_context(unet, cond_dropped)
            noise_pred = unet(noisy_latents, t,
                              encoder_hidden_states=cond_dropped).sample
            diff_loss  = F.mse_loss(noise_pred, noise)

            optimizer_G.zero_grad()
            (LAMBDA_DIFF * diff_loss).backward()

            sep_val   = None
            class_log = None

            if step % DISC_INTERVAL == 0:
                class_idx = confusion_tracker.sample_class(rng)
                class_log = SPECIES_LABELS[class_idx]

                row = df_rows_by_species[class_log].sample(1).iloc[0]
                meta_single = {
                    "species":        torch.tensor([species2idx[row["species_name"]]], device=device),
                    "throat_color":   torch.tensor([int(row["throat_color"])],   device=device),
                    "forehead_color": torch.tensor([int(row["forehead_color"])], device=device),
                    "belly_color":    torch.tensor([int(row["belly_color"])],    device=device),
                    "nape_color":     torch.tensor([int(row["nape_color"])],     device=device),
                }

                torch.cuda.empty_cache()
                fake_img = generate_with_grad(
                    unet, vae, cond_encoder, meta_single, device, reward_scheduler)

                fake_dino_in = preprocess_for_dino(fake_img, device)
                fake_feats   = F.normalize(dino_model(fake_dino_in), dim=-1)

                confusion_tracker.update(
                    class_idx, fake_feats.detach(), class_prototypes, SPECIES_LABELS)

                weights, neg_indices = get_neg_weights(meta_sim, class_idx, device)

                correct_proto = class_prototypes[class_log].unsqueeze(0)
                correct_sim   = F.cosine_similarity(fake_feats, correct_proto.detach())

                per_neg_losses = []
                for w, j in zip(weights, neg_indices):
                    neg_proto_j = class_prototypes[SPECIES_LABELS[j]].unsqueeze(0)
                    neg_sim_j   = F.cosine_similarity(fake_feats, neg_proto_j.detach())
                    pair_loss   = (MARGIN - correct_sim + neg_sim_j).clamp(min=0)
                    per_neg_losses.append(w * pair_loss)

                loss_sep   = torch.stack(per_neg_losses).sum()
                lambda_sep = get_lambda_sep(step)
                (lambda_sep * loss_sep).backward()

                torch.cuda.empty_cache()
                sep_val = float(loss_sep.detach())
                sep_loss_log[class_log].append(sep_val)

            torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
            optimizer_G.step()

            step += 1
            pbar.update(1)

            if step % 50 == 0:
                lam = get_lambda_sep(step)
                log = f"Step {step}/{STEPS}  diff={diff_loss.item():.4f}  λ_sep={lam:.4f}"
                if sep_val is not None:
                    log += f"  sep={sep_val:.4f}  species={class_log.split()[-1]}"
                print(log, flush=True)

            if step % LOG_INTERVAL == 0:
                print(f"\n  DINOv2 contrastive loss per species (last 20, lower = better):")
                for sp in SPECIES_LABELS:
                    recent = sep_loss_log[sp][-20:]
                    if recent:
                        print(f"    {sp.split()[-1]:<12}: {sum(recent)/len(recent):.4f}")
                confusion_tracker.print_heatmap(SPECIES_LABELS)

            if step % SAMPLE_INTERVAL == 0:
                save_all_species(unet, vae, cond_encoder, species2idx,
                                 OUT_DIR, device, step)

            if step % CKPT_INTERVAL == 0:
                ckpt_path = os.path.join(OUT_DIR, f"lora_cub_meta_step{step}.pth")
                torch.save({
                    "lora_layers":      [l.state_dict() for l in lora_layers],
                    "cond_encoder":     cond_encoder.state_dict(),
                    "species2idx":      species2idx,
                    "field_configs":    stage1_ckpt.get("field_configs"),
                    "lora_rank":        LORA_RANK,
                    "step":             step,
                    "lambda_diff":      LAMBDA_DIFF,
                    "lambda_sep_end":   LAMBDA_SEP_END,
                    "margin":           MARGIN,
                    "meta_tau":         META_TAU,
                    "backbone":         DINO_MODEL,
                    "species_emb_dim":  species_emb_dim,
                }, ckpt_path)
                print(f"  Saved checkpoint {ckpt_path}", flush=True)

    pbar.close()

    final_path = os.path.join(OUT_DIR, "lora_cub_meta.pth")
    torch.save({
        "lora_layers":      [l.state_dict() for l in lora_layers],
        "cond_encoder":     cond_encoder.state_dict(),
        "species2idx":      species2idx,
        "field_configs":    stage1_ckpt.get("field_configs"),
        "lora_rank":        LORA_RANK,
        "step":             step,
        "lambda_diff":      LAMBDA_DIFF,
        "lambda_sep_end":   LAMBDA_SEP_END,
        "margin":           MARGIN,
        "meta_tau":         META_TAU,
        "backbone":         DINO_MODEL,
        "species_emb_dim":  species_emb_dim,
    }, final_path)

    print(f"\nDone. Final weights {final_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    train()
