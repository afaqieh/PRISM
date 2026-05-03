import os
import random
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
from ...lora_utils import LoRALinear, inject_metadata_into_attention, apply_lora_to_unet
import timm

from PlantVillage.dataset import PlantVillageDataset
from PlantVillage.metadata_conditioning import MetadataConditionEncoder

MODEL_NAME   = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV    = "./data/plantvillage_train.csv"
LORA_CKPT    = "results/plantvillage_stage1/seed_42/lora_plantvillage_seed_42_step30000.pth"
CNX_MODEL    = "convnext_large.fb_in22k"
OUT_DIR      = "results/plantvillage_stage2_meta_cnx/seed_42_lr"
PROTO_CACHE  = "cnx_prototypes_plantvillage.pt"

STEPS      = 2000
BATCH_SIZE = 4
LR_G       = 1e-4
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

CLASS_LABELS = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
NUM_CLASSES = len(CLASS_LABELS)

CONDITION_MAP = {
    "Pepper__bell___Bacterial_spot":               ("Pepper", "Bacterial_spot"),
    "Pepper__bell___healthy":                      ("Pepper", "healthy"),
    "Potato___Early_blight":                       ("Potato", "Early_blight"),
    "Potato___Late_blight":                        ("Potato", "Late_blight"),
    "Potato___healthy":                            ("Potato", "healthy"),
    "Tomato_Bacterial_spot":                       ("Tomato", "Bacterial_spot"),
    "Tomato_Early_blight":                         ("Tomato", "Early_blight"),
    "Tomato_Late_blight":                          ("Tomato", "Late_blight"),
    "Tomato_Leaf_Mold":                            ("Tomato", "Leaf_Mold"),
    "Tomato_Septoria_leaf_spot":                   ("Tomato", "Septoria_leaf_spot"),
    "Tomato_Spider_mites_Two_spotted_spider_mite": ("Tomato", "Spider_mites"),
    "Tomato__Target_Spot":                         ("Tomato", "Target_Spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       ("Tomato", "Yellow_Leaf_Curl_Virus"),
    "Tomato__Tomato_mosaic_virus":                 ("Tomato", "mosaic_virus"),
    "Tomato_healthy":                              ("Tomato", "healthy"),
}


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context

_CNX_MEAN = torch.tensor([0.485, 0.456, 0.406])
_CNX_STD  = torch.tensor([0.229, 0.224, 0.225])

_CNX_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=_CNX_MEAN.tolist(), std=_CNX_STD.tolist()),
])


def load_convnext(model_name, device):
    print(f"  Loading frozen ConvNeXt-Large ({model_name}) via timm...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    with torch.no_grad():
        feat_dim = model(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    print(f"  ConvNeXt loaded — feature dim: {feat_dim}D")
    return model, feat_dim


def preprocess_for_convnext(img_tensor, device):
    x    = (img_tensor + 1.0) / 2.0
    x    = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = _CNX_MEAN.view(1, 3, 1, 1).to(device)
    std  = _CNX_STD.view(1, 3, 1, 1).to(device)
    return (x - mean) / std


def compute_or_load_cnx_prototypes(csv_path, class_labels,
                                   cnx_model, device,
                                   cache_path=PROTO_CACHE):
    if os.path.exists(cache_path):
        print(f"  Loading ConvNeXt prototypes from cache: {cache_path}")
        protos = torch.load(cache_path, map_location=device, weights_only=False)
        print(f"  Loaded {len(protos)} class prototypes.")
        return protos

    print("  Computing ConvNeXt prototypes from real training images...")
    df = pd.read_csv(csv_path).reset_index(drop=True)

    protos = {}
    cnx_model.eval()
    for cls in tqdm(class_labels, desc="  Building ConvNeXt prototypes"):
        cls_df = df[df["class_name"] == cls]
        feats  = []
        for _, row in cls_df.iterrows():
            img_path = row.get("full_path", "")
            if not img_path or not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                x   = _CNX_TRANSFORM(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = F.normalize(cnx_model(x), dim=-1)
                feats.append(f.squeeze(0).cpu())
            except Exception:
                continue
        if feats:
            proto = torch.stack(feats).mean(0)
            protos[cls] = F.normalize(proto, dim=0).to(device)
            print(f"    {cls:<55}: {len(feats)} imgs", flush=True)
        else:
            print(f"    WARNING: no images found for {cls}", flush=True)

    torch.save(protos, cache_path)
    print(f"  Prototypes saved {cache_path}")
    return protos

def build_metadata_sim_matrix(cond_encoder, class_labels, class2idx,
                               plant2idx, condition2idx, device):
    embeddings = []
    cond_encoder.eval()
    with torch.no_grad():
        for cls in class_labels:
            plant_name, cond_name = CONDITION_MAP[cls]
            meta_batch = {
                "plant":     torch.tensor([plant2idx[plant_name]],    device=device),
                "condition": torch.tensor([condition2idx[cond_name]], device=device),
            }
            cond    = cond_encoder(meta_batch)
            summary = cond[0, 2, :]
            summary = F.normalize(summary, dim=0)
            embeddings.append(summary.cpu())

    E   = torch.stack(embeddings)
    sim = E @ E.T
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

    def update(self, class_idx, fake_features, prototypes, class_labels):
        correct_sim = F.cosine_similarity(
            fake_features, prototypes[class_labels[class_idx]].unsqueeze(0)
        ).item()
        confused_vec = np.zeros(self.num_classes, dtype=np.float32)
        for j, lbl in enumerate(class_labels):
            if j == class_idx:
                continue
            wrong_sim = F.cosine_similarity(
                fake_features, prototypes[lbl].unsqueeze(0)
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

    def print_heatmap(self, class_labels):
        short = [lbl.split("_")[-1][:6] for lbl in class_labels]
        header = "        " + "  ".join(f"{s:>6}" for s in short)
        print("\n  Confusion rate heatmap:")
        print(header)
        for i, lbl in enumerate(class_labels):
            rates = self.confusion_rates(i)
            row   = f"  {lbl.split('_')[-1][:6]:<6}  " + "  ".join(
                f"{rates[j]*100:5.1f}" if j != i else "  --- "
                for j in range(len(class_labels))
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

def save_all_classes(unet, vae, cond_encoder, class2idx, plant2idx, condition2idx,
                     output_dir, device, step):
    folder = os.path.join(output_dir, "samples_by_class", f"step_{step}")
    os.makedirs(folder, exist_ok=True)
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    scheduler.set_timesteps(50)
    uncond = torch.zeros(1, 77, 768, device=device)

    for cls_name, cls_idx in sorted(class2idx.items(), key=lambda x: x[1]):
        plant_name, cond_name = CONDITION_MAP[cls_name]
        meta = {
            "plant":     torch.tensor([plant2idx[plant_name]],    device=device),
            "condition": torch.tensor([condition2idx[cond_name]], device=device),
        }
        cond    = cond_encoder(meta)
        latents = torch.randn(1, 4, 64, 64, device=device)

        with torch.no_grad():
            for t in scheduler.timesteps:
                set_attention_context(unet, uncond)
                n_u = unet(latents, t, encoder_hidden_states=uncond).sample
                set_attention_context(unet, cond)
                n_c = unet(latents, t, encoder_hidden_states=cond).sample
                latents = scheduler.step(
                    n_u + 7.5 * (n_c - n_u), t, latents
                ).prev_sample
            img = vae.decode(latents / 0.18215).sample

        img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")
        Image.fromarray(img).save(os.path.join(folder, f"{cls_name}.png"))

    print(f"  Saved per-class samples {folder}", flush=True)

def train():
    assert torch.cuda.is_available(), "CUDA not available."
    device = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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

    print("Loading ConvNeXt-Large...")
    cnx_model, feat_dim = load_convnext(CNX_MODEL, device)

    print("Building PlantVillage dataset...")
    diff_dataset  = PlantVillageDataset(TRAIN_CSV, vae=vae, device=device)
    class2idx     = diff_dataset.class2idx
    plant2idx     = diff_dataset.plant2idx
    condition2idx = diff_dataset.condition2idx

    assert sorted(class2idx.keys()) == sorted(CLASS_LABELS), \
        f"Class mismatch: {sorted(class2idx.keys())}"

    class_counts = diff_dataset.df["class_name"].value_counts().to_dict()
    diff_loader  = DataLoader(
        diff_dataset, batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(
            torch.DoubleTensor([1.0 / class_counts[c]
                                for c in diff_dataset.df["class_name"]]),
            len(diff_dataset), replacement=True),
    )

    print(f"Loading Stage 1 checkpoint from {LORA_CKPT}...")
    stage1_ckpt   = torch.load(LORA_CKPT, map_location=device, weights_only=False)
    field_configs = stage1_ckpt["field_configs"]

    cond_encoder = MetadataConditionEncoder(
        field_configs, hidden_dim=256, final_dim=768).to(device)

    for layer, state in zip(lora_layers, stage1_ckpt["lora_layers"]):
        layer.load_state_dict(state)
    cond_encoder.load_state_dict(stage1_ckpt["cond_encoder"])
    for p in cond_encoder.parameters():
        p.requires_grad = False
    cond_encoder.eval()
    print("  Stage 1 weights loaded. cond_encoder frozen.")

    print("Computing / loading ConvNeXt class prototypes...")
    class_prototypes = compute_or_load_cnx_prototypes(
        TRAIN_CSV, CLASS_LABELS, cnx_model, device, PROTO_CACHE)

    print("Building metadata similarity matrix...")
    meta_sim = build_metadata_sim_matrix(
        cond_encoder, CLASS_LABELS, class2idx,
        plant2idx, condition2idx, device)

    print("\n  Top hardest metadata pairs (highest contrastive weight):")
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(i + 1, NUM_CLASSES):
            pairs.append((meta_sim[i, j].item(), CLASS_LABELS[i], CLASS_LABELS[j]))
    pairs.sort(reverse=True)
    for sim_val, a, b in pairs[:5]:
        a_short = a.split("_")[-1]
        b_short = b.split("_")[-1]
        print(f"    {sim_val:.4f}  {a_short}  ↔  {b_short}")
    print("")

    full_df           = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
    df_rows_by_class  = {cls: full_df[full_df["class_name"] == cls]
                         for cls in CLASS_LABELS}
    
    confusion_tracker = ConfusionTracker(NUM_CLASSES, CONFUSION_WINDOW, CONFUSION_MIN_P)

    gen_params  = [p for l in lora_layers for p in l.parameters()]
    optimizer_G = optim.AdamW(gen_params, lr=LR_G)

    reward_scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    reward_scheduler.set_timesteps(TOTAL_STEPS)

    print(f"\nStarting Stage 2 — Metadata-Weighted ConvNeXt-Large Contrastive Loss")
    print(f"  Backbone    : {CNX_MODEL} ({feat_dim}D, frozen)")
    print(f"  Neg weights : metadata summary similarity (τ={META_TAU})")
    print(f"  lambda_diff : {LAMBDA_DIFF}")
    print(f"  lambda_sep  : {LAMBDA_SEP_START} {LAMBDA_SEP_END} over {WARMUP_STEPS} steps")
    print(f"  margin δ    : {MARGIN}")
    print(f"  Contrastive every {DISC_INTERVAL} steps\n")

    step         = 0
    sep_loss_log = {cls: [] for cls in CLASS_LABELS}
    pbar         = tqdm(total=STEPS, desc="Stage 2 PlantVillage Meta-Contrastive")

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
                class_log = CLASS_LABELS[class_idx]

                row = df_rows_by_class[class_log].sample(1).iloc[0]
                plant_name, cond_name = CONDITION_MAP[class_log]
                meta_single = {
                    "plant":     torch.tensor([plant2idx[plant_name]],    device=device),
                    "condition": torch.tensor([condition2idx[cond_name]], device=device),
                }

                torch.cuda.empty_cache()
                fake_img = generate_with_grad(
                    unet, vae, cond_encoder, meta_single, device, reward_scheduler)

                fake_cnx_in = preprocess_for_convnext(fake_img, device)
                fake_feats  = F.normalize(cnx_model(fake_cnx_in), dim=-1)

                confusion_tracker.update(
                    class_idx, fake_feats.detach(), class_prototypes, CLASS_LABELS)

                weights, neg_indices = get_neg_weights(meta_sim, class_idx, device)

                correct_proto = class_prototypes[class_log].unsqueeze(0)
                correct_sim   = F.cosine_similarity(fake_feats, correct_proto.detach())

                per_neg_losses = []
                for w, j in zip(weights, neg_indices):
                    neg_proto_j = class_prototypes[CLASS_LABELS[j]].unsqueeze(0)
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
                    log += f"  sep={sep_val:.4f}  class={class_log}"
                print(log, flush=True)

            if step % LOG_INTERVAL == 0:
                print(f"\n  ConvNeXt contrastive loss per class (last 20, lower = better):")
                for cls in CLASS_LABELS:
                    recent = sep_loss_log[cls][-20:]
                    if recent:
                        print(f"    {cls.split('_')[-1]:<20}: {sum(recent)/len(recent):.4f}")
                confusion_tracker.print_heatmap(CLASS_LABELS)

            if step % SAMPLE_INTERVAL == 0:
                save_all_classes(unet, vae, cond_encoder, class2idx,
                                 plant2idx, condition2idx, OUT_DIR, device, step)

            if step % CKPT_INTERVAL == 0:
                ckpt_path = os.path.join(OUT_DIR, f"lora_plantvillage_meta_step{step}.pth")
                torch.save({
                    "lora_layers":     [l.state_dict() for l in lora_layers],
                    "cond_encoder":    cond_encoder.state_dict(),
                    "field_configs":   field_configs,
                    "class2idx":       class2idx,
                    "plant2idx":       plant2idx,
                    "condition2idx":   condition2idx,
                    "lora_rank":       LORA_RANK,
                    "step":            step,
                    "lambda_diff":     LAMBDA_DIFF,
                    "lambda_sep_end":  LAMBDA_SEP_END,
                    "margin":          MARGIN,
                    "meta_tau":        META_TAU,
                    "backbone":        CNX_MODEL,
                }, ckpt_path)
                print(f"  Saved checkpoint {ckpt_path}", flush=True)

    pbar.close()

    final_path = os.path.join(OUT_DIR, "lora_plantvillage_meta.pth")
    torch.save({
        "lora_layers":     [l.state_dict() for l in lora_layers],
        "cond_encoder":    cond_encoder.state_dict(),
        "field_configs":   field_configs,
        "class2idx":       class2idx,
        "plant2idx":       plant2idx,
        "condition2idx":   condition2idx,
        "lora_rank":       LORA_RANK,
        "step":            step,
        "lambda_diff":     LAMBDA_DIFF,
        "lambda_sep_end":  LAMBDA_SEP_END,
        "margin":          MARGIN,
        "meta_tau":        META_TAU,
        "backbone":        CNX_MODEL,
    }, final_path)
    print(f"\nDone. Final weights {final_path}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available."
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    train()
