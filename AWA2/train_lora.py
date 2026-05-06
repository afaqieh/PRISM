import os
import random
import json
import subprocess
import sys
import shlex
from tqdm import tqdm
from pathlib import Path

import numpy as np

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from torch.utils.data import DataLoader, WeightedRandomSampler

from AWA2.dataset import AWA2Dataset
from AWA2.metadata_conditioning import MetadataConditionEncoder, awa2_field_configs
from ..lora_utils import LoRALinear, apply_lora_to_unet, inject_metadata_into_attention

MODEL_NAME          = "runwayml/stable-diffusion-v1-5"
TRAIN_CSV           = "./data/awa2_train.csv"
OUT_DIR             = "results/awa2_drop"
GENERATED_ROOT_BASE = "results/awa2_generated"

STEPS               = 20000
BATCH_SIZE          = 4
LR                  = 3e-5
LORA_RANK           = 16
CFG_DROPOUT         = 0.15

SEEDS       = [42]
NUM_WORKERS = 0

GENERATE_SCRIPT = "generate_fid_set_awa2.py"
EVAL_SCRIPT     = "evaluate_metrics.py"
RUN_GENERATION  = False
RUN_EVALUATION  = False


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_attention_context(unet, context):
    for _, module in unet.named_modules():
        if hasattr(module, "attn2"):
            module.attn2.metadata_context = context


def build_dataloader(dataset, seed):
    class_counts = dataset.df["class_name"].value_counts().to_dict()
    weights = dataset.df["class_name"].map(lambda x: 1.0 / class_counts[x]).tolist()

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=sampler_generator,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=False,
    )


def save_run_metadata(seed_out_dir, seed):
    meta = {
        "seed": seed,
        "model_name": MODEL_NAME,
        "train_csv": TRAIN_CSV,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lora_rank": LORA_RANK,
        "cfg_dropout": CFG_DROPOUT,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    with open(os.path.join(seed_out_dir, "run_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def run_subprocess(cmd, step_name):
    print(f"\nRunning {step_name}:")
    print(" ", " ".join(shlex.quote(str(x)) for x in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")


def run_generation(seed: int, ckpt_path: str):
    generated_root = os.path.join(GENERATED_ROOT_BASE, f"seed_{seed}")
    os.makedirs(generated_root, exist_ok=True)
    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--lora_path",      ckpt_path,
        "--generated_root", generated_root,
        "--seed",           str(seed),
    ]
    run_subprocess(cmd, f"generation for seed {seed}")
    return generated_root


def run_evaluation(seed: int, generated_root: str):
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--generated_root", generated_root,
        "--seed",           str(seed),
    ]
    run_subprocess(cmd, f"evaluation for seed {seed}")
    return os.path.join(generated_root, "metrics")


def train_one_seed(seed: int):
    print("=" * 80)
    print(f"Starting AWA2 run for seed = {seed}")
    print("=" * 80)

    seed_everything(seed)

    device       = "cuda" if torch.cuda.is_available() else "cpu"
    seed_out_dir = os.path.join(OUT_DIR, f"seed_{seed}")
    os.makedirs(seed_out_dir, exist_ok=True)
    save_run_metadata(seed_out_dir, seed)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    print("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    for p in unet.parameters():
        p.requires_grad = False

    print("Building dataset...")
    dataset    = AWA2Dataset(TRAIN_CSV, vae, device=device, base_seed=seed)
    dataloader = build_dataloader(dataset, seed)

    print("Building metadata encoder...")
    field_configs = awa2_field_configs(dataset)
    cond_encoder  = MetadataConditionEncoder(
        field_configs, hidden_dim=256, final_dim=768
    ).to(device)

    inject_metadata_into_attention(unet, device)

    print("Applying LoRA...")
    lora_layers = apply_lora_to_unet(unet, r=LORA_RANK)
    for layer in lora_layers:
        layer.to(device)

    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) +
        [p for layer in lora_layers for p in layer.parameters()],
        lr=LR,
    )

    print(f"Training for {STEPS} steps...")
    step         = 0
    loss_history = []
    pbar         = tqdm(total=STEPS, desc=f"seed={seed}", unit="step")

    while step < STEPS:
        for latents, noise, noisy_latents, t, meta in dataloader:
            if step >= STEPS:
                break

            latents       = latents.to(device)
            noise         = noise.to(device)
            noisy_latents = noisy_latents.to(device)
            t             = t.squeeze().to(device)
            meta_batch    = {k: v.to(device) for k, v in meta.items()}

            cond     = cond_encoder(meta_batch)

            cfg_mask = (torch.rand(cond.shape[0], 1, 1, device=device) < CFG_DROPOUT)
            cond     = torch.where(cfg_mask, torch.zeros_like(cond), cond)

            set_attention_context(unet, cond)
            noise_pred = unet(noisy_latents, t, encoder_hidden_states=cond).sample
            loss       = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step       += 1
            loss_value  = float(loss.item())
            loss_history.append({"step": step, "loss": loss_value})
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_value:.6f}")

            if step % 500 == 0:
                print(f"[seed={seed}] Step {step}/{STEPS}  loss={loss_value:.6f}")

    pbar.close()

    ckpt_path = os.path.join(seed_out_dir, f"lora_awa2_seed_{seed}.pth")
    torch.save(
        {
            "seed":         seed,
            "lora_layers":  [l.state_dict() for l in lora_layers],
            "cond_encoder": cond_encoder.state_dict(),
            "field_configs": field_configs,
            "class2idx":    dataset.class2idx,
            "attr_names":   dataset.attr_names,
            "lora_rank":    LORA_RANK,
            "loss_history": loss_history,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")
    return {"seed": seed, "checkpoint": ckpt_path, "seed_out_dir": seed_out_dir}


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(GENERATED_ROOT_BASE).mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in SEEDS:
        result = {
            "seed": seed, "train_ok": False, "generate_ok": False,
            "eval_ok": False, "checkpoint": None, "generated_root": None,
            "eval_root": None, "error": None,
        }

        try:
            train_info = train_one_seed(seed)
            result["train_ok"]   = True
            result["checkpoint"] = train_info["checkpoint"]

            if RUN_GENERATION:
                generated_root       = run_generation(seed, train_info["checkpoint"])
                result["generate_ok"]    = True
                result["generated_root"] = generated_root

            if RUN_EVALUATION and result["generated_root"]:
                eval_root        = run_evaluation(seed, result["generated_root"])
                result["eval_ok"]    = True
                result["eval_root"]  = eval_root

        except Exception as e:
            result["error"] = str(e)
            print(f"\nSeed {seed} failed: {e}\n")

        all_results.append(result)

        summary_path = os.path.join(OUT_DIR, f"seed_{seed}", "pipeline_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)

    final_path = os.path.join(OUT_DIR, "all_seeds_summary.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll runs finished:")
    for r in all_results:
        print(
            f"seed={r['seed']} | train={r['train_ok']} | "
            f"generate={r['generate_ok']} | eval={r['eval_ok']} | "
            f"ckpt={r['checkpoint']}"
        )


if __name__ == "__main__":
    main()
