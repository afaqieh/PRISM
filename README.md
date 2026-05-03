# PRISM: Class-Consistent Image Generation from Tabular Data

Official implementation of PRISM, a tabular-to-image generation framework that replaces CLIP conditioning with structured metadata tokens, enabling class-consistent image synthesis across fine-grained benchmarks.

---

## Setup

```bash
git clone <repository-url>
cd PRISM
conda env create -f environment.yml
conda activate prism
```

---

## Repository Structure

The repository is organized by dataset. Each dataset contains implementations of PRISM and baseline methods for comparison.

```text
PRISM/
├── CUB-200-2011/
├── HAM10000/
├── PlantVillage/
└── lora_utils.py
```

Inside each dataset folder:

```text
{DATASET}/
├── PRISM/              # Our method
├── CLIP/               # CLIP baseline
├── FT-CLIP/            # Fine-tuned CLIP baseline
└── BioCLIP / PLIP      # Domain-specific baseline
```

---

## Running PRISM

Replace `{DATASET}` with one of the dataset folders, for example `CUB-200-2011`, `HAM10000`, or `PlantVillage`.

### 1. Train: Stage 1 Tabular Conditioning

```bash
python {DATASET}/PRISM/train_lora.py
```

### 2. Refine: Stage 2 Similarity-Aware Class Refinement

```bash
python {DATASET}/PRISM/train_metadata_contrastive.py
```

### 3. Generate Images

```bash
python {DATASET}/PRISM/generate_set.py \
    --lora_path results/lora.pth \
    --output_dir results/generated
```

### 4. Evaluate

```bash
python {DATASET}/evaluate_class_consistency.py --generated_root results/generated
python {DATASET}/evaluate_CLIPIQA.py           --generated_root results/generated
python {DATASET}/compute_fid.py                --generated_root results/generated
```