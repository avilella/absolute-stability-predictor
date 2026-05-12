# ESM3ΔG & SaProtΔG — Protein Stability Prediction

Fine-tuned ESM3 and SaProt models for predicting per-residue protein stability (ΔG) and mutational effects (ΔΔG) directly from structure files (PDB/CIF).

## Models

| Model | Base | Parameters | Weights |
|---|---|---|---|
| **ESM3ΔG** | ESM3 (EvolutionaryScale) | LoRA r=4 + stability head | [Yehlin/absolute-stability](https://huggingface.co/Yehlin/absolute-stability) |
| **SaProtΔG** | SaProt 650M (Westlake) | LoRA r=4 + stability head | [Yehlin/absolute-stability](https://huggingface.co/Yehlin/absolute-stability) |

Both models are fine-tuned on a combined dataset of experimental ΔG and ΔΔG measurements (K50dG, DMSv4/v5/v7). Ensemble prediction over 3 checkpoints is recommended.

Augmented variants (`*_augmented_*.ckpt`) are trained with additional data augmentation.

## Installation

> **Important:** Install PyTorch with the correct CUDA version for your driver **before** installing this package. If you install the package first, pip may pull in a PyTorch build incompatible with your GPU driver.

### Step 1 — Create a conda environment

```bash
conda create -n stability python=3.10 -y
conda activate stability
```

### Step 2 — Install PyTorch (match your CUDA driver)

Check your driver's maximum supported CUDA version with `nvidia-smi`, then install the matching wheel:

```bash
# CUDA 12.4 (driver >= 550.x)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 (driver >= 530.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available before continuing:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Should print: True 12.4  (or whichever version you installed)
```

### Step 3 — Install this package

```bash
pip install git+https://github.com/yehlincho/absolute-stability-predictor.git
```

Or from source:

```bash
git clone https://github.com/yehlincho/absolute-stability-predictor.git
cd absolute-stability-predictor
pip install -e .
```

**ESM3** must be installed separately from [EvolutionaryScale](https://github.com/evolutionaryScale/esm):
```bash
pip install git+https://github.com/evolutionaryScale/esm.git
```

**Foldseek** (required for SaProtΔG only) — download the binary for your platform from [steineggerlab/foldseek](https://github.com/steineggerlab/foldseek/releases) and place it at `bin/foldseek`.

## Download Weights

Weights are hosted on Hugging Face: **[Yehlin/absolute-stability](https://huggingface.co/Yehlin/absolute-stability)**

### Option A — Python (recommended)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Yehlin/absolute-stability",
    local_dir=".",          # downloads into esm3dg_weights/ and saprotdg_weights/
    token="hf_...",         # required if repo is private
)
```

### Option B — Manual

Download individual files from [huggingface.co/Yehlin/absolute-stability](https://huggingface.co/Yehlin/absolute-stability) and place them as follows:

```
esm3dg_weights/
  ESM3dG_weights_1_lora.ckpt
  ESM3dG_weights_2_lora.ckpt
  ESM3dG_weights_3_lora.ckpt
  ESM3dG_weights_augmented_1_lora.ckpt   # augmented ensemble
  ESM3dG_weights_augmented_2_lora.ckpt
  ESM3dG_weights_augmented_3_lora.ckpt

saprotdg_weights/
  SaProtdG_weights_1_lora.ckpt
  SaProtdG_weights_2_lora.ckpt
  SaProtdG_weights_3_lora.ckpt
  SaProtdG_weights_augmented_1_lora.ckpt
  SaProtdG_weights_augmented_2_lora.ckpt
  SaProtdG_weights_augmented_3_lora.ckpt
```

## Quick Start

### ESM3ΔG

```python
from ESM3ABS import ESM3ABS, ESM3ABS_predict

# Load model
model = ESM3ABS("esm3dg_weights/ESM3dG_weights_1_lora.ckpt")

# Predict ΔG for a structure
pred_dg_per_res, pred_dg_avg, sequence = ESM3ABS_predict(
    model,
    pdb_path="examples/nanobody_1zvh.cif",
    chain_id="A",
)
print(f"Average ΔG: {pred_dg_avg[0]:.2f} kcal/mol")
```

### SaProtΔG

```python
from SaProtABS import SaProtABS, SaProtABS_predict

# Load model
model = SaProtABS("saprotdg_weights/SaProtdG_weights_1_lora.ckpt")

# Predict ΔG for a structure
pred_dg_per_res, pred_dg_avg, combined_seq = SaProtABS_predict(
    model,
    pdb_path="examples/nanobody_1zvh.cif",
    chain_id="A",
    foldseek_path="bin/foldseek",
)
print(f"Average ΔG: {pred_dg_avg[0]:.2f} kcal/mol")
```

### Ensemble Prediction (recommended)

```python
from SaProtABS import SaProtABS, SaProtABS_predict

WEIGHTS = [
    "saprotdg_weights/SaProtdG_weights_1_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_2_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_3_lora.ckpt",
]

models = [SaProtABS(w) for w in WEIGHTS]
preds = [SaProtABS_predict(m, "examples/nanobody_1zvh.cif", "A")[1][0] for m in models]
ensemble_avg = sum(preds) / len(preds)
print(f"Ensemble ΔG: {ensemble_avg:.2f} kcal/mol")
```

The same pattern applies to ESM3ABS.

## Mutational Scanning (ΔΔG)

```python
ddg_scan, scaled_ddg_scan, sequence = ESM3ABS_predict(
    model,
    pdb_path="examples/nanobody_1zvh.cif",
    chain_id="A",
    ddg_scanning=True,
)
# ddg_scan shape: (L, 20, 1, L) — all single-point mutations
```

## API Reference

### `ESM3ABS(weights_path)`
Loads ESM3 with LoRA weights and a stability head.

### `ESM3ABS_predict(model, pdb_path, chain_id="A", ddg_scanning=False, sigmoid_on=False)`

| Return | Shape | Description |
|---|---|---|
| `pred_dg_per_res` | `(1, L)` | Per-residue ΔG |
| `pred_dg_avg` | `[float]` | Mean ΔG across residues (kcal/mol) |
| `sequence` | `str` | Parsed amino acid sequence |

When `ddg_scanning=True`, returns `(ddg_scan, scaled_ddg_scan, sequence)` where `ddg_scan` has shape `(L, 20, 1, L)`.

---

### `SaProtABS(weights_path)`
Loads SaProt 650M with LoRA weights and a stability head.

### `SaProtABS_predict(model, pdb_path, chain_id="A", ddg_scanning=False, cdna_rescale=False, foldseek_path="bin/foldseek", given_seq=None, handle_missing_residues="mask")`

| Return | Shape | Description |
|---|---|---|
| `pred_dg_per_res` | `(1, L)` | Per-residue ΔG |
| `pred_dg_avg` | `[float]` | Mean ΔG across residues (kcal/mol) |
| `combined_seq` | `str` | SA-token sequence (alternating AA + structure letter) |

Pass `given_seq="AaGgLl..."` to skip Foldseek and provide a precomputed SA-token sequence directly.

`handle_missing_residues` options: `"mask"` (default, <5% gaps), `"interpolate"` (5–15%), `"skip"` (>15%), `"keep"` (raw).

## Example Scripts

| Script | Description |
|---|---|
| `scripts/run_paper_examples.py` | Run ESM3ΔG + SaProtΔG on all example structures |
| `scripts/SaProtABS_dms.py` | Deep mutational scanning pipeline |
| `scripts/SaProtABS_nanobody.py` | Nanobody stability scoring |
| `scripts/ESM3ABS_nanobody.py` | ESM3 nanobody stability scoring |
| `notebooks/SaProtABS.ipynb` | Interactive SaProt notebook |
| `notebooks/ESM3ABS.ipynb` | Interactive ESM3 notebook |

## Structure Folding

Input structures can be predicted with:

- **ESMFold** via the ESM Atlas API:
  ```bash
  curl -X POST --data "SEQUENCE" https://api.esmatlas.com/foldSequence/v1/pdb/
  ```
- **ColabFold / LocalColabFold**: [YoshitakaMo/localcolabfold](https://github.com/YoshitakaMo/localcolabfold)
  ```bash
  colabfold_batch input outputdir/
  ```

## Citation

If you use this code or models, please cite:

```bibtex
@article{,
  title   = {},
  author  = {},
  journal = {},
  year    = {2025},
}
```
