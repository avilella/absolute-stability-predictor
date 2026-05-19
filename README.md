# ESM3ΔG & SaProtΔG — Protein Stability Prediction

<p align="center">
  <img src="logo.png" alt="ESM3dG logo" width="480"/>
</p>

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
conda create -n stability python=3.12 -y
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

> **ESM3 is a gated model.** Before loading it you must:
> 1. Accept the license at [huggingface.co/EvolutionaryScale/esm3-sm-open-v1](https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1)
> 2. Log in with your HuggingFace token:
> ```bash
> huggingface-cli login
> ```

## Download Weights

```bash
bash download_weights.sh                    # public repo or already logged in
bash download_weights.sh --token hf_xxx     # supply token explicitly
```

Weights are also available directly on Hugging Face: [Yehlin/absolute-stability](https://huggingface.co/Yehlin/absolute-stability)

## Quick Start

### ESM3ΔG

```python
from ESM3dG import ESM3dG, ESM3dG_predict

# Augmented ensemble (recommended)
WEIGHTS = [
    "esm3dg_weights/ESM3dG_weights_augmented_1_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_augmented_2_lora.ckpt",
    "esm3dg_weights/ESM3dG_weights_augmented_3_lora.ckpt",
]

# Non-augmented ensemble
# WEIGHTS = [
#     "esm3dg_weights/ESM3dG_weights_1_lora.ckpt",
#     "esm3dg_weights/ESM3dG_weights_2_lora.ckpt",
#     "esm3dg_weights/ESM3dG_weights_3_lora.ckpt",
# ]

models = [ESM3dG(w) for w in WEIGHTS]
preds = [ESM3dG_predict(m, "examples/nanobody_1zvh.cif", "A")[1][0] for m in models]
ensemble_avg = sum(preds) / len(preds)
print(f"Ensemble ΔG: {ensemble_avg:.2f} kcal/mol")
```

### SaProtΔG

```python
from SaProtdG import SaProtdG, SaProtdG_predict

# Augmented ensemble (recommended)
WEIGHTS = [
    "saprotdg_weights/SaProtdG_weights_augmented_1_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_2_lora.ckpt",
    "saprotdg_weights/SaProtdG_weights_augmented_3_lora.ckpt",
]

# Non-augmented ensemble
# WEIGHTS = [
#     "saprotdg_weights/SaProtdG_weights_1_lora.ckpt",
#     "saprotdg_weights/SaProtdG_weights_2_lora.ckpt",
#     "saprotdg_weights/SaProtdG_weights_3_lora.ckpt",
# ]

models = [SaProtdG(w) for w in WEIGHTS]
preds = [SaProtdG_predict(m, "examples/nanobody_1zvh.cif", "A")[1][0] for m in models]
ensemble_avg = sum(preds) / len(preds)
print(f"Ensemble ΔG: {ensemble_avg:.2f} kcal/mol")
```

## Mutational Scanning (ΔΔG)

```python
ddg_scan, scaled_ddg_scan, sequence = ESM3dG_predict(
    model,
    pdb_path="examples/nanobody_1zvh.cif",
    chain_id="A",
    ddg_scanning=True,
)
# ddg_scan shape: (L, 20, 1, L) — all single-point mutations
```

## Example Scripts

| Script | Description |
|---|---|
| `notebooks/SaProtdG.ipynb` | Interactive SaProt notebook |
| `notebooks/ESM3dG.ipynb` | Interactive ESM3 notebook |

## Structure Folding

Input structures can be predicted with:

- **ESMFold** via the ESM Atlas API:
  ```bash
  curl -X POST --data "SEQUENCE" https://api.esmatlas.com/foldSequence/v1/pdb/ -o data/output.pdb
  ```
  The `-o output.pdb` flag saves the structure to `output.pdb`. Without it, the PDB content is printed to the terminal only.
- **ColabFold / LocalColabFold**: follow the installation instructions at [YoshitakaMo/localcolabfold](https://github.com/YoshitakaMo/localcolabfold).

  Run:
  ```bash
  colabfold_batch myprotein.fasta data/
  ```

  Create a FASTA file:
  ```bash
  echo ">my_protein
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL" > myprotein.fasta
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
