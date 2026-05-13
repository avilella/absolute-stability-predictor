"""
Fast evaluation of ESM3ΔG and SaProtΔG ensemble on the DMSv4-AF test split.

Approach:
  1. Pre-compute all SaProt combined sequences (foldseek, sequential).
  2. Pre-compute all ESM3 encoded tokens (VQ-VAE tokenizer, sequential).
  3. Run batched forward passes for each model in the ensemble.

Metrics: RMSE and Spearman correlation between predicted and experimental ΔG.
Test ΔG values are clipped to [-1, 5] kcal/mol before evaluation.
Sigmoid-scaled model outputs are used (matching the clipped ΔG range).

Data:
  CSV  : /home/jupyter-yehlin/ThermoMPNN/data_all/dmsv4_train_AF_splits.csv
  Splits: /home/jupyter-yehlin/ThermoMPNN/dataset_splits/dmsv4_AF_full_splits.pkl
  PDBs : /data/mgnify/structures/
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils.config import get_default_config
from utils.config_esm3 import get_default_config as get_esm3_config
from SaProtABS import SaProtABS
from ESM3ABS import ESM3ABS, get_esm3_input_info_direct
from utils.foldseek_util import get_struc_seq

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "/home/jupyter-yehlin/ThermoMPNN/data_all/dmsv4_train_AF_splits.csv"
SPLITS_PATH = "/home/jupyter-yehlin/ThermoMPNN/dataset_splits/dmsv4_AF_full_splits.pkl"
PDB_DIR     = "/data/mgnify/structures"
CHAIN_ID    = "A"
FOLDSEEK    = "bin/foldseek"
DG_MIN, DG_MAX = -1.0, 5.0
BATCH_SIZE  = 16
OUT_PATH    = os.path.join(_ROOT, "test_set_evaluation_results.json")
SAP_WEIGHTS     = [f"saprotdg_weights/SaProtdG_weights_{i}_lora.ckpt"          for i in [1, 2, 3]]
SAP_AUG_WEIGHTS = [f"saprotdg_weights/SaProtdG_weights_augmented_{i}_lora.ckpt" for i in [1, 2, 3]]
ESM_WEIGHTS     = [f"esm3dg_weights/ESM3dG_weights_{i}_lora.ckpt"             for i in [1, 2, 3]]
ESM_AUG_WEIGHTS = [f"esm3dg_weights/ESM3dG_weights_augmented_{i}_lora.ckpt"   for i in [1, 2, 3]]

# ── Dataset wrappers ──────────────────────────────────────────────────────────

class SaProtTestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (name, dg_true, combined_seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class ESM3TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (name, dg_true, info_dict)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# ── Pre-computation ───────────────────────────────────────────────────────────

def precompute_saprot(test_df, pdb_dir, chain_id="A"):
    """Run foldseek on every test PDB to get SaProt combined sequences."""
    samples = []
    n = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        name     = row["name"]
        dg_true  = row["deltaG_clipped"]
        pdb_path = os.path.join(pdb_dir, name)

        if not os.path.exists(pdb_path):
            continue

        try:
            pid = os.getpid()
            struct_seq_data = get_struc_seq(FOLDSEEK, pdb_path, [chain_id], pid)
            if chain_id not in struct_seq_data:
                continue
            _, _, combined_seq = struct_seq_data[chain_id]
            samples.append((name, float(dg_true), combined_seq))
        except Exception:
            pass

        if (idx + 1) % 500 == 0:
            print(f"  foldseek [{idx+1}/{n}]  valid so far: {len(samples)}")

    return samples


def precompute_esm3(test_df, pdb_dir, chain_id, esm3_base_model):
    """Tokenize every test structure via ESM3's VQ-VAE encoder."""
    samples = []
    n = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows()):
        name     = row["name"]
        dg_true  = row["deltaG_clipped"]
        pdb_path = os.path.join(pdb_dir, name)

        if not os.path.exists(pdb_path):
            continue

        try:
            aa_seq = row["aa_seq"] if "aa_seq" in row else None
            info_dict, _ = get_esm3_input_info_direct(pdb_path, chain_id, esm3_base_model, aa_seq=aa_seq)
            if info_dict is None:
                continue
            # Store on CPU to save GPU memory; tied_featurize moves to device at inference time.
            info_dict = {k: v.cpu() for k, v in info_dict.items()}
            samples.append((name, float(dg_true), info_dict))
        except Exception:
            pass

        if (idx + 1) % 500 == 0:
            print(f"  esm3 encode [{idx+1}/{n}]  valid so far: {len(samples)}")

    return samples


# ── Batched inference ─────────────────────────────────────────────────────────

def saprot_batch_eval(model, dataset, batch_size, device):
    """Return (names, trues, preds) for the full dataset using batched inference."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x)
    names, trues, preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_names, batch_trues, batch_seqs = zip(*batch)
            try:
                _, pred_scaled_dg, mask = model(list(batch_seqs))
                pred_scaled_dg = pred_scaled_dg.cpu()
                mask = mask.cpu()
                pred_scaled_dg *= mask
                avg = (pred_scaled_dg.sum(-1) / mask.sum(-1)).tolist()
                names.extend(batch_names)
                trues.extend([float(t) for t in batch_trues])
                preds.extend(avg)
            except Exception as e:
                print(f"  [batch skip] {e}")
    return names, trues, preds


def esm3_batch_eval(model, dataset, batch_size, device):
    """Return (names, trues, preds) for the full dataset using batched inference."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x)
    names, trues, preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_names, batch_trues, batch_dicts = zip(*batch)
            try:
                _, pred_scaled_dg, mask = model(list(batch_dicts))
                pred_scaled_dg = pred_scaled_dg.cpu()
                mask = mask.cpu()
                pred_scaled_dg *= mask
                avg = (pred_scaled_dg.sum(-1) / mask.sum(-1)).tolist()
                names.extend(batch_names)
                trues.extend([float(t) for t in batch_trues])
                preds.extend(avg)
            except Exception as e:
                print(f"  [batch skip] {e}")
    return names, trues, preds


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def spearman(y_true, y_pred):
    return float(spearmanr(y_true, y_pred).statistic)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Using device: {device} ===")

    # ── Load test split ───────────────────────────────────────────────────────
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    test_names = set(splits["test"])

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["name"].isin(test_names)].copy()
    test_df["deltaG_clipped"] = test_df["deltaG"].clip(DG_MIN, DG_MAX)
    print(f"Test set: {len(test_df)} samples  "
          f"(ΔG clipped to [{DG_MIN}, {DG_MAX}]  "
          f"original: {test_df['deltaG'].min():.2f} – {test_df['deltaG'].max():.2f})")

    # ── Phase 1: SaProtΔG ─────────────────────────────────────────────────────
    print("\n=== [SaProt] Pre-computing foldseek sequences ===")
    sap_samples = precompute_saprot(test_df, PDB_DIR, CHAIN_ID)
    print(f"  Valid: {len(sap_samples)} / {len(test_df)}")
    sap_dataset = SaProtTestDataset(sap_samples)

    sap_cfg = get_default_config()
    sap_cfg.testing.ddg_scanning = False

    sap_pred_matrix = {}   # name -> [pred_model1, pred_model2, pred_model3]
    sap_true_map    = {}   # name -> dg_true

    for i, w in enumerate(SAP_WEIGHTS, 1):
        print(f"\n=== [SaProt] Model {i}: {w} ===")
        model = SaProtABS(w, sap_cfg)
        model.to(device)
        model.device = torch.device(device)

        names, trues, preds = saprot_batch_eval(model, sap_dataset, BATCH_SIZE, device)
        for nm, tr, pr in zip(names, trues, preds):
            sap_true_map[nm] = tr
            sap_pred_matrix.setdefault(nm, []).append(pr)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Phase 1b: SaProtΔG augmented ─────────────────────────────────────────
    sap_aug_pred_matrix = {}
    sap_aug_true_map    = {}

    for i, w in enumerate(SAP_AUG_WEIGHTS, 1):
        print(f"\n=== [SaProt-Aug] Model {i}: {w} ===")
        model = SaProtABS(w, sap_cfg)
        model.to(device)
        model.device = torch.device(device)

        names, trues, preds = saprot_batch_eval(model, sap_dataset, BATCH_SIZE, device)
        for nm, tr, pr in zip(names, trues, preds):
            sap_aug_true_map[nm] = tr
            sap_aug_pred_matrix.setdefault(nm, []).append(pr)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Phase 2: ESM3ΔG ───────────────────────────────────────────────────────
    print("\n=== [ESM3] Loading first model (base encoder used for pre-encoding) ===")
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False

    first_esm = ESM3ABS(ESM_WEIGHTS[0], esm_cfg)
    first_esm.to(device)
    first_esm.device = torch.device(device)
    base_esm3 = first_esm.esm3_stability_model.base_model

    print("\n=== [ESM3] Pre-computing structure encodings ===")
    esm_samples = precompute_esm3(test_df, PDB_DIR, CHAIN_ID, base_esm3)
    print(f"  Valid: {len(esm_samples)} / {len(test_df)}")
    esm_dataset = ESM3TestDataset(esm_samples)

    esm_pred_matrix = {}
    esm_true_map    = {}

    print(f"\n=== [ESM3] Model 1: {ESM_WEIGHTS[0]} ===")
    names, trues, preds = esm3_batch_eval(first_esm, esm_dataset, BATCH_SIZE, device)
    for nm, tr, pr in zip(names, trues, preds):
        esm_true_map[nm] = tr
        esm_pred_matrix.setdefault(nm, []).append(pr)
    del first_esm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i, w in enumerate(ESM_WEIGHTS[1:], 2):
        print(f"\n=== [ESM3] Model {i}: {w} ===")
        model = ESM3ABS(w, esm_cfg)
        model.to(device)
        model.device = torch.device(device)

        names, trues, preds = esm3_batch_eval(model, esm_dataset, BATCH_SIZE, device)
        for nm, tr, pr in zip(names, trues, preds):
            esm_true_map[nm] = tr
            esm_pred_matrix.setdefault(nm, []).append(pr)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Phase 3: ESM3ΔG augmented ─────────────────────────────────────────────
    esm_aug_pred_matrix = {}
    esm_aug_true_map    = {}

    for i, w in enumerate(ESM_AUG_WEIGHTS, 1):
        print(f"\n=== [ESM3-Aug] Model {i}: {w} ===")
        model = ESM3ABS(w, esm_cfg)
        model.to(device)
        model.device = torch.device(device)

        names, trues, preds = esm3_batch_eval(model, esm_dataset, BATCH_SIZE, device)
        for nm, tr, pr in zip(names, trues, preds):
            esm_aug_true_map[nm] = tr
            esm_aug_pred_matrix.setdefault(nm, []).append(pr)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute metrics ───────────────────────────────────────────────────────
    def ensemble_metrics(pred_matrix, true_map):
        valid = {n: ps for n, ps in pred_matrix.items() if ps}
        y_true = [true_map[n]                    for n in valid]
        y_pred = [float(np.mean(valid[n]))        for n in valid]
        return {"n": len(valid), "rmse": rmse(y_true, y_pred), "spearman": spearman(y_true, y_pred)}

    metrics = {}
    if sap_pred_matrix:
        metrics["saprot"]     = ensemble_metrics(sap_pred_matrix,     sap_true_map)
    if sap_aug_pred_matrix:
        metrics["saprot_aug"] = ensemble_metrics(sap_aug_pred_matrix, sap_aug_true_map)
    if esm_pred_matrix:
        metrics["esm3"]       = ensemble_metrics(esm_pred_matrix,     esm_true_map)
    if esm_aug_pred_matrix:
        metrics["esm3_aug"]   = ensemble_metrics(esm_aug_pred_matrix, esm_aug_true_map)

    print("\n\n=== Results ===")
    for model_name, m in metrics.items():
        print(f"{model_name:12s}  n={m['n']}  RMSE={m['rmse']:.4f}  Spearman={m['spearman']:.4f}")

    # ── Build per-sample records ───────────────────────────────────────────────
    all_names = sorted(set(
        list(sap_pred_matrix) + list(sap_aug_pred_matrix) +
        list(esm_pred_matrix) + list(esm_aug_pred_matrix)
    ))
    records = []
    for nm in all_names:
        sap_ps     = sap_pred_matrix.get(nm, [])
        sap_aug_ps = sap_aug_pred_matrix.get(nm, [])
        esm_ps     = esm_pred_matrix.get(nm, [])
        esm_aug_ps = esm_aug_pred_matrix.get(nm, [])
        dg_true = sap_true_map.get(nm, sap_aug_true_map.get(nm, esm_true_map.get(nm, esm_aug_true_map.get(nm))))
        records.append({
            "name":          nm,
            "dg_true":       dg_true,
            "sap_preds":     sap_ps,
            "sap_mean":      float(np.mean(sap_ps))     if sap_ps     else None,
            "sap_aug_preds": sap_aug_ps,
            "sap_aug_mean":  float(np.mean(sap_aug_ps)) if sap_aug_ps else None,
            "esm_preds":     esm_ps,
            "esm_mean":      float(np.mean(esm_ps))     if esm_ps     else None,
            "esm_aug_preds": esm_aug_ps,
            "esm_aug_mean":  float(np.mean(esm_aug_ps)) if esm_aug_ps else None,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {"metrics": metrics, "records": records}
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
