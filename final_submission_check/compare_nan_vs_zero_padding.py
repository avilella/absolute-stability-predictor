"""
Compare ESM3 ensemble metrics before and after fixing NaN coordinate padding.

OLD: tied_featurize uses np.nan for padded coordinate positions  (batch-size dependent)
NEW: tied_featurize uses 0.0   for padded coordinate positions  (batch-size independent)

SaProt is unaffected (already matches), so only ESM3 is re-evaluated here.
Requires ESM3ABS.py to already have the zero-pad fix applied.
Saves updated results to test_set_evaluation_results_zero_pad.json.
"""

import os, sys, json, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from ESM3ABS import ESM3ABS, get_esm3_input_info_direct
from utils.config_esm3 import get_default_config as get_esm3_config

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH    = "/home/jupyter-yehlin/ThermoMPNN/data_all/dmsv4_train_AF_splits.csv"
SPLITS_PATH = "/home/jupyter-yehlin/ThermoMPNN/dataset_splits/dmsv4_AF_full_splits.pkl"
PDB_DIR     = "/data/mgnify/structures"
CHAIN_ID    = "A"
BATCH_SIZE  = 16
DG_MIN, DG_MAX = -1.0, 5.0

OLD_JSON = os.path.join(_ROOT, "test_set_evaluation_results.json")
NEW_JSON = os.path.join(_ROOT, "test_set_evaluation_results_zero_pad.json")

ESM_WEIGHTS     = [f"esm3dg_weights/ESM3dG_weights_{i}_lora.ckpt"             for i in [1, 2, 3]]
ESM_AUG_WEIGHTS = [f"esm3dg_weights/ESM3dG_weights_augmented_{i}_lora.ckpt"   for i in [1, 2, 3]]


# ── Dataset ────────────────────────────────────────────────────────────────────
class ESM3TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


# ── Helpers ────────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def spearman_r(y_true, y_pred):
    return float(spearmanr(y_true, y_pred).statistic)

def esm3_batch_eval(model, dataset, batch_size, device):
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

def precompute_esm3(test_df, pdb_dir, chain_id, base_esm3_model):
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
            info_dict, _ = get_esm3_input_info_direct(pdb_path, chain_id, base_esm3_model, aa_seq=aa_seq)
            if info_dict is None:
                continue
            info_dict = {k: v.cpu() for k, v in info_dict.items()}
            samples.append((name, float(dg_true), info_dict))
        except Exception:
            pass
        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{n}]  valid: {len(samples)}")
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load old results
    print(f"\nLoading old results (NaN padding) from:\n  {OLD_JSON}")
    with open(OLD_JSON) as f:
        old_data = json.load(f)
    old_metrics = old_data["metrics"]
    old_records = {r["name"]: r for r in old_data["records"]}

    # Load test split
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    test_names = set(splits["test"])
    df = pd.read_csv(CSV_PATH)
    df["deltaG_clipped"] = df["deltaG"].clip(DG_MIN, DG_MAX)
    test_df = df[df["name"].isin(test_names)].reset_index(drop=True)
    print(f"Test samples: {len(test_df)}")

    # Pre-compute ESM3 inputs
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False

    print("\n=== Loading first ESM3 model (base encoder for pre-encoding) ===")
    first_esm = ESM3ABS(ESM_WEIGHTS[0], esm_cfg)
    first_esm.to(device)
    first_esm.device = torch.device(device)
    base_esm3 = first_esm.esm3_stability_model.base_model

    print("\n=== Pre-computing structure encodings (zero-padded) ===")
    esm_samples = precompute_esm3(test_df, PDB_DIR, CHAIN_ID, base_esm3)
    print(f"  Valid: {len(esm_samples)} / {len(test_df)}")
    esm_dataset = ESM3TestDataset(esm_samples)

    # ESM3 non-aug
    esm_new_preds = {}
    print(f"\n=== [ESM3] Model 1: {ESM_WEIGHTS[0]} ===")
    names, trues, preds = esm3_batch_eval(first_esm, esm_dataset, BATCH_SIZE, device)
    for nm, pr in zip(names, preds):
        esm_new_preds.setdefault(nm, []).append(pr)
    del first_esm
    torch.cuda.empty_cache()

    for i, w in enumerate(ESM_WEIGHTS[1:], 2):
        print(f"\n=== [ESM3] Model {i}: {w} ===")
        model = ESM3ABS(w, esm_cfg)
        model.to(device)
        model.device = torch.device(device)
        names, trues, preds = esm3_batch_eval(model, esm_dataset, BATCH_SIZE, device)
        for nm, pr in zip(names, preds):
            esm_new_preds.setdefault(nm, []).append(pr)
        del model
        torch.cuda.empty_cache()

    # ESM3 aug
    esm_aug_new_preds = {}
    for i, w in enumerate(ESM_AUG_WEIGHTS, 1):
        print(f"\n=== [ESM3-Aug] Model {i}: {w} ===")
        model = ESM3ABS(w, esm_cfg)
        model.to(device)
        model.device = torch.device(device)
        names, trues, preds = esm3_batch_eval(model, esm_dataset, BATCH_SIZE, device)
        for nm, pr in zip(names, preds):
            esm_aug_new_preds.setdefault(nm, []).append(pr)
        del model
        torch.cuda.empty_cache()

    # Compute new metrics
    common = sorted(set(esm_new_preds) & set(old_records))
    true_vals       = [old_records[n]["dg_true"]                    for n in common]
    esm_new_avg     = [float(np.mean(esm_new_preds[n]))             for n in common]
    esm_aug_new_avg = [float(np.mean(esm_aug_new_preds[n]))         for n in common]
    esm_old_avg     = [float(np.mean(old_records[n]["esm_preds"]))  for n in common]
    esm_aug_old_avg = [float(np.mean(old_records[n]["esm_aug_preds"])) for n in common]

    new_esm_rmse,     new_esm_sp     = rmse(true_vals, esm_new_avg),     spearman_r(true_vals, esm_new_avg)
    new_esm_aug_rmse, new_esm_aug_sp = rmse(true_vals, esm_aug_new_avg), spearman_r(true_vals, esm_aug_new_avg)

    diffs_esm     = [abs(a - b) for a, b in zip(esm_new_avg,     esm_old_avg)]
    diffs_esm_aug = [abs(a - b) for a, b in zip(esm_aug_new_avg, esm_aug_old_avg)]

    # Print comparison
    print("\n" + "=" * 68)
    print(f"{'Model':<14} {'Metric':<10} {'NaN pad (old)':>14} {'Zero pad (new)':>14} {'Delta':>8}")
    print("-" * 68)
    for model, metric, old_v, new_v in [
        ("ESM3",     "RMSE",     old_metrics["esm3"]["rmse"],         new_esm_rmse),
        ("ESM3",     "Spearman", old_metrics["esm3"]["spearman"],     new_esm_sp),
        ("ESM3-Aug", "RMSE",     old_metrics["esm3_aug"]["rmse"],     new_esm_aug_rmse),
        ("ESM3-Aug", "Spearman", old_metrics["esm3_aug"]["spearman"], new_esm_aug_sp),
    ]:
        delta = new_v - old_v
        print(f"{model:<14} {metric:<10} {old_v:>14.4f} {new_v:>14.4f} {delta:>+8.4f}")

    print(f"\nPer-sample ensemble-mean change (NaN → zero pad):")
    print(f"  ESM3     n={len(diffs_esm):4d}  mean_diff={np.mean(diffs_esm):.4f}  max_diff={np.max(diffs_esm):.4f}")
    print(f"  ESM3-Aug n={len(diffs_esm_aug):4d}  mean_diff={np.mean(diffs_esm_aug):.4f}  max_diff={np.max(diffs_esm_aug):.4f}")

    # Save updated JSON (ESM3 fields replaced, SaProt fields kept from old)
    new_records = []
    for r in old_data["records"]:
        n = r["name"]
        new_r = dict(r)
        if n in esm_new_preds:
            new_r["esm_preds"] = esm_new_preds[n]
            new_r["esm_mean"]  = float(np.mean(esm_new_preds[n]))
        if n in esm_aug_new_preds:
            new_r["esm_aug_preds"] = esm_aug_new_preds[n]
            new_r["esm_aug_mean"]  = float(np.mean(esm_aug_new_preds[n]))
        new_records.append(new_r)

    new_metrics = dict(old_metrics)
    new_metrics["esm3"]     = {"rmse": new_esm_rmse,     "spearman": new_esm_sp}
    new_metrics["esm3_aug"] = {"rmse": new_esm_aug_rmse, "spearman": new_esm_aug_sp}

    with open(NEW_JSON, "w") as f:
        json.dump({"metrics": new_metrics, "records": new_records}, f)
    print(f"\nSaved: {NEW_JSON}")


if __name__ == "__main__":
    main()
