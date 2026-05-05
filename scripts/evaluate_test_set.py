"""
Evaluate ESM3ΔG and SaProtΔG ensemble on the DMSv4-AF test split.

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
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from utils.config import get_default_config
from utils.config_esm3 import get_default_config as get_esm3_config
from SaProtABS import SaProtABS, SaProtABS_predict
from ESM3ABS import ESM3ABS, ESM3ABS_predict

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "/home/jupyter-yehlin/ThermoMPNN/data_all/dmsv4_train_AF_splits.csv"
SPLITS_PATH = "/home/jupyter-yehlin/ThermoMPNN/dataset_splits/dmsv4_AF_full_splits.pkl"
PDB_DIR     = "/data/mgnify/structures"
CHAIN_ID    = "A"
DG_MIN, DG_MAX = -1.0, 5.0
OUT_PATH    = os.path.join(_ROOT, "test_set_evaluation_results.json")

SAP_WEIGHTS = [f"saprotdg_weights/SaProtdG_weights_{i}_lora.ckpt" for i in [1, 2, 3]]
ESM_WEIGHTS = [f"esm3dg_weights/ESM3dG_weights_{i}_lora.ckpt"    for i in [1, 2, 3]]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def spearman(y_true, y_pred):
    return float(spearmanr(y_true, y_pred).statistic)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Using device: {device} ===")

    # ── Load test split ───────────────────────────────────────────────────────
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    test_names = set(splits["test"])

    df = pd.read_csv(CSV_PATH)
    test_df = df[df["name"].isin(test_names)].copy()
    test_df["deltaG_clipped"] = test_df["deltaG"].clip(DG_MIN, DG_MAX)

    print(f"Test set: {len(test_df)} samples")
    print(f"ΔG clipped to [{DG_MIN}, {DG_MAX}]  "
          f"(original range: {test_df['deltaG'].min():.2f} – {test_df['deltaG'].max():.2f})")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n=== Loading SaProtΔG models ===")
    sap_cfg = get_default_config()
    sap_cfg.testing.ddg_scanning = False
    sap_models = []
    for w in SAP_WEIGHTS:
        print(f"  {w}")
        m = SaProtABS(w, sap_cfg)
        m.to(device); m.device = torch.device(device)
        sap_models.append(m)

    print("\n=== Loading ESM3ΔG models ===")
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False
    esm_models = []
    for w in ESM_WEIGHTS:
        print(f"  {w}")
        m = ESM3ABS(w, esm_cfg)
        m.to(device); m.device = torch.device(device)
        esm_models.append(m)

    # ── Run predictions ───────────────────────────────────────────────────────
    records = []
    n = len(test_df)

    for idx, (_, row) in enumerate(test_df.iterrows()):
        name      = row["name"]
        dg_true   = row["deltaG_clipped"]
        pdb_path  = os.path.join(PDB_DIR, name)

        if not os.path.exists(pdb_path):
            print(f"[{idx+1}/{n}] MISSING {name}")
            continue

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"\n[{idx+1}/{n}] {name}  true ΔG={dg_true:.3f}")

        # SaProtΔG ensemble (sigmoid output)
        sap_preds = []
        for m in sap_models:
            try:
                _, avg, _ = SaProtABS_predict(m, pdb_path, CHAIN_ID, cdna_rescale=True)
                sap_preds.append(avg[0])
            except Exception:
                pass

        # ESM3ΔG ensemble (sigmoid output)
        esm_preds = []
        for m in esm_models:
            try:
                _, avg, _ = ESM3ABS_predict(m, pdb_path, CHAIN_ID, sigmoid_on=True)
                esm_preds.append(avg[0])
            except Exception:
                pass

        records.append({
            "name":       name,
            "dg_true":    float(dg_true),
            "sap_preds":  sap_preds,
            "sap_mean":   float(np.mean(sap_preds)) if sap_preds else None,
            "esm_preds":  esm_preds,
            "esm_mean":   float(np.mean(esm_preds)) if esm_preds else None,
        })

    # ── Compute metrics ───────────────────────────────────────────────────────
    sap_rows = [r for r in records if r["sap_mean"] is not None]
    esm_rows = [r for r in records if r["esm_mean"] is not None]

    sap_true  = [r["dg_true"]  for r in sap_rows]
    sap_pred  = [r["sap_mean"] for r in sap_rows]
    esm_true  = [r["dg_true"]  for r in esm_rows]
    esm_pred  = [r["esm_mean"] for r in esm_rows]

    metrics = {}
    if sap_rows:
        metrics["saprot"] = {
            "n":        len(sap_rows),
            "rmse":     rmse(sap_true, sap_pred),
            "spearman": spearman(sap_true, sap_pred),
        }
    if esm_rows:
        metrics["esm3"] = {
            "n":        len(esm_rows),
            "rmse":     rmse(esm_true, esm_pred),
            "spearman": spearman(esm_true, esm_pred),
        }

    print("\n\n=== Results ===")
    for model, m in metrics.items():
        print(f"{model:10s}  n={m['n']}  RMSE={m['rmse']:.4f}  Spearman={m['spearman']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {"metrics": metrics, "records": records}
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
