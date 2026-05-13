"""
Verify that with zero-pad fix, batch=1 and batch=16 give identical predictions.
Uses model 1 only on the first N_SAMPLES test samples for speed.
"""

import os, sys, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from ESM3ABS import ESM3ABS, get_esm3_input_info_direct
from utils.config_esm3 import get_default_config as get_esm3_config

CSV_PATH    = "/home/jupyter-yehlin/ThermoMPNN/data_all/dmsv4_train_AF_splits.csv"
SPLITS_PATH = "/home/jupyter-yehlin/ThermoMPNN/dataset_splits/dmsv4_AF_full_splits.pkl"
PDB_DIR     = "/data/mgnify/structures"
CHAIN_ID    = "A"
DG_MIN, DG_MAX = -1.0, 5.0
N_SAMPLES   = 200   # subset for speed
MODEL_PATH  = "esm3dg_weights/ESM3dG_weights_1_lora.ckpt"


class ESM3TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


def run_inference(model, samples, batch_size, device):
    dataset = ESM3TestDataset(samples)
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x)
    name2pred = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_names, _, batch_dicts = zip(*batch)
            try:
                _, pred_scaled_dg, mask = model(list(batch_dicts))
                pred_scaled_dg = pred_scaled_dg.cpu()
                mask = mask.cpu()
                pred_scaled_dg *= mask
                avg = (pred_scaled_dg.sum(-1) / mask.sum(-1)).tolist()
                for nm, p in zip(batch_names, avg):
                    name2pred[nm] = p
            except Exception as e:
                print(f"  [skip] {e}")
    return name2pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load test split
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
    test_names = list(splits["test"])[:N_SAMPLES]

    df = pd.read_csv(CSV_PATH)
    df["deltaG_clipped"] = df["deltaG"].clip(DG_MIN, DG_MAX)
    test_df = df[df["name"].isin(test_names)].reset_index(drop=True)
    print(f"Using {len(test_df)} samples")

    # Load model + pre-compute encodings
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False
    model = ESM3ABS(MODEL_PATH, esm_cfg)
    model.to(device)
    model.device = torch.device(device)
    base_esm3 = model.esm3_stability_model.base_model

    print("\nPre-computing ESM3 encodings...")
    samples = []
    for _, row in test_df.iterrows():
        pdb_path = os.path.join(PDB_DIR, row["name"])
        if not os.path.exists(pdb_path):
            continue
        try:
            aa_seq = row["aa_seq"] if "aa_seq" in row else None
            info_dict, _ = get_esm3_input_info_direct(pdb_path, CHAIN_ID, base_esm3, aa_seq=aa_seq)
            if info_dict is None:
                continue
            info_dict = {k: v.cpu() for k, v in info_dict.items()}
            samples.append((row["name"], float(row["deltaG_clipped"]), info_dict))
        except Exception:
            pass
    print(f"Encoded {len(samples)} samples")

    # Run with batch=1
    print("\nRunning inference  batch=1  ...")
    preds_b1 = run_inference(model, samples, batch_size=1,  device=device)

    # Run with batch=16
    print("Running inference  batch=16 ...")
    preds_b16 = run_inference(model, samples, batch_size=16, device=device)

    # Compare
    common = sorted(set(preds_b1) & set(preds_b16))
    diffs  = [abs(preds_b1[n] - preds_b16[n]) for n in common]
    n_mismatch = sum(1 for d in diffs if d > 1e-4)

    print(f"\n{'='*55}")
    print(f"Samples compared : {len(common)}")
    print(f"Max diff         : {max(diffs):.2e}")
    print(f"Mean diff        : {np.mean(diffs):.2e}")
    print(f"Samples > 1e-4   : {n_mismatch}")
    if n_mismatch == 0:
        print("RESULT: batch=1 and batch=16 are IDENTICAL (zero-pad fix works)")
    else:
        print("RESULT: still differ — fix did not fully resolve batch effect")
        worst = sorted(zip(diffs, common), reverse=True)[:5]
        print("Top mismatches:")
        for d, n in worst:
            print(f"  {n}  diff={d:.4e}  b1={preds_b1[n]:.6f}  b16={preds_b16[n]:.6f}")


if __name__ == "__main__":
    main()
