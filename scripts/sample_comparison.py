"""
Sample diverse proteins and compare SaProtdG vs ESM3dG predictions.
Sources:
  - thermomut full_seq_pdb  (36 proteins, closer to training distribution)
  - PDB_S4038               (random sample, more diverse)
"""
import os, sys, random, json
import torch
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from utils.config import get_default_config
from utils.config_esm3 import get_default_config as get_esm3_config
from SaProtABS import SaProtABS, SaProtABS_predict
from ESM3ABS import ESM3ABS, ESM3ABS_predict
from utils.foldseek_util import get_struc_seq

THERMOMUT_DIR = "/home/jupyter-yehlin/ThermoMPNN/data_all/thermomut_new/v3/full_seq_pdb"
PDB_S4038_DIR = "/home/jupyter-yehlin/ThermoMPNN/data_all/PDB_S4038"
N_S4038       = 30   # random sample size from PDB_S4038
RANDOM_SEED   = 123

def valid_chain_a(pdb_path, min_len=50, max_len=500):
    """Return True if chain A exists with length in [min_len, max_len]."""
    try:
        data = get_struc_seq("bin/foldseek", pdb_path, ["A"], os.getpid())
        if "A" not in data:
            return False
        _, _, combined = data["A"]
        n = len(combined) // 2
        return min_len <= n <= max_len
    except Exception:
        return False

def predict_both(pdb_path, sap_models, esm_models):
    sap_preds, esm_preds = [], []
    for m in sap_models:
        try:
            _, avg, _ = SaProtABS_predict(m, pdb_path, "A")
            sap_preds.append(avg[0])
        except Exception as e:
            pass
    for m in esm_models:
        try:
            _, avg, _ = ESM3ABS_predict(m, pdb_path, "A")
            esm_preds.append(avg[0])
        except Exception as e:
            pass
    sap = float(np.mean(sap_preds)) if sap_preds else None
    esm = float(np.mean(esm_preds)) if esm_preds else None
    return sap, esm


def main():
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading SaProtdG models...")
    sap_cfg = get_default_config()
    sap_cfg.testing.ddg_scanning = False
    sap_models = []
    for i in [1, 2, 3]:
        m = SaProtABS(f"saprotdg_weights/SaProtdG_weights_{i}_lora.ckpt", sap_cfg)
        m.to(device); m.device = torch.device(device)
        sap_models.append(m)

    print("Loading ESM3dG models...")
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False
    esm_models = []
    for i in [1, 2, 3]:
        m = ESM3ABS(f"esm3dg_weights/ESM3dG_weights_{i}_lora.ckpt", esm_cfg)
        m.to(device); m.device = torch.device(device)
        esm_models.append(m)

    # ── Build file list ───────────────────────────────────────────────
    entries = []   # (label, category, pdb_path)

    # thermomut: all
    for f in sorted(os.listdir(THERMOMUT_DIR)):
        if f.endswith(".pdb"):
            entries.append((f[:-4], "thermomut", os.path.join(THERMOMUT_DIR, f)))

    # PDB_S4038: random sample filtered by valid chain A length
    random.seed(RANDOM_SEED)
    all_s4038 = [f for f in os.listdir(PDB_S4038_DIR) if f.endswith(".pdb")]
    random.shuffle(all_s4038)
    count = 0
    for f in all_s4038:
        if count >= N_S4038:
            break
        path = os.path.join(PDB_S4038_DIR, f)
        if valid_chain_a(path):
            entries.append((f[:-4], "PDB_S4038", path))
            count += 1

    print(f"\nTotal proteins to evaluate: {len(entries)}")
    print(f"  thermomut : {sum(1 for e in entries if e[1]=='thermomut')}")
    print(f"  PDB_S4038 : {sum(1 for e in entries if e[1]=='PDB_S4038')}\n")

    # ── Run predictions ───────────────────────────────────────────────
    results = []
    for name, cat, path in entries:
        sap, esm = predict_both(path, sap_models, esm_models)
        if sap is None or esm is None:
            print(f"  SKIP {name} ({cat})")
            continue
        diff = sap - esm
        flag = "SAP>" if diff > 0 else "ESM>"
        print(f"  {flag}  {name:12s} [{cat:10s}]  SaProt={sap:.3f}  ESM3={esm:.3f}  diff={diff:+.3f}")
        results.append({"name": name, "category": cat, "saprot": sap, "esm3": esm, "diff": diff})

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    diffs = [r["diff"] for r in results]
    n_sap_higher = sum(1 for d in diffs if d > 0)
    print(f"SaProt > ESM3 in {n_sap_higher}/{len(results)} = {n_sap_higher/len(results)*100:.1f}% of proteins")
    print(f"Mean (SaProt - ESM3) = {np.mean(diffs):+.3f} kcal/mol")
    print(f"Std  (SaProt - ESM3) = {np.std(diffs):.3f} kcal/mol")

    for cat in ["thermomut", "PDB_S4038"]:
        sub = [r["diff"] for r in results if r["category"] == cat]
        if sub:
            n_hi = sum(1 for d in sub if d > 0)
            print(f"\n  [{cat}] n={len(sub)}  SaProt>{n_hi/len(sub)*100:.0f}%  mean diff={np.mean(sub):+.3f}")

    out = os.path.join(_ROOT, "sample_comparison_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
