"""
Run ESM3ΔG and SaProtΔG predictions on example structures from the examples/ directory.
Covers nanobodies, dark matter proteins, MGnify stability domains, and TED/AlphaFold domains.

Outputs paper_example_results.json with:
  - Ensemble mean/std ΔG per protein per model
  - Ensemble mean (L × 20) ΔΔG mutational scan per protein per model
    (ddg_scan shape from model: (L, 20, 1, L) → reduced to (L, 20) by mean over context dim)
"""
import os, sys, json
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

from utils.config import get_default_config
from utils.config_esm3 import get_default_config as get_esm3_config
from SaProtABS import SaProtABS, SaProtABS_predict
from ESM3ABS import ESM3ABS, ESM3ABS_predict

EXAMPLES_DIR = os.path.join(_ROOT, "examples")

EXAMPLES = {
    # Nanobodies (Boltz-2 predicted)
    "nanobody_1zvh":      ("nanobody_1zvh.cif",      "A", "Nanobody 1ZVH",              125),
    "nanobody_2A3":       ("nanobody_2A3.cif",        "A", "Nanobody 2A3",               123),
    "nanobody_7C12":      ("nanobody_7C12.cif",       "A", "Nanobody 7C12",              124),
    # Dark matter proteins (ESMFold)
    "darkmatter_n344":    ("darkmatter_n344.pdb",     "A", "Dark matter domain n344",     74),
    "darkmatter_n346":    ("darkmatter_n346.pdb",     "A", "Dark matter domain n346",     74),
    # MGnify stability dataset
    "stability_1998469":  ("stability_1998469.pdb",   "A", "MGnify domain 1998469",       60),
    "stability_1998520":  ("stability_1998520.pdb",   "A", "MGnify domain 1998520",       70),
    # MGnify dark proteome
    "mgnify_1A0N":        ("mgnify_1A0N.pdb",         "A", "Metagenomic domain 1A0N",     58),
    "mgnify_1A32":        ("mgnify_1A32.pdb",         "A", "Metagenomic domain 1A32",     63),
    # TED / AlphaFold domains
    "ted_AF-A0A011LYJ0":  ("ted_AF-A0A011LYJ0.cif",  "A", "AlphaFold domain A0A011LYJ0", 73),
    "ted_AF-A0A011MHR4":  ("ted_AF-A0A011MHR4.cif",  "A", "AlphaFold domain A0A011MHR4", 70),
}


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Using device: {device} ===")

    # ── Load models (ensemble of 3) ───────────────────────────────────
    print("\n=== Loading SaProtΔG models ===")
    sap_cfg = get_default_config()
    sap_cfg.testing.ddg_scanning = False
    sap_models = []
    for i in [1, 2, 3]:
        w = f"saprotdg_weights/SaProtdG_weights_{i}_lora.ckpt"
        print(f"  Loading {w}")
        model = SaProtABS(w, sap_cfg)
        model.to(device)
        model.device = torch.device(device)
        sap_models.append(model)

    print("\n=== Loading ESM3ΔG models ===")
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False
    esm_models = []
    for i in [1, 2, 3]:
        w = f"esm3dg_weights/ESM3dG_weights_{i}_lora.ckpt"
        print(f"  Loading {w}")
        model = ESM3ABS(w, esm_cfg)
        model.to(device)
        model.device = torch.device(device)
        esm_models.append(model)

    # ── Run predictions ───────────────────────────────────────────────
    results = {}

    for pdb_id, (filename, chain, desc, length) in EXAMPLES.items():
        print(f"\n{'='*50}")
        print(f"  {pdb_id} — {desc} (chain {chain})")
        pdb_path = os.path.join(EXAMPLES_DIR, filename)

        sap_preds = []
        for i, model in enumerate(sap_models, 1):
            try:
                _, avg, _ = SaProtABS_predict(model, pdb_path, chain)
                val = avg[0]
                sap_preds.append(round(val, 3))
                print(f"  SaProtΔG model {i}: {val:.3f} kcal/mol")
            except Exception as e:
                print(f"  SaProtΔG model {i} FAILED: {e}")

        esm_preds = []
        for i, model in enumerate(esm_models, 1):
            try:
                _, avg, _ = ESM3ABS_predict(model, pdb_path, [chain])
                val = avg[0]
                esm_preds.append(round(val, 3))
                print(f"  ESM3ΔG   model {i}: {val:.3f} kcal/mol")
            except Exception as e:
                print(f"  ESM3ΔG   model {i} FAILED: {e}")

        sap_avg = round(sum(sap_preds) / len(sap_preds), 3) if sap_preds else None
        sap_std = round((sum((x - sap_avg)**2 for x in sap_preds) / len(sap_preds))**0.5, 3) if len(sap_preds) > 1 else 0

        esm_avg = round(sum(esm_preds) / len(esm_preds), 3) if esm_preds else None
        esm_std = round((sum((x - esm_avg)**2 for x in esm_preds) / len(esm_preds))**0.5, 3) if len(esm_preds) > 1 else 0

        results[pdb_id] = {
            "description": desc,
            "chain":       chain,
            "length":      length,
            "saprot": {"individual": sap_preds, "mean": sap_avg, "std": sap_std},
            "esm3":   {"individual": esm_preds, "mean": esm_avg, "std": esm_std},
        }

        if sap_avg is not None:
            print(f"  → SaProtΔG ensemble: {sap_avg:.3f} ± {sap_std:.3f} kcal/mol")
        if esm_avg is not None:
            print(f"  → ESM3ΔG   ensemble: {esm_avg:.3f} ± {esm_std:.3f} kcal/mol")

    # ── Run mutational scans ──────────────────────────────────────────
    print("\n\n=== Mutational scans ===")

    for pdb_id, (filename, chain, desc, length) in EXAMPLES.items():
        print(f"\n{'='*50}")
        print(f"  SCAN: {pdb_id} — {desc}")
        pdb_path = os.path.join(EXAMPLES_DIR, filename)

        sap_scans = []
        for i, model in enumerate(sap_models, 1):
            try:
                result = SaProtABS_predict(model, pdb_path, chain, ddg_scanning=True)
                ddg = result[0] if len(result) == 2 else result[0]
                mat = ddg.mean(dim=-1).squeeze(2)
                sap_scans.append(mat)
                print(f"  SaProtΔG scan model {i}: shape {list(mat.shape)}")
            except Exception as e:
                print(f"  SaProtΔG scan model {i} FAILED: {e}")

        esm_scans = []
        for i, model in enumerate(esm_models, 1):
            try:
                result = ESM3ABS_predict(model, pdb_path, [chain], ddg_scanning=True)
                ddg = result[0] if len(result) == 2 else result[0]
                mat = ddg.mean(dim=-1).squeeze(2)
                esm_scans.append(mat)
                print(f"  ESM3ΔG   scan model {i}: shape {list(mat.shape)}")
            except Exception as e:
                print(f"  ESM3ΔG   scan model {i} FAILED: {e}")

        if sap_scans:
            sap_scan_mean = torch.stack(sap_scans).mean(dim=0)
            results[pdb_id]["saprot"]["ddg_scan"] = [
                [round(float(v), 3) for v in row] for row in sap_scan_mean
            ]
            print(f"  → SaProtΔG ensemble scan saved {list(sap_scan_mean.shape)}")

        if esm_scans:
            esm_scan_mean = torch.stack(esm_scans).mean(dim=0)
            results[pdb_id]["esm3"]["ddg_scan"] = [
                [round(float(v), 3) for v in row] for row in esm_scan_mean
            ]
            print(f"  → ESM3ΔG   ensemble scan saved {list(esm_scan_mean.shape)}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = os.path.join(_ROOT, "paper_example_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nDone! Results saved to {out_path}")
    for pdb_id, r in results.items():
        has_scan = "ddg_scan" in r.get("esm3", {}) and "ddg_scan" in r.get("saprot", {})
        print(f"  {pdb_id}: ESM3 {r['esm3']['mean']:.3f}, SaProt {r['saprot']['mean']:.3f}  scan={'yes' if has_scan else 'NO'}")


if __name__ == "__main__":
    main()
