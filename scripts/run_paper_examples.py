"""
Run ESM3ΔG and SaProtΔG predictions on example structures from the examples/ directory.
Covers nanobodies, dark matter proteins, MGnify stability domains, and TED/AlphaFold domains.

Outputs paper_example_results.json with:
  - Ensemble mean/std ΔG per protein per model (base + augmented)
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
from SaProtdG import SaProtdG, SaProtdG_predict
from ESM3dG import ESM3dG, ESM3dG_predict

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


def ensemble_stats(preds):
    if not preds:
        return None, 0
    avg = round(sum(preds) / len(preds), 3)
    std = round((sum((x - avg)**2 for x in preds) / len(preds))**0.5, 3) if len(preds) > 1 else 0
    return avg, std


def load_sap_ensemble(indices, label, cfg, device):
    models = []
    print(f"\n=== Loading SaProtΔG {label} models ===")
    for i in indices:
        tag = "" if label == "base" else "_augmented"
        w = f"saprotdg_weights/SaProtdG_weights{tag}_{i}_lora.ckpt"
        print(f"  Loading {w}")
        m = SaProtdG(w, cfg)
        m.to(device); m.device = torch.device(device)
        models.append(m)
    return models


def load_esm_ensemble(indices, label, cfg, device):
    models = []
    print(f"\n=== Loading ESM3ΔG {label} models ===")
    for i in indices:
        tag = "" if label == "base" else "_augmented"
        w = f"esm3dg_weights/ESM3dG_weights{tag}_{i}_lora.ckpt"
        print(f"  Loading {w}")
        m = ESM3dG(w, cfg)
        m.to(device); m.device = torch.device(device)
        models.append(m)
    return models


def run_dg_preds(sap_models, esm_models, pdb_path, chain, label):
    sap_preds, esm_preds = [], []
    for i, model in enumerate(sap_models, 1):
        try:
            _, avg, _ = SaProtdG_predict(model, pdb_path, chain)
            val = round(avg[0], 3)
            sap_preds.append(val)
            print(f"  SaProtΔG-{label} model {i}: {val:.3f} kcal/mol")
        except Exception as e:
            print(f"  SaProtΔG-{label} model {i} FAILED: {e}")
    for i, model in enumerate(esm_models, 1):
        try:
            _, avg, _ = ESM3dG_predict(model, pdb_path, chain)
            val = round(avg[0], 3)
            esm_preds.append(val)
            print(f"  ESM3ΔG-{label}   model {i}: {val:.3f} kcal/mol")
        except Exception as e:
            print(f"  ESM3ΔG-{label}   model {i} FAILED: {e}")
    return sap_preds, esm_preds


def run_scans(sap_models, esm_models, pdb_path, chain, label):
    sap_scans, esm_scans = [], []
    for i, model in enumerate(sap_models, 1):
        try:
            result = SaProtdG_predict(model, pdb_path, chain, ddg_scanning=True)
            mat = result[0].mean(dim=-1).squeeze(2)
            sap_scans.append(mat)
            print(f"  SaProtΔG-{label} scan model {i}: shape {list(mat.shape)}")
        except Exception as e:
            print(f"  SaProtΔG-{label} scan model {i} FAILED: {e}")
    for i, model in enumerate(esm_models, 1):
        try:
            result = ESM3dG_predict(model, pdb_path, chain, ddg_scanning=True)
            mat = result[0].mean(dim=-1).squeeze(2)
            esm_scans.append(mat)
            print(f"  ESM3ΔG-{label}   scan model {i}: shape {list(mat.shape)}")
        except Exception as e:
            print(f"  ESM3ΔG-{label}   scan model {i} FAILED: {e}")
    return sap_scans, esm_scans


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Using device: {device} ===")

    sap_cfg = get_default_config()
    sap_cfg.testing.ddg_scanning = False
    esm_cfg = get_esm3_config()
    esm_cfg.testing.ddg_scanning = False

    sap_base = load_sap_ensemble([1, 2, 3], "base",      sap_cfg, device)
    sap_aug  = load_sap_ensemble([1, 2, 3], "augmented", sap_cfg, device)
    esm_base = load_esm_ensemble([1, 2, 3], "base",      esm_cfg, device)
    esm_aug  = load_esm_ensemble([1, 2, 3], "augmented", esm_cfg, device)

    # ── Run ΔG predictions ────────────────────────────────────────────
    results = {}

    for pdb_id, (filename, chain, desc, length) in EXAMPLES.items():
        print(f"\n{'='*55}")
        print(f"  {pdb_id} — {desc} (chain {chain})")
        pdb_path = os.path.join(EXAMPLES_DIR, filename)

        sb_preds, eb_preds = run_dg_preds(sap_base, esm_base, pdb_path, chain, "base")
        sa_preds, ea_preds = run_dg_preds(sap_aug,  esm_aug,  pdb_path, chain, "aug")

        sb_avg, sb_std = ensemble_stats(sb_preds)
        sa_avg, sa_std = ensemble_stats(sa_preds)
        eb_avg, eb_std = ensemble_stats(eb_preds)
        ea_avg, ea_std = ensemble_stats(ea_preds)

        print(f"  → SaProtΔG      : {sb_avg:.3f} ± {sb_std:.3f} kcal/mol")
        print(f"  → SaProtΔG-Aug  : {sa_avg:.3f} ± {sa_std:.3f} kcal/mol")
        print(f"  → ESM3ΔG        : {eb_avg:.3f} ± {eb_std:.3f} kcal/mol")
        print(f"  → ESM3ΔG-Aug    : {ea_avg:.3f} ± {ea_std:.3f} kcal/mol")

        results[pdb_id] = {
            "description": desc,
            "chain":       chain,
            "length":      length,
            "saprot":     {"individual": sb_preds, "mean": sb_avg, "std": sb_std},
            "saprot_aug": {"individual": sa_preds, "mean": sa_avg, "std": sa_std},
            "esm3":       {"individual": eb_preds, "mean": eb_avg, "std": eb_std},
            "esm3_aug":   {"individual": ea_preds, "mean": ea_avg, "std": ea_std},
        }

    # ── Run mutational scans ──────────────────────────────────────────
    print("\n\n=== Mutational scans ===")

    for pdb_id, (filename, chain, desc, length) in EXAMPLES.items():
        print(f"\n{'='*55}")
        print(f"  SCAN: {pdb_id} — {desc}")
        pdb_path = os.path.join(EXAMPLES_DIR, filename)

        sb_scans, eb_scans = run_scans(sap_base, esm_base, pdb_path, chain, "base")
        sa_scans, ea_scans = run_scans(sap_aug,  esm_aug,  pdb_path, chain, "aug")

        for key, scans in [
            ("saprot",     sb_scans),
            ("saprot_aug", sa_scans),
            ("esm3",       eb_scans),
            ("esm3_aug",   ea_scans),
        ]:
            if scans:
                mean_scan = torch.stack(scans).mean(dim=0)
                results[pdb_id][key]["ddg_scan"] = [
                    [round(float(v), 3) for v in row] for row in mean_scan
                ]
                print(f"  → {key} scan saved {list(mean_scan.shape)}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = os.path.join(_ROOT, "paper_example_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nDone! Results saved to {out_path}")
    print(f"\n{'PDB':<22} {'SaP':>7} {'SaP-Aug':>9} {'ESM3':>7} {'ESM3-Aug':>10}")
    print("-" * 57)
    for pdb_id, r in results.items():
        print(f"  {pdb_id:<20} {r['saprot']['mean']:>7.3f} {r['saprot_aug']['mean']:>9.3f} "
              f"{r['esm3']['mean']:>7.3f} {r['esm3_aug']['mean']:>10.3f}")


if __name__ == "__main__":
    main()
