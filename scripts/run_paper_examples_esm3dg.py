"""
Run ESM3ΔG predictions on example structures from the examples/ directory.
Covers nanobodies, dark matter proteins, MGnify stability domains, and TED/AlphaFold domains.
"""
import os, sys, json
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

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


def load_ensemble(indices, label, device):
    models = []
    print(f"\n=== Loading ESM3ΔG {label} models ===")
    for i in indices:
        tag = "" if label == "base" else "_augmented"
        w = f"esm3dg_weights/ESM3dG_weights{tag}_{i}_lora.ckpt"
        print(f"  Loading {w}")
        m = ESM3dG(w)
        m.to(device)
        models.append(m)
    return models


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Using device: {device} ===")

    base_models = load_ensemble([1, 2, 3], "base",      device)
    aug_models  = load_ensemble([1, 2, 3], "augmented", device)

    results = {}

    for pdb_id, (filename, chain, desc, length) in EXAMPLES.items():
        print(f"\n{'='*55}")
        print(f"  {pdb_id} — {desc} (chain {chain})")
        pdb_path = os.path.join(EXAMPLES_DIR, filename)

        b_preds, a_preds = [], []
        for i, model in enumerate(base_models, 1):
            try:
                _, avg, _ = ESM3dG_predict(model, pdb_path, chain)
                val = round(avg[0], 3)
                b_preds.append(val)
                print(f"  ESM3ΔG-base model {i}: {val:.3f} kcal/mol")
            except Exception as e:
                print(f"  ESM3ΔG-base model {i} FAILED: {e}")

        for i, model in enumerate(aug_models, 1):
            try:
                _, avg, _ = ESM3dG_predict(model, pdb_path, chain)
                val = round(avg[0], 3)
                a_preds.append(val)
                print(f"  ESM3ΔG-aug  model {i}: {val:.3f} kcal/mol")
            except Exception as e:
                print(f"  ESM3ΔG-aug  model {i} FAILED: {e}")

        b_avg, b_std = ensemble_stats(b_preds)
        a_avg, a_std = ensemble_stats(a_preds)
        print(f"  → ESM3ΔG     : {b_avg:.3f} ± {b_std:.3f} kcal/mol")
        print(f"  → ESM3ΔG-Aug : {a_avg:.3f} ± {a_std:.3f} kcal/mol")

        results[pdb_id] = {
            "description": desc,
            "chain":       chain,
            "length":      length,
            "esm3":     {"individual": b_preds, "mean": b_avg, "std": b_std},
            "esm3_aug": {"individual": a_preds, "mean": a_avg, "std": a_std},
        }

    # Mutational scans
    print("\n\n=== Mutational scans ===")
    for pdb_id, (filename, chain, desc, length) in EXAMPLES.items():
        print(f"\n{'='*55}")
        print(f"  SCAN: {pdb_id} — {desc}")
        pdb_path = os.path.join(EXAMPLES_DIR, filename)

        for key, models in [("esm3", base_models), ("esm3_aug", aug_models)]:
            scans = []
            for i, model in enumerate(models, 1):
                try:
                    result = ESM3dG_predict(model, pdb_path, chain, ddg_scanning=True)
                    mat = result[0].mean(dim=-1).squeeze(2)
                    scans.append(mat)
                    print(f"  {key} model {i}: shape {list(mat.shape)}")
                except Exception as e:
                    print(f"  {key} model {i} FAILED: {e}")
            if scans:
                mean_scan = torch.stack(scans).mean(dim=0)
                results[pdb_id][key]["ddg_scan"] = [
                    [round(float(v), 3) for v in row] for row in mean_scan
                ]
                print(f"  → {key} scan saved {list(mean_scan.shape)}")

    out_path = os.path.join(_ROOT, "paper_example_results_esm3dg.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nDone! Results saved to {out_path}")
    print(f"\n{'PDB':<22} {'ESM3':>7} {'ESM3-Aug':>10}")
    print("-" * 40)
    for pdb_id, r in results.items():
        print(f"  {pdb_id:<20} {r['esm3']['mean']:>7.3f} {r['esm3_aug']['mean']:>10.3f}")


if __name__ == "__main__":
    main()
