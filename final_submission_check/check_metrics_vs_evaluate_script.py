"""
Cross-check: compare metrics from evaluate_test_set.py (model-inference run)
against metrics recomputed from the per-sample prediction CSVs used in the
paper figure notebook (00_mgnify_test_spearman_fig2ab.ipynb).

evaluate_test_set.py uses the NON-augmented ensemble:
  saprotdg_weights/SaProtdG_weights_{1,2,3}_lora.ckpt
  esm3dg_weights/ESM3dG_weights_{1,2,3}_lora.ckpt

The notebook CSVs for the non-augmented models are listed below.
If the numbers match, the two pipelines are consistent.

Per-sample ordering (determined empirically):
  ESM3  : CSV[0] → JSON esm_preds[2], CSV[1] → [1], CSV[2] → [0]
  SaProt: CSV[0] → JSON sap_preds[0], CSV[1] → [1], CSV[2] → [2]
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

DG_MIN, DG_MAX = -1.0, 5.0
TOL = 1e-4   # per-sample tolerance

# ── CSV paths from the notebook ───────────────────────────────────────────────

ESM3_CSV = [
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model3_resume_model2_norm/dmsv4_af_dg_epoch01_test_result.csv",
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model1_resume_model1_norm/dmsv4_af_dg_epoch03_test_result.csv",
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_sigmoid_5e5_model2_resume_model1_norm/dmsv4_af_dg_epoch07_test_result.csv",
]

ESM3_AUG_CSV = [
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model1_resume_augment_model1_norm/dmsv4_af_dg_epoch06_test_result.csv",
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model2_resume_augment_model1_norm/dmsv4_af_dg_epoch10_test_result.csv",
    "/home/jupyter-yehlin/esm3_attention/weights/checkpoint_unfreeze_warum_up_ddg_bins_lora_dg_and_ddg_filtered_augmented_sigmoid_5e5_model3_resume_augment_model1_norm/dmsv4_af_dg_epoch05_test_result.csv",
]

SAPROT_CSV = [
    "/ssd/yehlin/SaProt_dG/weights/Saprot_lora_mgnify_ddg_no_confidence_sigmoid_both_model1_1e4/dmsv4_af_dg_epoch04_test_result.csv",
    "/ssd/yehlin/SaProt_dG/weights/Saprot_lora_mgnify_ddg_no_confidence_sigmoid_both_model2_1e4/dmsv4_af_dg_epoch02_test_result.csv",
    "/data/yehlin/SaprotABS/weights/Saprot_lora_mgnify_ddg_no_confidence_sigmoid_both_model9_1e4/dmsv4_af_dg_epoch00_test_result.csv",
]

SAPROT_AUG_CSV = [
    "/home/jupyter-yehlin/SaProt_dG/weights/Saprot_lora_mgnify_dg_and_ddg_no_confidence_model3_1e4_terminus_trancate_augment_d_sigmoid_both/dmsv4_af_dg_epoch01_test_result.csv",
    "/home/jupyter-yehlin/SaProt_dG/weights/Saprot_lora_mgnify_dg_and_ddg_no_confidence_model6_5e5_terminus_trancate_augment_d_sigmoid_both/dmsv4_af_dg_epoch07_test_result.csv",
    "/home/jupyter-yehlin/SaProt_dG/weights/Saprot_lora_mgnify_dg_and_ddg_no_confidence_model7_5e5_terminus_trancate_augment_d_sigmoid_both/dmsv4_af_dg_epoch11_test_result.csv",
]

EVAL_JSON = "/home/jupyter-yehlin/ESM3_SaProt_dG/test_set_evaluation_results.json"
# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    df["True"] = df["True"].clip(DG_MIN, DG_MAX)
    pred_col = "Predicted_sigmoid" if "Predicted_sigmoid" in df.columns else "Predicted"
    return df["True"].values, df[pred_col].values


def load_csv_by_name(path):
    """Return dict: pdb_name -> prediction value."""
    df = pd.read_csv(path)
    pred_col = "Predicted_sigmoid" if "Predicted_sigmoid" in df.columns else "Predicted"
    return dict(zip(df["pdb"], df[pred_col]))


def metrics(true, pred):
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    sp   = float(spearmanr(pred, true).statistic)
    return rmse, sp


def ensemble_metrics(csv_list, label):
    dfs = [load_csv(p) for p in csv_list]
    true = dfs[0][0]
    avg_pred = np.mean([d[1] for d in dfs], axis=0)
    rmse, sp = metrics(true, avg_pred)
    print(f"  {label}: n={len(true)}  RMSE={rmse:.4f}  Spearman={sp:.4f}")
    return rmse, sp


def individual_metrics(csv_list, label_prefix):
    for i, path in enumerate(csv_list, 1):
        true, pred = load_csv(path)
        rmse, sp = metrics(true, pred)
        print(f"  {label_prefix} model {i}: RMSE={rmse:.4f}  Spearman={sp:.4f}")


# ── Per-sample individual-value comparison ────────────────────────────────────

def compare_individual_values(csv_list, json_records, json_pred_key, label):
    """
    For every sample present in both the CSVs and the JSON, compare each
    model's individual prediction value.

    Strategy: sort the 3 predictions from each source per sample and compare
    element-wise (order-invariant, robust to model-index differences).
    """
    # Build per-name lookup from CSVs: name -> sorted list of predictions
    csv_preds = {}   # name -> [pred_csv0, pred_csv1, pred_csv2]
    for path in csv_list:
        d = load_csv_by_name(path)
        for name, pred in d.items():
            csv_preds.setdefault(name, []).append(float(pred))

    # Build per-name lookup from JSON
    json_preds = {r["name"]: r[json_pred_key] for r in json_records
                  if r[json_pred_key]}

    common = sorted(set(csv_preds) & set(json_preds))
    n_total = len(common)

    max_diff_overall  = 0.0
    n_sample_mismatch = 0
    worst_samples     = []   # (max_diff, name, csv_sorted, json_sorted)

    for name in common:
        csv_sorted  = sorted(csv_preds[name])
        json_sorted = sorted(json_preds[name])
        if len(csv_sorted) != len(json_sorted):
            n_sample_mismatch += 1
            continue
        diffs = [abs(a - b) for a, b in zip(csv_sorted, json_sorted)]
        max_diff = max(diffs)
        if max_diff > max_diff_overall:
            max_diff_overall = max_diff
        if max_diff > TOL:
            n_sample_mismatch += 1
            worst_samples.append((max_diff, name, csv_sorted, json_sorted))

    worst_samples.sort(reverse=True)

    print(f"\n[{label}] Per-sample individual-value comparison")
    print(f"  Samples compared : {n_total}")
    print(f"  Tolerance        : {TOL:.0e}")
    print(f"  Max diff (all)   : {max_diff_overall:.2e}")
    print(f"  Samples > tol    : {n_sample_mismatch}")

    if n_sample_mismatch == 0:
        print("  RESULT: all individual values match within tolerance.")
    else:
        print(f"  RESULT: {n_sample_mismatch} sample(s) differ — top mismatches:")
        for diff, name, csv_s, json_s in worst_samples[:5]:
            print(f"    {name}")
            print(f"      CSV  (sorted): {[f'{v:.6f}' for v in csv_s]}")
            print(f"      JSON (sorted): {[f'{v:.6f}' for v in json_s]}")
            print(f"      max diff     : {diff:.2e}")

    return n_sample_mismatch == 0


def compare_per_sample_means(csv_list, json_records, json_mean_key, label):
    """Compare per-sample ensemble-mean predictions between CSV and JSON."""
    # Build per-name mean from CSVs
    csv_by_name = {}  # name -> list of preds
    for path in csv_list:
        for name, pred in load_csv_by_name(path).items():
            csv_by_name.setdefault(name, []).append(float(pred))
    csv_means = {n: float(np.mean(ps)) for n, ps in csv_by_name.items()}

    json_means = {r["name"]: r[json_mean_key] for r in json_records
                  if r[json_mean_key] is not None}

    common = sorted(set(csv_means) & set(json_means))
    diffs  = [abs(csv_means[n] - json_means[n]) for n in common]
    max_diff  = max(diffs) if diffs else 0.0
    n_mismatch = sum(1 for d in diffs if d > TOL)

    print(f"\n[{label}] Per-sample ensemble-mean comparison")
    print(f"  Samples compared : {len(common)}")
    print(f"  Max diff         : {max_diff:.2e}")
    print(f"  Samples > tol    : {n_mismatch}")
    if n_mismatch == 0:
        print("  RESULT: all per-sample means match within tolerance.")
    else:
        print(f"  RESULT: {n_mismatch} mean(s) differ.")
    return n_mismatch == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Metrics from notebook CSVs (non-augmented ensembles)")
    print("=" * 65)

    print("\nIndividual ESM3 (non-aug):")
    individual_metrics(ESM3_CSV, "ESM3")

    print("\nIndividual SaProt (non-aug):")
    individual_metrics(SAPROT_CSV, "SaProt")

    print("\nEnsemble (non-augmented):")
    nb_esm3_rmse,   nb_esm3_sp   = ensemble_metrics(ESM3_CSV,    "ESM3 Ensemble")
    nb_saprot_rmse, nb_saprot_sp = ensemble_metrics(SAPROT_CSV,  "SaProt Ensemble")

    print("\nEnsemble (augmented):")
    nb_esm3_aug_rmse,   nb_esm3_aug_sp   = ensemble_metrics(ESM3_AUG_CSV,   "ESM3 Augmented Ensemble")
    nb_saprot_aug_rmse, nb_saprot_aug_sp = ensemble_metrics(SAPROT_AUG_CSV, "SaProt Augmented Ensemble")

    # ── Load JSON ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Comparing with {EVAL_JSON}")
    print("=" * 65)

    try:
        with open(EVAL_JSON) as f:
            ev_data = json.load(f)
    except FileNotFoundError:
        print("  [!] JSON not found — run evaluate_test_set.py first")
        return

    ev      = ev_data["metrics"]
    records = ev_data["records"]

    # ── Aggregate-metric comparison ───────────────────────────────────────────
    AGG_TOL = 1e-3
    rows = [
        ("ESM3",       "RMSE",     nb_esm3_rmse,   ev["esm3"]["rmse"]),
        ("ESM3",       "Spearman", nb_esm3_sp,     ev["esm3"]["spearman"]),
        ("SaProt",     "RMSE",     nb_saprot_rmse, ev["saprot"]["rmse"]),
        ("SaProt",     "Spearman", nb_saprot_sp,   ev["saprot"]["spearman"]),
        ("ESM3-Aug",   "RMSE",     nb_esm3_aug_rmse,   ev["esm3_aug"]["rmse"]),
        ("ESM3-Aug",   "Spearman", nb_esm3_aug_sp,     ev["esm3_aug"]["spearman"]),
        ("SaProt-Aug", "RMSE",     nb_saprot_aug_rmse, ev["saprot_aug"]["rmse"]),
        ("SaProt-Aug", "Spearman", nb_saprot_aug_sp,   ev["saprot_aug"]["spearman"]),
    ]

    print(f"\n{'Model':<12} {'Metric':<10} {'Notebook CSV':>14} {'evaluate.py':>14} {'Diff':>10} {'Match?':>8}")
    print("-" * 70)
    all_agg_match = True
    for model, metric, nb_val, ev_val in rows:
        diff  = abs(nb_val - ev_val)
        match = diff < AGG_TOL
        all_agg_match = all_agg_match and match
        flag  = "OK" if match else "MISMATCH"
        print(f"{model:<12} {metric:<10} {nb_val:>14.4f} {ev_val:>14.4f} {diff:>10.4f} {flag:>8}")

    print()
    if all_agg_match:
        print("Aggregate metrics: all match within tolerance.")
    else:
        print("Aggregate metrics: MISMATCH detected.")

    # ── Per-sample ensemble-mean comparison ───────────────────────────────────
    print("\n" + "=" * 65)
    print("Per-sample ensemble-mean comparison")
    print("=" * 65)
    ok_esm3_mean       = compare_per_sample_means(ESM3_CSV,        records, "esm_mean",     "ESM3")
    ok_saprot_mean     = compare_per_sample_means(SAPROT_CSV,      records, "sap_mean",     "SaProt")
    ok_esm3_aug_mean   = compare_per_sample_means(ESM3_AUG_CSV,    records, "esm_aug_mean", "ESM3-Aug")
    ok_saprot_aug_mean = compare_per_sample_means(SAPROT_AUG_CSV,  records, "sap_aug_mean", "SaProt-Aug")

    # ── Per-sample individual-value comparison ────────────────────────────────
    print("\n" + "=" * 65)
    print("Per-sample individual-value comparison (sorted, tol={:.0e})".format(TOL))
    print("=" * 65)
    ok_esm3_ind       = compare_individual_values(ESM3_CSV,       records, "esm_preds",     "ESM3")
    ok_saprot_ind     = compare_individual_values(SAPROT_CSV,     records, "sap_preds",     "SaProt")
    ok_esm3_aug_ind   = compare_individual_values(ESM3_AUG_CSV,   records, "esm_aug_preds", "ESM3-Aug")
    ok_saprot_aug_ind = compare_individual_values(SAPROT_AUG_CSV, records, "sap_aug_preds", "SaProt-Aug")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    all_ok = (all_agg_match
              and ok_esm3_mean and ok_saprot_mean
              and ok_esm3_aug_mean and ok_saprot_aug_mean
              and ok_esm3_ind and ok_saprot_ind
              and ok_esm3_aug_ind and ok_saprot_aug_ind)
    if all_ok:
        print("FINAL RESULT: pipelines are fully consistent at every level.")
    else:
        print("FINAL RESULT: discrepancies found — see details above.")


if __name__ == "__main__":
    main()
