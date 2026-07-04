#!/usr/bin/env python3
"""
Analyse the judge correlation study after the human rater has filled in
the human_C/S/Co/F columns in judge_correlation_sheet.csv.

Computes per-criterion Pearson + Spearman correlations between the human
rater and the GPT-4o judge, plus overall agreement.

Run:
  python judge_correlation_analysis.py [path_to_filled_csv]
"""
import csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr

DEFAULT = Path(__file__).resolve().parent.parent / "results" / "judge_correlation_sheet.csv"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
print(f"Reading: {path}")

rows = list(csv.DictReader(open(path)))
filled = [r for r in rows if all(r[f"human_{c} (1-5)"].strip() for c in ["C","S","Co","F"])]
print(f"Total rows: {len(rows)}  |  Rows with human ratings: {len(filled)}")
if len(filled) < 5:
    print("⚠  Fewer than 5 rows filled — fill the CSV first, then re-run this script.")
    sys.exit(0)

# Extract paired arrays
def col(name): return np.array([float(r[name]) for r in filled])
human = {
    "C":  col("human_C (1-5)"),
    "S":  col("human_S (1-5)"),
    "Co": col("human_Co (1-5)"),
    "F":  col("human_F (1-5)"),
}
gpt = {
    "C":  col("GPT_C_HIDE_until_done"),
    "S":  col("GPT_S_HIDE"),
    "Co": col("GPT_Co_HIDE"),
    "F":  col("GPT_F_HIDE"),
}

print()
print(f"{'Criterion':<14s}  {'human mean':>11s}  {'gpt mean':>9s}  {'Pearson r':>10s}  {'Spearman ρ':>11s}  {'MAE':>5s}")
print("-" * 78)
results = {}
for c, name in [("C","Correctness"),("S","Specificity"),("Co","Completeness"),("F","Fluency")]:
    h = human[c]; g = gpt[c]
    r, _   = pearsonr(h, g)
    rho, _ = spearmanr(h, g)
    mae = np.mean(np.abs(h - g))
    results[c] = {"pearson": float(r), "spearman": float(rho), "mae": float(mae), "n": len(h),
                  "human_mean": float(h.mean()), "gpt_mean": float(g.mean())}
    print(f"{name:<14s}  {h.mean():>11.2f}  {g.mean():>9.2f}  {r:>10.3f}  {rho:>11.3f}  {mae:>5.2f}")

# Overall (mean of 4 criteria) correlation
h_ovr = np.mean([human[c] for c in human], axis=0)
g_ovr = np.mean([gpt[c]   for c in gpt],   axis=0)
r, _   = pearsonr(h_ovr, g_ovr)
rho, _ = spearmanr(h_ovr, g_ovr)
mae = np.mean(np.abs(h_ovr - g_ovr))
print("-" * 78)
print(f"{'OVERALL (mean)':<14s}  {h_ovr.mean():>11.2f}  {g_ovr.mean():>9.2f}  {r:>10.3f}  {rho:>11.3f}  {mae:>5.2f}")

print()
print("Interpretation guide:")
print("  Pearson  r ≥ 0.7  → strong linear agreement (judge tracks human magnitude)")
print("  Spearman ρ ≥ 0.7  → strong rank agreement (judge orders the same as human)")
print("  MAE      ≤ 1.0    → typical disagreement <1 point on the 1-5 Likert")
EOF
