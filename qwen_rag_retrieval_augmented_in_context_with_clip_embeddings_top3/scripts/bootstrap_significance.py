#!/usr/bin/env python3
"""
Bootstrap CI analysis for all pairwise comparisons in the paper.
Reproduces the significance table that goes into the paper appendix.

Usage:
  python bootstrap_significance.py            # print table
  python bootstrap_significance.py --json out.json  # also dump JSON
"""
import argparse
import json
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PATHS = {
    "zero-shot": REPO / "rtfm/rtfm_pipeline_outputs/qwen_zeroshot_judge_summary.json",
    "LoRA v1":   REPO / "qwen_finetune_judge_summary.json",
    "LoRA v2":   REPO / "qwen_finetune_v2/qwen_finetune_v2_judge_summary.json",
    "LoRA v3":   REPO / "qwen_finetune_v3_enriched_labels_single_strategy_with_normal_video_negatives/qwen_finetune_v3_judge_summary.json",
    "RAG":       REPO / "qwen_rag_retrieval_augmented_in_context_with_clip_embeddings_top3/qwen_rag_top3_clip_judge_summary.json",
}

CRITERIA = ["correctness", "specificity", "completeness", "fluency"]


def load_anom(p):
    data = json.load(open(p))
    return {r["video_id"]: r["scores"] for r in data if r.get("video_type") == "anomalous"}


def overall(score):
    return sum(score[k] for k in CRITERIA) / len(CRITERIA)


def paired_bootstrap(A, B, n_boot=10000, seed=42):
    common = sorted(set(A) & set(B))
    if not common:
        return None
    a = np.array([overall(A[v]) for v in common])
    b = np.array([overall(B[v]) for v in common])
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    boot_deltas = diff[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    return {
        "n_paired":     len(common),
        "mean_A":       float(a.mean()),
        "mean_B":       float(b.mean()),
        "delta":        float(diff.mean()),
        "ci_lo":        float(ci_lo),
        "ci_hi":        float(ci_hi),
        "p_pos":        float((boot_deltas > 0).mean()),
        "significant":  bool(ci_lo > 0 or ci_hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    variants = {name: load_anom(p) for name, p in PATHS.items()}
    print(f"Loaded variants: " + ", ".join(f"{n}={len(v)}" for n, v in variants.items()))
    print()

    # All pairwise (directional)
    pairs = [
        ("RAG", "zero-shot"),
        ("RAG", "LoRA v1"), ("RAG", "LoRA v2"), ("RAG", "LoRA v3"),
        ("zero-shot", "LoRA v1"), ("zero-shot", "LoRA v2"), ("zero-shot", "LoRA v3"),
        ("LoRA v2", "LoRA v1"), ("LoRA v3", "LoRA v2"), ("LoRA v3", "LoRA v1"),
    ]

    results = []
    print(f"{'A':<12s} vs {'B':<12s} {'n':>3s} {'A_mean':>7s} {'B_mean':>7s} "
          f"{'Δ':>7s} {'95% CI':>22s} {'p>0':>6s} {'sig':>4s}")
    print("-" * 90)
    for A, B in pairs:
        r = paired_bootstrap(variants[A], variants[B])
        if not r: continue
        sig = "✓" if r["significant"] else " "
        print(f"{A:<12s} vs {B:<12s} {r['n_paired']:>3d} {r['mean_A']:>7.3f} {r['mean_B']:>7.3f} "
              f"{r['delta']:>+7.3f}   [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]   {r['p_pos']:>5.3f}  {sig}")
        results.append({"A": A, "B": B, **r})

    print()
    print("Legend: ✓ = 95% bootstrap CI excludes 0; p>0 = fraction of bootstrap "
          "resamples where mean(A) > mean(B)")

    if args.json:
        args.json.write_text(json.dumps({"variants": {k: len(v) for k,v in variants.items()},
                                         "comparisons": results}, indent=2))
        print(f"\nJSON written to {args.json}")


if __name__ == "__main__":
    main()
