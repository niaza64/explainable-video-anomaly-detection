#!/usr/bin/env python3
"""
Evaluate video-level anomaly detection on all SHANGHAI test videos.

By default, this script loads best (k, threshold) from:
    video_level_detection/results/video_level_tuning_report.json

Then it:
1) runs every test video through AED-MAE (or cache),
2) computes video score = mean(top-k frame scores),
3) predicts anomaly using threshold,
4) reports metrics + confusion matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipeline.run_pipeline import load_model, score_all_frames, smooth_and_normalize
from video_level_detection.run_video_level_detection import parse_test_list, video_topk_mean

TEST_TXT = BASE / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
DEFAULT_CACHE = BASE / "video_level_detection" / "scores_cache"
DEFAULT_OUT = BASE / "video_level_detection" / "results"
DEFAULT_TUNING = DEFAULT_OUT / "video_level_tuning_report.json"


def confusion_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = 0.5 * (recall + tnr)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
    }


def load_params_from_tuning(path: Path) -> Tuple[int, float]:
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    best = report.get("best_params")
    if not best:
        raise ValueError(f"Could not find 'best_params' in tuning report: {path}")
    return int(best["k"]), float(best["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best video-level params on all test videos.")
    parser.add_argument("--test-list", type=Path, default=TEST_TXT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tuning-report", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--k", type=int, default=None, help="Override k (else load from tuning report)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold (else load from tuning report)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached score files and recompute by model inference.",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.k is not None and args.threshold is not None:
        k = int(args.k)
        threshold = float(args.threshold)
    elif args.k is None and args.threshold is None:
        k, threshold = load_params_from_tuning(args.tuning_report)
    else:
        raise ValueError("Provide both --k and --threshold together, or neither.")

    videos = parse_test_list(args.test_list)
    print(f"Evaluating {len(videos)} videos with k={k}, threshold={threshold:.6f}")
    print("Loading model...")
    model = load_model()

    y_true: List[int] = []
    y_pred: List[int] = []
    per_video: List[Dict[str, object]] = []

    for idx, (video_id, num_frames, label) in enumerate(videos, start=1):
        print(f"[{idx}/{len(videos)}] {video_id} (label={label}, frames={num_frames})")
        cache_path = args.cache_dir / f"{video_id}_smoothed_scores.npy"

        if cache_path.exists() and not args.no_cache:
            smoothed = np.load(cache_path)
            print(f"  using cache: {cache_path}")
        else:
            raw = score_all_frames(model, video_id, num_frames)
            smoothed = smooth_and_normalize(raw)
            np.save(cache_path, smoothed)
            print(f"  wrote cache: {cache_path}")

        vscore = video_topk_mean(smoothed, k)
        pred = 1 if vscore >= threshold else 0

        y_true.append(label)
        y_pred.append(pred)
        per_video.append(
            {
                "video_id": video_id,
                "label": int(label),
                "video_score": float(vscore),
                "prediction": int(pred),
                "correct": bool(pred == label),
                "num_frames": int(num_frames),
                "num_scored_frames": int(smoothed.shape[0]),
            }
        )

    metrics = confusion_metrics(y_true, y_pred)
    tp = int(metrics["tp"])
    tn = int(metrics["tn"])
    fp = int(metrics["fp"])
    fn = int(metrics["fn"])

    report = {
        "config": {
            "k": int(k),
            "threshold": float(threshold),
            "test_list": str(args.test_list),
            "cache_dir": str(args.cache_dir),
            "no_cache": bool(args.no_cache),
        },
        "num_videos": len(videos),
        "metrics": metrics,
        "confusion_matrix": {
            "rows": "actual [normal, anomaly]",
            "cols": "predicted [normal, anomaly]",
            "matrix": [[tn, fp], [fn, tp]],
        },
        "per_video": per_video,
    }

    out_path = args.output_dir / "video_level_full_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Confusion matrix [[TN, FP], [FN, TP]] = [[{tn}, {fp}], [{fn}, {tp}]]")
    print(f"Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, BA={metrics['balanced_accuracy']:.4f}")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
