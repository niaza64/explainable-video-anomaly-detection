#!/usr/bin/env python3
"""
Video-level anomaly detection by running videos through AED-MAE.

Pipeline:
1) Select a balanced subset of SHANGHAI test videos (default: 10 anomaly + 10 normal)
2) Run each selected video through model inference to get per-frame scores
3) Smooth/normalize scores
4) Compute video-level score = mean(top-k frame scores)
5) Grid-search best (k, threshold)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Reuse inference code from existing pipeline.
from pipeline.run_pipeline import load_model, score_all_frames, smooth_and_normalize

TEST_TXT = BASE / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
DEFAULT_OUT = BASE / "video_level_detection" / "results"
DEFAULT_CACHE = BASE / "video_level_detection" / "scores_cache"


def parse_test_list(path: Path) -> List[Tuple[str, int, int]]:
    """Return list of (video_id, num_frames, label). label: 1 anomaly, 0 normal."""
    videos: List[Tuple[str, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            video_id = parts[0].split("/")[-1]
            num_frames = int(parts[1])
            label = int(parts[2])
            videos.append((video_id, num_frames, label))
    return videos


def choose_balanced_subset(
    videos: Sequence[Tuple[str, int, int]],
    num_anomaly: int,
    num_normal: int,
    seed: int,
) -> List[Tuple[str, int, int]]:
    rng = random.Random(seed)
    anomalies = [v for v in videos if v[2] == 1]
    normals = [v for v in videos if v[2] == 0]
    if len(anomalies) < num_anomaly:
        raise ValueError(f"Requested {num_anomaly} anomaly videos, found {len(anomalies)}.")
    if len(normals) < num_normal:
        raise ValueError(f"Requested {num_normal} normal videos, found {len(normals)}.")
    chosen = rng.sample(anomalies, num_anomaly) + rng.sample(normals, num_normal)
    rng.shuffle(chosen)
    return chosen


def video_topk_mean(scores: np.ndarray, k: int) -> float:
    k_eff = max(1, min(k, int(scores.shape[0])))
    topk = np.partition(scores, scores.shape[0] - k_eff)[-k_eff:]
    return float(np.mean(topk))


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


def search_best_k_threshold(
    labels: Sequence[int],
    per_video_scores: Sequence[np.ndarray],
    k_values: Sequence[int],
    threshold_steps: int,
) -> Dict[str, object]:
    best_entry = None
    leaderboard: List[Dict[str, object]] = []

    for k in k_values:
        video_scores = [video_topk_mean(s, k) for s in per_video_scores]
        vmin, vmax = min(video_scores), max(video_scores)
        thresholds = [vmin] if abs(vmax - vmin) < 1e-12 else np.linspace(vmin, vmax, threshold_steps)

        for thr in thresholds:
            preds = [1 if s >= float(thr) else 0 for s in video_scores]
            metrics = confusion_metrics(labels, preds)
            entry = {
                "k": int(k),
                "threshold": float(thr),
                **metrics,
            }
            leaderboard.append(entry)
            rank_key = (metrics["accuracy"], metrics["f1"], metrics["balanced_accuracy"])
            if best_entry is None or rank_key > best_entry["rank_key"]:
                best_entry = {"rank_key": rank_key, "entry": entry}

    assert best_entry is not None
    leaderboard.sort(key=lambda x: (x["accuracy"], x["f1"], x["balanced_accuracy"]), reverse=True)
    return {"best": best_entry["entry"], "top_results": leaderboard[:20]}


def parse_k_values(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k <= 0:
            raise ValueError("k values must be positive integers.")
        vals.append(k)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("No valid k values provided.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-level anomaly detection with top-k frame scores.")
    parser.add_argument("--num-anomaly", type=int, default=10)
    parser.add_argument("--num-normal", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=str, default="1,3,5,8,10,15,20,30,50")
    parser.add_argument("--threshold-steps", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached score files and recompute via model inference.",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_k_values(args.k_values)
    all_videos = parse_test_list(TEST_TXT)
    subset = choose_balanced_subset(
        all_videos,
        num_anomaly=args.num_anomaly,
        num_normal=args.num_normal,
        seed=args.seed,
    )

    print(f"Selected {len(subset)} videos: {args.num_anomaly} anomaly + {args.num_normal} normal")
    print("Loading model...")
    model = load_model()

    records = []
    labels: List[int] = []
    score_series: List[np.ndarray] = []

    for idx, (video_id, num_frames, label) in enumerate(subset, start=1):
        print(f"[{idx}/{len(subset)}] {video_id} (label={label}, frames={num_frames})")
        cache_path = args.cache_dir / f"{video_id}_smoothed_scores.npy"

        if cache_path.exists() and not args.no_cache:
            smoothed = np.load(cache_path)
            print(f"  using cache: {cache_path}")
        else:
            raw = score_all_frames(model, video_id, num_frames)
            smoothed = smooth_and_normalize(raw)
            np.save(cache_path, smoothed)
            print(f"  wrote cache: {cache_path}")

        labels.append(label)
        score_series.append(smoothed)
        records.append(
            {
                "video_id": video_id,
                "label": label,
                "num_frames": num_frames,
                "num_scored_frames": int(smoothed.shape[0]),
                "cache_file": str(cache_path),
            }
        )

    tuning = search_best_k_threshold(
        labels=labels,
        per_video_scores=score_series,
        k_values=k_values,
        threshold_steps=args.threshold_steps,
    )
    best = tuning["best"]
    best_k = int(best["k"])
    best_thr = float(best["threshold"])

    final_per_video = []
    preds = []
    for rec, sc in zip(records, score_series):
        vscore = video_topk_mean(sc, best_k)
        pred = 1 if vscore >= best_thr else 0
        preds.append(pred)
        final_per_video.append(
            {
                "video_id": rec["video_id"],
                "label": rec["label"],
                "video_score": float(vscore),
                "prediction": int(pred),
                "correct": bool(pred == rec["label"]),
            }
        )

    final_metrics = confusion_metrics(labels, preds)

    report = {
        "config": {
            "num_anomaly": args.num_anomaly,
            "num_normal": args.num_normal,
            "seed": args.seed,
            "k_values": k_values,
            "threshold_steps": args.threshold_steps,
            "cache_dir": str(args.cache_dir),
            "no_cache": bool(args.no_cache),
        },
        "selected_videos": records,
        "tuning": tuning,
        "best_params": {"k": best_k, "threshold": best_thr},
        "final_metrics_on_subset": final_metrics,
        "final_per_video": final_per_video,
    }

    out_path = args.output_dir / "video_level_tuning_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"Best k={best_k}, threshold={best_thr:.6f}")
    print(f"Subset accuracy={final_metrics['accuracy']:.4f}, F1={final_metrics['f1']:.4f}")
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
