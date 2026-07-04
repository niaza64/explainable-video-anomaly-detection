#!/usr/bin/env python3
"""
Video-level anomaly detection V2.

Compares two scoring methods on the same calibration split:
1) full_frames_mean: mean score across all scored frames in a video.
2) segment_mean_max: detect high-score segments and take max(mean segment score).

For each method, calibrate threshold on a balanced subset (default 10 anomaly + 10 normal),
then choose the better method by calibration metrics and evaluate full test set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from pipeline.run_pipeline import (  # noqa: E402
    find_anomaly_segments,
    load_model,
    score_all_frames,
    smooth_and_normalize,
)

TEST_TXT = BASE / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
OUT_DIR = BASE / "results_v2_segment_threshold_detection"
CACHE_DIR = OUT_DIR / "scores_cache_v2"


@dataclass
class VideoRecord:
    video_id: str
    num_frames: int
    label: int  # 1=anomalous, 0=normal


def parse_test_list(path: Path) -> List[VideoRecord]:
    videos: List[VideoRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            videos.append(
                VideoRecord(
                    video_id=parts[0].split("/")[-1],
                    num_frames=int(parts[1]),
                    label=int(parts[2]),
                )
            )
    return videos


def choose_balanced_subset(
    videos: Sequence[VideoRecord],
    num_anomaly: int,
    num_normal: int,
    seed: int,
) -> List[VideoRecord]:
    rng = random.Random(seed)
    anomalies = [v for v in videos if v.label == 1]
    normals = [v for v in videos if v.label == 0]
    if len(anomalies) < num_anomaly:
        raise ValueError(f"Need {num_anomaly} anomaly videos, found {len(anomalies)}")
    if len(normals) < num_normal:
        raise ValueError(f"Need {num_normal} normal videos, found {len(normals)}")
    picked = rng.sample(anomalies, num_anomaly) + rng.sample(normals, num_normal)
    rng.shuffle(picked)
    return picked


def confusion_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": float((tp + tn) / total if total else 0.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (recall + tnr)),
    }


def load_or_compute_scores(model, video: VideoRecord, no_cache: bool) -> np.ndarray:
    cache_path = CACHE_DIR / f"{video.video_id}_smoothed_scores.npy"
    if cache_path.exists() and not no_cache:
        return np.load(cache_path)

    raw = score_all_frames(model, video.video_id, video.num_frames)
    smoothed = smooth_and_normalize(raw)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, smoothed)
    return smoothed


def segment_mean_max_score(
    scores: np.ndarray,
    tau_high: float,
    tau_low: float,
    min_len: int,
) -> Tuple[List[Dict[str, float]], float]:
    """
    Returns:
      - segment summaries with mean scores
      - video_score = max segment mean score (0 if no segment)
    """
    segments = find_anomaly_segments(
        scores=scores,
        tau_high=tau_high,
        tau_low=tau_low,
        min_len=min_len,
    )

    segment_rows: List[Dict[str, float]] = []
    for start, end in segments:
        seg_scores = scores[start:end]
        if seg_scores.size == 0:
            continue
        seg_mean = float(np.mean(seg_scores))
        segment_rows.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "length": int(end - start),
                "mean_score": seg_mean,
                "max_score": float(np.max(seg_scores)),
            }
        )

    if not segment_rows:
        return [], 0.0

    video_score = max(r["mean_score"] for r in segment_rows)
    return segment_rows, float(video_score)


def full_frames_mean_score(scores: np.ndarray) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores))


def find_best_threshold(
    labels: Sequence[int],
    video_scores: Sequence[float],
    threshold_steps: int = 400,
) -> Dict[str, object]:
    smin = min(video_scores)
    smax = max(video_scores)
    thresholds = [smin] if abs(smax - smin) < 1e-12 else np.linspace(smin, smax, threshold_steps)

    best = None
    leaderboard = []
    for thr in thresholds:
        preds = [1 if s >= float(thr) else 0 for s in video_scores]
        m = confusion_metrics(labels, preds)
        row = {"threshold": float(thr), **m}
        leaderboard.append(row)
        rank = (m["accuracy"], m["f1"], m["balanced_accuracy"])
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "row": row}

    leaderboard.sort(key=lambda x: (x["accuracy"], x["f1"], x["balanced_accuracy"]), reverse=True)
    return {
        "best": best["row"],
        "top_results": leaderboard[:20],
    }


def score_videos_with_both_methods(
    model,
    videos: Sequence[VideoRecord],
    no_cache: bool,
    tau_high: float,
    tau_low: float,
    min_len: int,
    title: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    print(f"\n=== {title} ===")
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.video_id} (label={video.label}, frames={video.num_frames})")
        scores = load_or_compute_scores(model, video, no_cache=no_cache)
        segments, seg_score = segment_mean_max_score(
            scores=scores,
            tau_high=tau_high,
            tau_low=tau_low,
            min_len=min_len,
        )
        full_score = full_frames_mean_score(scores)
        print(
            f"  segments={len(segments)}  "
            f"full_mean={full_score:.6f}  "
            f"segment_mean_max={seg_score:.6f}"
        )
        rows.append(
            {
                "video_id": video.video_id,
                "label": int(video.label),
                "num_frames": int(video.num_frames),
                "num_scored_frames": int(scores.shape[0]),
                "num_segments": int(len(segments)),
                "segments": segments,
                "scores": {
                    "full_frames_mean": float(full_score),
                    "segment_mean_max": float(seg_score),
                },
            }
        )
    return rows


def evaluate_method(rows: Sequence[Dict[str, object]], method_key: str, threshold: float) -> Dict[str, object]:
    y_true = [int(r["label"]) for r in rows]
    y_pred = [1 if float(r["scores"][method_key]) >= threshold else 0 for r in rows]
    metrics = confusion_metrics(y_true, y_pred)
    per_video = []
    for r, pred in zip(rows, y_pred):
        per_video.append(
            {
                **r,
                "selected_method": method_key,
                "threshold": float(threshold),
                "video_score": float(r["scores"][method_key]),
                "prediction": int(pred),
                "correct": bool(pred == int(r["label"])),
            }
        )
    return {"metrics": metrics, "per_video": per_video}


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-level segment-threshold anomaly detection V2.")
    parser.add_argument("--num-anomaly", type=int, default=10, help="Calibration anomalies")
    parser.add_argument("--num-normal", type=int, default=10, help="Calibration normals")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for calibration subset")
    parser.add_argument("--threshold-steps", type=int, default=400, help="Threshold sweep granularity")
    parser.add_argument("--tau-high", type=float, default=0.5, help="Segment enter threshold")
    parser.add_argument("--tau-low", type=float, default=0.3, help="Segment exit threshold")
    parser.add_argument("--min-len", type=int, default=8, help="Minimum segment length")
    parser.add_argument("--no-cache", action="store_true", help="Recompute scores from model")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_videos = parse_test_list(TEST_TXT)
    calib_videos = choose_balanced_subset(
        all_videos,
        num_anomaly=args.num_anomaly,
        num_normal=args.num_normal,
        seed=args.seed,
    )

    print("Loading AED-MAE model...")
    model = load_model()

    # 1) Calibration scoring
    calib_rows = score_videos_with_both_methods(
        model=model,
        videos=calib_videos,
        no_cache=args.no_cache,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        min_len=args.min_len,
        title="Calibration subset (10 anomaly + 10 normal by default)",
    )

    calib_labels = [int(r["label"]) for r in calib_rows]
    calib_scores_by_method = {
        "full_frames_mean": [float(r["scores"]["full_frames_mean"]) for r in calib_rows],
        "segment_mean_max": [float(r["scores"]["segment_mean_max"]) for r in calib_rows],
    }
    tuning_by_method: Dict[str, Dict[str, object]] = {}
    for method_key, method_scores in calib_scores_by_method.items():
        tuning_by_method[method_key] = find_best_threshold(
            labels=calib_labels,
            video_scores=method_scores,
            threshold_steps=args.threshold_steps,
        )

    # Pick method with best calibration rank key: accuracy, f1, balanced_accuracy.
    def rank_from_best(best_row: Dict[str, object]) -> Tuple[float, float, float]:
        return (
            float(best_row["accuracy"]),
            float(best_row["f1"]),
            float(best_row["balanced_accuracy"]),
        )

    best_method = max(
        tuning_by_method.keys(),
        key=lambda k: rank_from_best(tuning_by_method[k]["best"]),
    )
    best_thr = float(tuning_by_method[best_method]["best"]["threshold"])

    print("\n=== Calibration per-video scores ===")
    for row in calib_rows:
        lbl = "ANOM" if row["label"] == 1 else "NORM"
        print(
            f"  {row['video_id']:10s}  label={lbl}  "
            f"segments={row['num_segments']:2d}  "
            f"full_mean={row['scores']['full_frames_mean']:.6f}  "
            f"segment_mean_max={row['scores']['segment_mean_max']:.6f}"
        )
    print("\n=== Calibration winners by method ===")
    for method_key in ["full_frames_mean", "segment_mean_max"]:
        b = tuning_by_method[method_key]["best"]
        print(
            f"  {method_key:18s} thr={float(b['threshold']):.6f}  "
            f"acc={float(b['accuracy']):.4f}  f1={float(b['f1']):.4f}  "
            f"bal_acc={float(b['balanced_accuracy']):.4f}"
        )
    print(f"\nChosen method: {best_method}")
    print(f"Chosen threshold: {best_thr:.6f}")

    # 2) Full-set scoring with calibrated threshold
    full_rows = score_videos_with_both_methods(
        model=model,
        videos=all_videos,
        no_cache=args.no_cache,
        tau_high=args.tau_high,
        tau_low=args.tau_low,
        min_len=args.min_len,
        title="Full test-set evaluation",
    )

    full_eval_best = evaluate_method(full_rows, best_method, best_thr)
    metrics = full_eval_best["metrics"]

    report = {
        "config": {
            "num_anomaly": args.num_anomaly,
            "num_normal": args.num_normal,
            "seed": args.seed,
            "threshold_steps": args.threshold_steps,
            "tau_high": args.tau_high,
            "tau_low": args.tau_low,
            "min_len": args.min_len,
            "no_cache": bool(args.no_cache),
            "cache_dir": str(CACHE_DIR),
        },
        "calibration": {
            "videos": calib_rows,
            "tuning_by_method": tuning_by_method,
            "chosen_method": best_method,
            "chosen_threshold": best_thr,
        },
        "full_eval": {
            "chosen_method": best_method,
            "chosen_threshold": best_thr,
            "metrics": metrics,
            "per_video": full_eval_best["per_video"],
        },
    }

    report_path = OUT_DIR / "video_level_v2_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Final full-set metrics ===")
    print(
        f"Accuracy={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"BalancedAcc={metrics['balanced_accuracy']:.4f}"
    )
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
