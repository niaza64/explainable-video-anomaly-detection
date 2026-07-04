#!/usr/bin/env python3
"""
Full evaluation pipeline for the paper.

Orchestrates all three stages and compiles results:
  Stage 1: run_pipeline.py   --batch  (AED-MAE scoring + segment detection + frame sampling)
  Stage 2: generate_explanations.py   (GPT-4o explanations: with heatmap + without heatmap)
  Stage 3: judge_explanations.py      (GPT-4o-as-judge scoring against human annotations)
  Stage 4: Compile results into paper-ready tables

Usage:
    export OPENAI_API_KEY="sk-..."
    python run_full_evaluation.py                    # run everything
    python run_full_evaluation.py --skip-pipeline    # skip Stage 1 (already ran)
    python run_full_evaluation.py --stage 2          # run only Stage 2+
    python run_full_evaluation.py --stage 4          # compile results only
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = PIPELINE_DIR / "outputs"
RESULTS_DIR = PIPELINE_DIR / "results"
ANNOTATIONS_PATH = PIPELINE_DIR.parent / "SHANGHAI" / "anomalous_videos" / "annotations.json"
TEST_TXT = PIPELINE_DIR.parent / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"

PYTHON = sys.executable
VARIANTS = ["with_heatmap", "no_heatmap"]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_annotations():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    return {e["video_id"]: e["explanation"] for e in data if e["explanation"]}


def load_annotations_full():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    return {e["video_id"]: e for e in data if e["explanation"]}


def get_anomalous_videos():
    videos = []
    with open(str(TEST_TXT)) as f:
        for line in f:
            parts = line.strip().split()
            vid = parts[0].split("/")[-1]
            nframes = int(parts[1])
            is_anom = int(parts[2])
            if is_anom == 1:
                videos.append((vid, nframes))
    return videos


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: AED-MAE pipeline
# ──────────────────────────────────────────────────────────────────────────────

def stage1_pipeline(K=5):
    log("=" * 60)
    log("STAGE 1: AED-MAE Anomaly Detection Pipeline")
    log("=" * 60)

    already_done = [
        d.name for d in OUTPUT_BASE.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ] if OUTPUT_BASE.exists() else []

    videos = get_anomalous_videos()
    todo = [(v, n) for v, n in videos if v not in already_done]

    log(f"Total anomalous videos: {len(videos)}")
    log(f"Already processed: {len(already_done)}")
    log(f"Remaining: {len(todo)}")

    if not todo:
        log("All videos already processed. Skipping Stage 1.")
        return

    for i, (vid, nframes) in enumerate(todo):
        log(f"  [{i+1}/{len(todo)}] Processing {vid} ({nframes} frames) ...")
        cmd = [
            PYTHON, str(PIPELINE_DIR / "run_pipeline.py"),
            "--video", vid, "--K", str(K),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"    ERROR: {result.stderr[-200:]}")
        else:
            log(f"    Done.")

    log("Stage 1 complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: GPT-4o explanations
# ──────────────────────────────────────────────────────────────────────────────

def stage2_explanations():
    log("=" * 60)
    log("STAGE 2: GPT-4o Explanation Generation")
    log("=" * 60)

    from generate_explanations import process_video

    video_dirs = sorted([
        d.name for d in OUTPUT_BASE.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    for variant_flag, variant_name in [(True, "with_heatmap"), (False, "no_heatmap")]:
        suffix = "" if variant_flag else "_no_heatmap"
        log(f"\n--- Variant: {variant_name} ---")

        for i, vid in enumerate(video_dirs):
            out_file = OUTPUT_BASE / vid / f"explanations{suffix}.json"
            if out_file.exists():
                log(f"  [{i+1}/{len(video_dirs)}] {vid} [{variant_name}] already done, skipping.")
                continue

            log(f"  [{i+1}/{len(video_dirs)}] {vid} [{variant_name}] ...")
            try:
                process_video(vid, include_heatmap=variant_flag)
            except Exception as e:
                log(f"    ERROR: {e}")
            time.sleep(0.3)

    log("Stage 2 complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: GPT-4o-as-judge
# ──────────────────────────────────────────────────────────────────────────────

def stage3_judging():
    log("=" * 60)
    log("STAGE 3: GPT-4o-as-Judge Evaluation")
    log("=" * 60)

    from judge_explanations import judge_video

    annotations = load_annotations()
    log(f"Human annotations loaded: {len(annotations)}")

    video_dirs = sorted([
        d.name for d in OUTPUT_BASE.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    for variant_suffix, variant_name in [("", "with_heatmap"), ("no_heatmap", "no_heatmap")]:
        log(f"\n--- Judging variant: {variant_name} ---")

        judge_suffix = f"_{variant_suffix}" if variant_suffix else ""
        all_results = []

        for i, vid in enumerate(video_dirs):
            judge_file = OUTPUT_BASE / vid / f"judge{judge_suffix}.json"
            if judge_file.exists():
                with open(judge_file) as f:
                    all_results.append(json.load(f))
                log(f"  [{i+1}/{len(video_dirs)}] {vid} [{variant_name}] already judged, loading.")
                continue

            log(f"  [{i+1}/{len(video_dirs)}] {vid} [{variant_name}] ...")
            try:
                result = judge_video(vid, annotations, variant=variant_suffix)
                if result:
                    all_results.append(result)
            except Exception as e:
                log(f"    ERROR: {e}")
            time.sleep(0.3)

        summary_path = OUTPUT_BASE / f"judge_summary{judge_suffix}.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_results, f, indent=2)

    log("Stage 3 complete.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Compile paper-ready results
# ──────────────────────────────────────────────────────────────────────────────

def stage4_compile():
    log("=" * 60)
    log("STAGE 4: Compiling Paper-Ready Results")
    log("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    annotations_full = load_annotations_full()

    # ── Collect all judge results ──
    variant_data = {}
    for variant_suffix, variant_name in [
        ("", "with_heatmap"),
        ("_no_heatmap", "no_heatmap"),
        ("_video_only", "video_only"),
    ]:
        summary_path = OUTPUT_BASE / f"judge_summary{variant_suffix}.json"
        if not summary_path.exists():
            log(f"  WARNING: {summary_path} not found, skipping {variant_name}")
            continue
        with open(summary_path) as f:
            variant_data[variant_name] = json.load(f)

    if not variant_data:
        log("  No judge results found. Run stages 1-3 first.")
        return

    # ── Table 1: Aggregate scores per variant ──
    table1 = {}
    metrics = ["correctness", "specificity", "completeness", "fluency"]
    for variant_name, results in variant_data.items():
        scores = {m: [] for m in metrics}
        for r in results:
            for seg in r["segments"]:
                for m in metrics:
                    val = seg["scores"].get(m)
                    if val is not None:
                        scores[m].append(val)
        table1[variant_name] = {
            m: {
                "mean": round(sum(scores[m]) / len(scores[m]), 2) if scores[m] else None,
                "n": len(scores[m]),
            }
            for m in metrics
        }

    # ── Table 2: Per-video breakdown ──
    table2 = []
    processed_vids = set()
    for variant_name, results in variant_data.items():
        for r in results:
            processed_vids.add(r["video_id"])

    for vid in sorted(processed_vids):
        row = {
            "video_id": vid,
            "scene": vid.split("_")[0],
            "human_explanation": annotations_full.get(vid, {}).get("explanation", ""),
        }
        for variant_name, results in variant_data.items():
            match = [r for r in results if r["video_id"] == vid]
            if match and match[0]["segments"]:
                seg = match[0]["segments"][0]
                row[f"{variant_name}_explanation"] = seg.get("ai_explanation", "")
                for m in metrics:
                    row[f"{variant_name}_{m}"] = seg["scores"].get(m)
            else:
                row[f"{variant_name}_explanation"] = ""
                for m in metrics:
                    row[f"{variant_name}_{m}"] = None
        table2.append(row)

    # ── Table 3: Per-scene aggregation ──
    scenes = {}
    for row in table2:
        scene = row["scene"]
        if scene not in scenes:
            scenes[scene] = {v: {m: [] for m in metrics} for v in variant_data}
        for variant_name in variant_data:
            for m in metrics:
                val = row.get(f"{variant_name}_{m}")
                if val is not None:
                    scenes[scene][variant_name][m].append(val)

    table3 = []
    for scene in sorted(scenes):
        row = {"scene": scene}
        for variant_name in variant_data:
            for m in metrics:
                vals = scenes[scene][variant_name][m]
                row[f"{variant_name}_{m}"] = round(sum(vals) / len(vals), 2) if vals else None
            row[f"{variant_name}_n"] = len(scenes[scene][variant_name]["correctness"])
        table3.append(row)

    # ── Table 4: Segment detection accuracy ──
    table4 = []
    for vid in sorted(processed_vids):
        meta_path = OUTPUT_BASE / vid / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        ann = annotations_full.get(vid, {})
        gt_start = ann.get("anomaly_start_frame", None)
        gt_end = ann.get("anomaly_end_frame", None)

        detected_segments = meta.get("segments", [])
        if detected_segments:
            det_start = detected_segments[0]["start_frame"]
            det_end = detected_segments[0]["end_frame"]
        else:
            det_start = det_end = None

        overlap = 0
        if gt_start is not None and det_start is not None:
            o_start = max(gt_start, det_start)
            o_end = min(gt_end, det_end)
            if o_end > o_start:
                overlap = o_end - o_start
            gt_len = gt_end - gt_start
            det_len = det_end - det_start
            iou = overlap / (gt_len + det_len - overlap) if (gt_len + det_len - overlap) > 0 else 0
        else:
            iou = 0

        table4.append({
            "video_id": vid,
            "gt_start": gt_start,
            "gt_end": gt_end,
            "detected_start": det_start,
            "detected_end": det_end,
            "num_segments": meta["num_segments_detected"],
            "temporal_iou": round(iou, 3),
        })

    # ── Save everything ──
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_videos": len(processed_vids),
            "num_annotated": len(annotations_full),
            "variants": list(variant_data.keys()),
        },
        "table1_aggregate_scores": table1,
        "table2_per_video": table2,
        "table3_per_scene": table3,
        "table4_temporal_localization": table4,
    }

    with open(RESULTS_DIR / "full_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # ── Print paper-ready summary ──
    print("\n")
    print("=" * 70)
    print("  TABLE 1: Explanation Quality — Aggregate Scores (GPT-4o-as-Judge)")
    print("=" * 70)
    header = f"{'Variant':<20s}"
    for m in metrics:
        header += f"  {m:>13s}"
    header += f"  {'n':>5s}"
    print(header)
    print("-" * 70)
    for variant_name, scores in table1.items():
        row = f"{variant_name:<20s}"
        for m in metrics:
            val = scores[m]["mean"]
            row += f"  {val:>13.2f}" if val else f"  {'N/A':>13s}"
        row += f"  {scores[metrics[0]]['n']:>5d}"
        print(row)
    print("-" * 70)

    # ── Temporal localization summary ──
    ious = [r["temporal_iou"] for r in table4 if r["temporal_iou"] > 0]
    detected = sum(1 for r in table4 if r["num_segments"] > 0)
    print(f"\n  Temporal Localization (AED-MAE):")
    print(f"    Videos with detected segments: {detected}/{len(table4)}")
    if ious:
        print(f"    Mean Temporal IoU: {sum(ious)/len(ious):.3f}")

    # ── Video-only anomaly detection rate ──
    vo_summary_path = OUTPUT_BASE / "explanations_summary_video_only.json"
    if vo_summary_path.exists():
        with open(vo_summary_path) as f:
            vo_data = json.load(f)
        vo_detected = sum(1 for r in vo_data if r.get("anomaly_detected"))
        print(f"\n  Video-Only Baseline (GPT-4o raw):")
        print(f"    Anomaly detection rate: {vo_detected}/{len(vo_data)} ({100*vo_detected/len(vo_data):.1f}%)")
        full_results["video_only_detection_rate"] = {
            "detected": vo_detected, "total": len(vo_data),
            "rate": round(vo_detected / len(vo_data), 3),
        }
    print()

    # ── Per-scene breakdown ──
    print("=" * 70)
    print("  TABLE 3: Per-Scene Breakdown (correctness)")
    print("=" * 70)
    header = f"{'Scene':<8s}  {'n':>3s}"
    for v in variant_data:
        header += f"  {v:>15s}"
    print(header)
    print("-" * 70)
    for row in table3:
        line = f"{row['scene']:<8s}"
        first_v = list(variant_data.keys())[0]
        line += f"  {row.get(f'{first_v}_n', 0):>3d}"
        for v in variant_data:
            val = row.get(f"{v}_correctness")
            line += f"  {val:>15.2f}" if val else f"  {'N/A':>15s}"
        print(line)
    print("-" * 70)

    log(f"\nAll results saved to: {RESULTS_DIR / 'full_results.json'}")
    log("Stage 4 complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full evaluation for paper")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip Stage 1 (AED-MAE), use existing outputs")
    parser.add_argument("--stage", type=int, default=1,
                        help="Start from this stage (1-4)")
    parser.add_argument("--K", type=int, default=5,
                        help="Frames to sample per segment")
    args = parser.parse_args()

    if args.stage <= 2 and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY for stages 2-3.")
        sys.exit(1)

    start = time.time()

    if args.stage <= 1 and not args.skip_pipeline:
        stage1_pipeline(K=args.K)

    if args.stage <= 2:
        stage2_explanations()

    if args.stage <= 3:
        stage3_judging()

    if args.stage <= 4:
        stage4_compile()

    elapsed = time.time() - start
    log(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
