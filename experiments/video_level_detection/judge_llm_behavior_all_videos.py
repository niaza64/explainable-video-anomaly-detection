#!/usr/bin/env python3
"""
Judge all 3 LLM explanation variants for all test videos.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_TXT = ROOT / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
ANNOTATIONS_PATH = ROOT / "SHANGHAI" / "anomalous_videos" / "annotations.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "llm_variants"
PER_VIDEO_DIR = RESULTS_DIR / "per_video"

VARIANTS = ["frames_only", "frames_plus_score", "frames_score_heatmap"]

JUDGE_PROMPT = """\
You are an impartial judge evaluating whether an AI-generated explanation \
matches human ground truth for surveillance footage.

You will get:
1) HUMAN explanation
2) AI explanation

Some videos are anomalous and some are normal.
- For normal videos, high correctness means AI clearly says no anomaly.
- For anomalous videos, high correctness means AI identifies the same anomaly.

Score 1-5:
- correctness
- specificity
- completeness
- fluency

Return ONLY JSON:
{"correctness":1-5,"specificity":1-5,"completeness":1-5,"fluency":1-5,"justification":"..."}\
"""


def load_annotations():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    return {entry["video_id"]: entry["explanation"] for entry in data}


def load_video_labels():
    labels = {}
    with open(TEST_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            video_id = parts[0].split("/")[-1]
            labels[video_id] = int(parts[2])
    return labels


def query_judge(human_explanation, ai_explanation):
    from openai import OpenAI

    client = OpenAI()
    user_msg = (
        f'HUMAN explanation:\n"{human_explanation}"\n\n'
        f'AI explanation:\n"{ai_explanation}"'
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        return {
            "correctness": None,
            "specificity": None,
            "completeness": None,
            "fluency": None,
            "justification": f"ERROR: {exc}",
        }


def parse_variants(variants_arg):
    if variants_arg.strip().lower() == "all":
        return list(VARIANTS)
    requested = [v.strip() for v in variants_arg.split(",") if v.strip()]
    bad = [v for v in requested if v not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variant(s): {bad}. Valid: {VARIANTS}")
    return requested


def load_llm_outputs_for_variant(variant):
    if not PER_VIDEO_DIR.exists():
        return []
    outputs = []
    for path in sorted(PER_VIDEO_DIR.glob(f"*_{variant}.json")):
        with open(path) as f:
            outputs.append(json.load(f))
    return outputs


def compute_metrics(rows):
    metrics = ["correctness", "specificity", "completeness", "fluency"]
    summary = {}
    for m in metrics:
        vals = [r["scores"].get(m) for r in rows if r["scores"].get(m) is not None]
        summary[m] = {
            "mean": round(sum(vals) / len(vals), 3) if vals else None,
            "n": len(vals),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Judge LLM behavior on all test videos (3 variants)")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant list or 'all': frames_only,frames_plus_score,frames_score_heatmap",
    )
    parser.add_argument("--sleep", type=float, default=0.3, help="Delay between judge calls")
    parser.add_argument("--force", action="store_true", help="Re-judge even if score exists")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY first")
        sys.exit(1)

    try:
        variants = parse_variants(args.variants)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    annotations = load_annotations()
    labels = load_video_labels()
    for vid, is_anom in labels.items():
        if is_anom == 0 and vid not in annotations:
            annotations[vid] = "No anomaly in the video."

    all_variant_metrics = {}
    for variant in variants:
        llm_outputs = load_llm_outputs_for_variant(variant)
        if not llm_outputs:
            print(f"\n[{variant}] No outputs found. Run generator first.")
            continue

        scored_path = RESULTS_DIR / f"judged_{variant}.json"
        existing = {}
        if scored_path.exists() and not args.force:
            with open(scored_path) as f:
                prior = json.load(f)
            existing = {r["video_id"]: r for r in prior}

        print(f"\n=== Judging variant: {variant} ===")
        rows = []
        for i, item in enumerate(llm_outputs, 1):
            video_id = item["video_id"]
            if video_id in existing:
                rows.append(existing[video_id])
                print(f"[{i}/{len(llm_outputs)}] {video_id} [cached]")
                continue

            human = annotations.get(video_id, "No anomaly in the video.")
            ai_explanation = item.get("explanation", "")
            scores = query_judge(human, ai_explanation)
            row = {
                "video_id": video_id,
                "variant": variant,
                "is_anomalous_gt": item.get("is_anomalous_gt"),
                "human_explanation": human,
                "ai_explanation": ai_explanation,
                "anomaly_detected": item.get("anomaly_detected"),
                "scores": scores,
            }
            rows.append(row)
            print(
                f"[{i}/{len(llm_outputs)}] {video_id} "
                f"correctness={scores.get('correctness')} "
                f"specificity={scores.get('specificity')} "
                f"completeness={scores.get('completeness')} "
                f"fluency={scores.get('fluency')}"
            )
            time.sleep(args.sleep)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(scored_path, "w") as f:
            json.dump(rows, f, indent=2)

        metrics = compute_metrics(rows)
        metrics_path = RESULTS_DIR / f"judged_metrics_{variant}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        all_variant_metrics[variant] = metrics
        print(f"Saved judged results: {scored_path}")
        print(f"Saved metrics: {metrics_path}")

    if all_variant_metrics:
        combined_path = RESULTS_DIR / "judged_metrics_all_variants.json"
        with open(combined_path, "w") as f:
            json.dump(all_variant_metrics, f, indent=2)
        print(f"\nSaved combined metrics: {combined_path}")


if __name__ == "__main__":
    main()
