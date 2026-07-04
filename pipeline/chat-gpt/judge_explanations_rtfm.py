#!/usr/bin/env python3
"""
Judge RTFM pipeline explanations against human ground truth using GPT-4o-as-judge.

Compares each generated explanation to the human annotation from annotations.json
and scores it on correctness, specificity, completeness, and fluency (1-5 each).

Usage:
    export OPENAI_API_KEY="sk-..."
    python judge_explanations_rtfm.py --video 01_0015
    python judge_explanations_rtfm.py --batch
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

OUTPUT_BASE = Path(__file__).resolve().parent / "rtfm_outputs"
ANNOTATIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "SHANGHAI" / "anomalous_videos" / "annotations.json"
)

JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI-generated \
explanation of an anomalous event in a surveillance video.

You will be given:
1. A HUMAN ground-truth explanation written by someone who watched the full video.
2. An AI-GENERATED explanation produced by a vision-language model that only \
saw a few sampled frames guided by anomaly scores from a weakly-supervised \
anomaly detection model.

Score the AI explanation on these 4 criteria (each 1-5):

- **correctness**: Does the AI identify the same anomaly as the human? \
(1 = completely wrong anomaly, 3 = partially correct / right category but \
wrong details, 5 = exact same anomaly identified)
- **specificity**: Does the AI mention specific details (objects, people, \
actions, clothing, location)? (1 = very vague "something unusual", \
3 = mentions general activity, 5 = rich detail matching human level)
- **completeness**: Does the AI capture all aspects the human mentioned? \
(1 = misses everything, 3 = captures main point but misses secondary details, \
5 = covers all key points)
- **fluency**: Is the AI explanation well-written, clear, and natural? \
(1 = incoherent, 3 = understandable but awkward, 5 = natural and clear)

Also provide a brief justification (1-2 sentences) for your scores.

Respond with ONLY a JSON object:
{"correctness": 1-5, "specificity": 1-5, "completeness": 1-5, "fluency": 1-5, "justification": "..."}\
"""


def load_annotations():
    """Load human annotations keyed by video_id."""
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    annotations = {}
    for entry in data:
        if "video_id" in entry:
            annotations[entry["video_id"]] = entry
    return annotations


def query_judge(human_explanation, ai_explanation):
    from openai import OpenAI
    client = OpenAI()

    user_msg = (
        f"HUMAN ground-truth explanation:\n\"{human_explanation}\"\n\n"
        f"AI-generated explanation:\n\"{ai_explanation}\""
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
    except Exception as e:
        print(f"    ERROR: {e}")
        return {
            "correctness": None, "specificity": None,
            "completeness": None, "fluency": None,
            "justification": f"ERROR: {e}",
        }


def judge_video(video_id, annotations):
    """Judge a single video's explanation against ground truth."""
    expl_path = OUTPUT_BASE / video_id / "explanation.json"

    if not expl_path.exists():
        print(f"  SKIP {video_id}: no explanation.json found")
        return None

    annotation = annotations.get(video_id)
    if not annotation:
        print(f"  SKIP {video_id}: no human annotation (may be normal video)")
        return None

    human_explanation = annotation["explanation"]

    with open(expl_path) as f:
        expl_data = json.load(f)

    ai_explanation = expl_data.get("explanation", "")
    if not ai_explanation or ai_explanation.startswith("ERROR"):
        print(f"  SKIP {video_id}: AI explanation is empty or errored")
        return None

    print(f"  {video_id}:")
    print(f"    AI : \"{ai_explanation[:90]}{'...' if len(ai_explanation) > 90 else ''}\"")
    print(f"    GT : \"{human_explanation[:90]}{'...' if len(human_explanation) > 90 else ''}\"")

    scores = query_judge(human_explanation, ai_explanation)

    print(f"    -> correctness={scores.get('correctness')} "
          f"specificity={scores.get('specificity')} "
          f"completeness={scores.get('completeness')} "
          f"fluency={scores.get('fluency')}")

    result = {
        "video_id": video_id,
        "human_explanation": human_explanation,
        "ai_explanation": ai_explanation,
        "gate_score": expl_data.get("gate_score"),
        "gt_anomaly_start": annotation.get("anomaly_start_frame"),
        "gt_anomaly_end": annotation.get("anomaly_end_frame"),
        "scores": scores,
    }

    judge_path = OUTPUT_BASE / video_id / "judge.json"
    with open(judge_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def print_summary(all_results):
    """Print aggregate statistics."""
    if not all_results:
        print("\nNo results to summarize.")
        return

    metrics = {"correctness": [], "specificity": [], "completeness": [], "fluency": []}
    for r in all_results:
        for key in metrics:
            val = r["scores"].get(key)
            if val is not None:
                metrics[key].append(val)

    print(f"\n{'='*65}")
    print(f"  JUDGE SUMMARY  ({len(all_results)} videos)")
    print(f"{'='*65}")
    print(f"  {'Metric':<15s} {'Mean':>6s} {'Std':>6s} {'Min':>5s} {'Max':>5s}  {'N':>3s}")
    print(f"  {'-'*50}")

    overall_scores = []
    for key in metrics:
        vals = metrics[key]
        if vals:
            arr = np.array(vals)
            print(f"  {key:<15s} {arr.mean():>6.2f} {arr.std():>6.2f} "
                  f"{arr.min():>5.1f} {arr.max():>5.1f}  {len(vals):>3d}")
            overall_scores.extend(vals)

    if overall_scores:
        overall = np.array(overall_scores)
        print(f"  {'-'*50}")
        print(f"  {'OVERALL':<15s} {overall.mean():>6.2f} {overall.std():>6.2f} "
              f"{overall.min():>5.1f} {overall.max():>5.1f}  {len(overall):>3d}")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(
        description="Judge RTFM explanations against ground truth"
    )
    parser.add_argument("--video", type=str, help="Single video ID")
    parser.add_argument("--batch", action="store_true",
                        help="Judge all videos with explanations")
    args = parser.parse_args()

    if not args.video and not args.batch:
        parser.error("Specify --video VIDEO_ID or --batch")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    annotations = load_annotations()
    print(f"Loaded {len(annotations)} human annotations\n")

    if args.batch:
        video_dirs = sorted([
            d.name for d in OUTPUT_BASE.iterdir()
            if d.is_dir() and (d / "explanation.json").exists()
        ])
        print(f"Batch mode: {len(video_dirs)} video(s) with explanations\n")

        all_results = []
        for i, vid in enumerate(video_dirs):
            print(f"[{i+1}/{len(video_dirs)}]", end=" ")
            result = judge_video(vid, annotations)
            if result:
                all_results.append(result)
            time.sleep(0.5)

        summary_path = OUTPUT_BASE / "judge_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_results, f, indent=2)

        print_summary(all_results)
        print(f"\nSaved to: {summary_path}")
    else:
        result = judge_video(args.video, annotations)
        if result:
            print_summary([result])


if __name__ == "__main__":
    main()
