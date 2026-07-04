#!/usr/bin/env python3
"""
Judge generated explanations against human ground truth using GPT-4o-as-judge.

Compares each generated explanation to the human annotation and scores it on
multiple dimensions. Supports judging both with-heatmap and no-heatmap variants.

Usage:
    export OPENAI_API_KEY="sk-..."
    python judge_explanations.py --video 01_0162
    python judge_explanations.py --video 01_0162 --variant no_heatmap
    python judge_explanations.py --batch
    python judge_explanations.py --batch --variant no_heatmap
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

OUTPUT_BASE = Path(__file__).resolve().parent / "outputs"
ANNOTATIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "SHANGHAI" / "anomalous_videos" / "annotations.json"
)

JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI-generated \
explanation of an anomalous event in a surveillance video.

You will be given:
1. A HUMAN ground-truth explanation written by someone who watched the video.
2. An AI-GENERATED explanation produced by a vision-language model that only \
saw a few sampled frames and anomaly heatmaps.

Score the AI explanation on these 4 criteria (each 1-5):

- **correctness**: Does the AI identify the same anomaly as the human? \
(1 = completely wrong anomaly, 3 = partially correct, 5 = exact same anomaly)
- **specificity**: Does the AI mention specific details (objects, people, \
actions, clothing)? (1 = very vague, 5 = rich detail)
- **completeness**: Does the AI capture all aspects the human mentioned? \
(1 = misses everything, 5 = covers all key points)
- **fluency**: Is the AI explanation well-written and clear? \
(1 = incoherent, 5 = natural and clear)

Also provide a brief justification (1-2 sentences) for your scores.

Respond with ONLY a JSON object:
{"correctness": 1-5, "specificity": 1-5, "completeness": 1-5, "fluency": 1-5, "justification": "..."}\
"""


def load_annotations():
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    return {entry["video_id"]: entry["explanation"] for entry in data}


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
        return {"correctness": None, "specificity": None, "completeness": None,
                "fluency": None, "justification": f"ERROR: {e}"}


def judge_video(video_id, annotations, variant=""):
    suffix = f"_{variant}" if variant else ""
    expl_path = OUTPUT_BASE / video_id / f"explanations{suffix}.json"

    if not expl_path.exists():
        print(f"  SKIP {video_id}: no explanations{suffix}.json found")
        return None

    human = annotations.get(video_id, "")
    if not human:
        print(f"  SKIP {video_id}: no human annotation")
        return None

    with open(expl_path) as f:
        data = json.load(f)

    variant_label = variant if variant else "with_heatmap"
    print(f"  {video_id} [{variant_label}]:")

    results = {
        "video_id": video_id,
        "variant": variant_label,
        "human_explanation": human,
        "segments": [],
    }

    for seg in data["segments"]:
        ai_explanation = seg.get("explanation", "")
        if not ai_explanation or ai_explanation.startswith("ERROR"):
            continue

        print(f"    Segment {seg['segment_index']}: ", end="", flush=True)
        print(f"AI: \"{ai_explanation[:60]}...\"")
        print(f"    {'':19s}GT: \"{human[:60]}...\"")

        scores = query_judge(human, ai_explanation)

        print(f"    {'':19s}-> correctness={scores.get('correctness')} "
              f"specificity={scores.get('specificity')} "
              f"completeness={scores.get('completeness')} "
              f"fluency={scores.get('fluency')}")

        seg_result = {
            "segment_index": seg["segment_index"],
            "ai_explanation": ai_explanation,
            "scores": scores,
        }
        results["segments"].append(seg_result)
        time.sleep(0.5)

    # Save
    judge_path = OUTPUT_BASE / video_id / f"judge{suffix}.json"
    with open(judge_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(all_results):
    if not all_results:
        return

    total = {"correctness": [], "specificity": [], "completeness": [], "fluency": []}
    for r in all_results:
        for seg in r["segments"]:
            for key in total:
                val = seg["scores"].get(key)
                if val is not None:
                    total[key].append(val)

    print(f"\n{'='*60}")
    print(f"  JUDGE SUMMARY ({len(all_results)} videos)")
    print(f"{'='*60}")
    for key in total:
        vals = total[key]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {key:15s}: {avg:.2f}/5  (n={len(vals)})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Judge explanations against ground truth")
    parser.add_argument("--video", type=str, help="Single video ID")
    parser.add_argument("--batch", action="store_true", help="Judge all videos with explanations")
    parser.add_argument("--variant", type=str, default="",
                        help="Variant suffix: '' (default/with heatmap) or 'no_heatmap'")
    args = parser.parse_args()

    if not args.video and not args.batch:
        parser.error("Specify --video VIDEO_ID or --batch")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    annotations = load_annotations()
    print(f"Loaded {sum(1 for v in annotations.values() if v)} human annotations\n")

    if args.batch:
        video_dirs = sorted([
            d.name for d in OUTPUT_BASE.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])

        all_results = []
        for i, vid in enumerate(video_dirs):
            print(f"[{i+1}/{len(video_dirs)}]", end=" ")
            result = judge_video(vid, annotations, variant=args.variant)
            if result:
                all_results.append(result)

        suffix = f"_{args.variant}" if args.variant else ""
        summary_path = OUTPUT_BASE / f"judge_summary{suffix}.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_results, f, indent=2)

        print_summary(all_results)
        print(f"\nSaved to: {summary_path}")
    else:
        result = judge_video(args.video, annotations, variant=args.variant)
        if result:
            print_summary([result])


if __name__ == "__main__":
    main()
