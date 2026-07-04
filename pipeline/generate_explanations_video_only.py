#!/usr/bin/env python3
"""
Variant 3: Video-only baseline — no AED-MAE scores, no heatmaps.

Samples K frames uniformly from the ENTIRE video and asks GPT-4o:
"Is there any anomaly? If yes, describe it."

This ablation tests whether the AED-MAE pipeline (temporal localization,
heatmaps, scores) actually helps, or if GPT-4o can detect anomalies
purely from raw surveillance frames.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_explanations_video_only.py --video 01_0015
    python generate_explanations_video_only.py --batch
    python generate_explanations_video_only.py --batch --K 10
"""

import os
import sys
import json
import base64
import time
import argparse
import numpy as np
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE = PIPELINE_DIR / "outputs"
FRAMES_DIR = PIPELINE_DIR.parent / "SHANGHAI" / "SHANGHAI_Test" / "frames"
TEST_TXT = PIPELINE_DIR.parent / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"

SYSTEM_PROMPT = """\
You are analyzing a surveillance video. You will be shown a set of frames \
sampled uniformly from a surveillance camera recording.

Your task: Examine all the frames carefully. Is there any anomalous or \
unusual activity visible? Normal activity for these scenes is pedestrians \
walking calmly on walkways and paths.

If you detect an anomaly, provide a concise explanation (1-2 sentences) of \
WHAT is unusual and WHY it deviates from normal pedestrian behaviour.

If everything appears normal, say so.

Respond with ONLY a JSON object in this exact format:
{"anomaly_detected": true/false, "explanation": "..."}\
"""


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_image_block(path, detail="high"):
    ext = Path(path).suffix.lower()
    media = "image/png" if ext == ".png" else "image/jpeg"
    b64 = encode_image_b64(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media};base64,{b64}", "detail": detail},
    }


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


def sample_frame_indices(total_frames, K=8):
    if total_frames <= K:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, K, dtype=int).tolist()


def query_gpt4o(user_content):
    from openai import OpenAI
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=250,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"anomaly_detected": None, "explanation": f"ERROR: {e}"}


def process_video(video_id, nframes, K=8):
    frames_dir = FRAMES_DIR / video_id
    if not frames_dir.exists():
        print(f"  SKIP {video_id}: frames directory not found")
        return None

    out_dir = OUTPUT_BASE / video_id
    out_file = out_dir / "explanations_video_only.json"
    if out_file.exists():
        print(f"  SKIP {video_id}: already done")
        with open(out_file) as f:
            return json.load(f)

    indices = sample_frame_indices(nframes, K)

    text = (
        f"Here are {len(indices)} frames sampled uniformly from a {nframes}-frame "
        f"surveillance video. Examine them carefully for any anomalous activity."
    )
    content = [{"type": "text", "text": text}]

    loaded = 0
    for idx in indices:
        fpath = frames_dir / f"{str(idx).zfill(3)}.jpg"
        if not fpath.exists():
            continue
        content.append({
            "type": "text",
            "text": f"Frame {idx}/{nframes - 1}:"
        })
        content.append(make_image_block(str(fpath)))
        loaded += 1

    if loaded == 0:
        print(f"  SKIP {video_id}: no frames could be loaded")
        return None

    print(f"  {video_id}: {loaded} frames from {nframes} total ... ", end="", flush=True)

    response = query_gpt4o(content)
    explanation = response.get("explanation", "")
    detected = response.get("anomaly_detected", None)

    tag = "ANOM" if detected else "NORM" if detected is False else "ERR"
    print(f"[{tag}] \"{explanation[:70]}{'...' if len(explanation) > 70 else ''}\"")

    result = {
        "video_id": video_id,
        "variant": "video_only",
        "total_frames": nframes,
        "frames_sampled": loaded,
        "sampled_indices": indices,
        "anomaly_detected": detected,
        "segments": [{
            "segment_index": 0,
            "start_frame": 0,
            "end_frame": nframes - 1,
            "include_heatmap": False,
            "explanation": explanation,
        }],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Video-only baseline: GPT-4o without AED-MAE")
    parser.add_argument("--video", type=str, help="Single video ID")
    parser.add_argument("--batch", action="store_true", help="Process all anomalous videos")
    parser.add_argument("--K", type=int, default=8, help="Frames to sample per video (default 8)")
    args = parser.parse_args()

    if not args.video and not args.batch:
        parser.error("Specify --video VIDEO_ID or --batch")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    videos = get_anomalous_videos()
    vid_map = {v: n for v, n in videos}

    if args.batch:
        print(f"Video-only baseline: {len(videos)} anomalous videos, K={args.K}\n")
        all_results = []
        for i, (vid, nframes) in enumerate(videos):
            print(f"[{i+1}/{len(videos)}]", end=" ")
            result = process_video(vid, nframes, K=args.K)
            if result:
                all_results.append(result)
            time.sleep(0.3)

        summary_path = OUTPUT_BASE / "explanations_summary_video_only.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_results, f, indent=2)

        detected = sum(1 for r in all_results if r.get("anomaly_detected"))
        print(f"\nDone. {detected}/{len(all_results)} detected anomalies.")
        print(f"Saved to: {summary_path}")
    else:
        nframes = vid_map.get(args.video)
        if not nframes:
            print(f"ERROR: {args.video} not found in anomalous video list")
            sys.exit(1)
        result = process_video(args.video, nframes, K=args.K)
        if result:
            print(f"\nDone. Saved to: {OUTPUT_BASE / args.video / 'explanations_video_only.json'}")


if __name__ == "__main__":
    main()
