#!/usr/bin/env python3
"""
Generate anomaly explanations using GPT-4o for RTFM pipeline outputs.

Reads the sampled frames and per-snippet anomaly scores produced by
run_rtfm_pipeline.py, sends them to GPT-4o with a structured prompt,
and saves the generated explanations.

Usage:
    export OPENAI_API_KEY="sk-..."
    python generate_explanations_rtfm.py --video 01_0015
    python generate_explanations_rtfm.py --batch
"""

import os
import sys
import json
import base64
import time
import argparse
from pathlib import Path

OUTPUT_BASE = Path(__file__).resolve().parent / "rtfm_outputs"


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_image_block(path, detail="high"):
    ext = Path(path).suffix.lower()
    media = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    b64 = encode_image_b64(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media};base64,{b64}", "detail": detail},
    }


SYSTEM_PROMPT = """\
You are a surveillance video anomaly analyst. You will be shown a set of \
frames sampled from a surveillance video that has been flagged as anomalous \
by a weakly-supervised anomaly detection model (RTFM).

The frames are ordered temporally. Each frame comes from a specific temporal \
snippet of the video, and you are given the anomaly score for that snippet \
(0 = normal, 1 = highly anomalous).

The frames were specifically selected from the anomalous portions of the \
video — they represent the onset, peak, and resolution of the detected \
anomaly.

Your task: Based on ALL the frames and their anomaly scores together, provide \
a single concise explanation (2-3 sentences) of what anomalous activity is \
happening. Focus on:
- WHAT is happening (the specific anomalous activity)
- WHO/WHAT is involved (people, vehicles, objects — describe appearance)
- WHEN in the sequence it starts and ends
- WHY it is anomalous (how it deviates from normal pedestrian behaviour)

Respond with ONLY a JSON object in this exact format:
{"explanation": "..."}\
"""


def build_user_content(video_dir, metadata):
    """Build multi-modal user message with sampled frames + scores."""
    video_dir = Path(video_dir)
    frames = metadata["extracted_frames"]
    segments = metadata.get("anomalous_segments", [])

    seg_desc = []
    for seg in segments:
        seg_desc.append(
            f"snippets {seg['start_snippet']}-{seg['end_snippet']}"
        )
    seg_str = ", ".join(seg_desc) if seg_desc else "unknown"

    score_list = ", ".join(
        f"snippet {f['snippet_idx']}={f['score']:.3f}" for f in frames
    )

    text = (
        f"This video has {metadata['n_segments']} temporal snippets (each ~16 frames). "
        f"The anomaly detection model flagged these contiguous segments as anomalous: [{seg_str}]. "
        f"Video-level anomaly gate score: {metadata['gate_score']:.3f}. "
        f"Below are {len(frames)} frames sampled from the anomalous segments, ordered temporally. "
        f"Per-snippet anomaly scores: [{score_list}]."
    )

    content = [{"type": "text", "text": text}]

    for f in frames:
        if f.get("file") is None:
            continue
        img_path = video_dir / f["file"]
        if not img_path.exists():
            continue

        content.append({
            "type": "text",
            "text": f"Frame from snippet {f['snippet_idx']} "
                    f"(frame #{f['frame_num']}, anomaly score: {f['score']:.3f}):",
        })
        content.append(make_image_block(str(img_path)))

    return content


def query_gpt4o(system_prompt, user_content):
    from openai import OpenAI
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"    ERROR: {e}")
        return {"explanation": f"ERROR: {e}"}


def process_video(video_id):
    video_dir = OUTPUT_BASE / video_id
    meta_path = video_dir / "metadata.json"

    if not meta_path.exists():
        print(f"  SKIP {video_id}: no pipeline output (run run_rtfm_pipeline.py first)")
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    frames = metadata.get("extracted_frames", [])
    valid_frames = [f for f in frames if f.get("file") is not None]

    if not valid_frames:
        print(f"  SKIP {video_id}: no extracted frames")
        return None

    print(f"  {video_id}: {len(valid_frames)} frames, "
          f"gate={metadata['gate_score']:.3f}, "
          f"{len(metadata.get('anomalous_segments', []))} segment(s) ...",
          end=" ", flush=True)

    user_content = build_user_content(video_dir, metadata)
    response = query_gpt4o(SYSTEM_PROMPT, user_content)

    explanation = response.get("explanation", "")
    print(f"-> \"{explanation[:80]}{'...' if len(explanation) > 80 else ''}\"")

    result = {
        "video_id": video_id,
        "gate_score": metadata["gate_score"],
        "n_segments": metadata["n_segments"],
        "anomalous_segments": metadata.get("anomalous_segments", []),
        "n_frames_sent": len(valid_frames),
        "explanation": explanation,
    }

    with open(video_dir / "explanation.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate anomaly explanations via GPT-4o for RTFM pipeline"
    )
    parser.add_argument("--video", type=str, help="Single video ID, e.g. 01_0015")
    parser.add_argument("--batch", action="store_true",
                        help="Process all videos with pipeline outputs")
    args = parser.parse_args()

    if not args.video and not args.batch:
        parser.error("Specify --video VIDEO_ID or --batch")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    if args.batch:
        video_dirs = sorted([
            d.name for d in OUTPUT_BASE.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])
        print(f"Batch mode: {len(video_dirs)} video(s) with pipeline outputs\n")

        all_results = []
        for i, vid in enumerate(video_dirs):
            print(f"[{i+1}/{len(video_dirs)}]", end=" ")
            result = process_video(vid)
            if result:
                all_results.append(result)
            time.sleep(0.5)

        summary_path = OUTPUT_BASE / "explanations_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDone. {len(all_results)} explanations saved to: {summary_path}")
    else:
        result = process_video(args.video)
        if result:
            print(f"\nDone. Saved to: {OUTPUT_BASE / args.video / 'explanation.json'}")


if __name__ == "__main__":
    main()
