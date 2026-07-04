#!/usr/bin/env python3
"""
Generate 3 LLM explanation variants for all SHANGHAI test videos:
1) frames_only
2) frames_plus_score
3) frames_score_heatmap
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEST_TXT = ROOT / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
FRAMES_DIR = ROOT / "SHANGHAI" / "SHANGHAI_Test" / "frames"
PIPELINE_OUTPUTS = ROOT / "pipeline" / "outputs"

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "llm_variants"
PER_VIDEO_DIR = RESULTS_DIR / "per_video"

VARIANTS = {
    "frames_only": {
        "use_scores": False,
        "use_heatmap": False,
    },
    "frames_plus_score": {
        "use_scores": True,
        "use_heatmap": False,
    },
    "frames_score_heatmap": {
        "use_scores": True,
        "use_heatmap": True,
    },
}

SYSTEM_PROMPT = """\
You are analyzing a surveillance video segment.

Decide if there is anomalous activity (relative to normal pedestrian walking), \
and explain briefly in 1-2 sentences.

If everything appears normal, clearly say there is no anomaly in the video.

Respond with ONLY JSON:
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


def get_all_test_videos():
    videos = []
    with open(TEST_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            video_id = parts[0].split("/")[-1]
            nframes = int(parts[1])
            is_anom = int(parts[2])
            videos.append((video_id, nframes, is_anom))
    return videos


def sample_indices(total_frames, k):
    if total_frames <= k:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, k, dtype=int).tolist()


def resolve_raw_frame_path(video_id, idx):
    frame_dir = FRAMES_DIR / video_id
    candidates = [
        frame_dir / f"{str(idx).zfill(3)}.jpg",
        frame_dir / f"{str(idx).zfill(4)}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_pipeline_segment(video_id):
    """Return first detected segment metadata, else None."""
    meta_path = PIPELINE_OUTPUTS / video_id / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    segments = meta.get("segments", [])
    if not segments:
        return None

    seg = segments[0]
    seg_idx = seg["segment_index"]
    seg_dir = PIPELINE_OUTPUTS / video_id / f"segment_{seg_idx}"
    seg_meta_path = seg_dir / "metadata.json"
    if not seg_meta_path.exists():
        return None

    with open(seg_meta_path) as f:
        seg_meta = json.load(f)

    return {
        "segment_index": seg_idx,
        "segment_dir": seg_dir,
        "segment_meta": seg_meta,
    }


def build_user_content(video_id, nframes, variant, k_fallback):
    cfg = VARIANTS[variant]
    use_scores = cfg["use_scores"]
    use_heatmap = cfg["use_heatmap"]

    seg = load_pipeline_segment(video_id)
    if seg is not None:
        seg_meta = seg["segment_meta"]
        seg_dir = seg["segment_dir"]
        sampled = seg_meta["sampled_frames"]

        score_list = ", ".join(
            f"frame {f['frame_idx']}={f['anomaly_score']:.2f}" for f in sampled
        )
        text = (
            f"Segment frames {seg_meta['start_frame']}-{seg_meta['end_frame']} "
            f"({seg_meta['duration_frames']} frames). "
            f"You are shown {len(sampled)} sampled frames."
        )
        if use_scores:
            text += f" Per-frame anomaly scores: [{score_list}]."
        content = [{"type": "text", "text": text}]

        loaded = 0
        sampled_indices = []
        for frame in sampled:
            idx = frame["frame_idx"]
            score = frame["anomaly_score"]
            sampled_indices.append(idx)
            prefix = f"frame_{str(idx).zfill(4)}"
            orig = seg_dir / f"{prefix}_original.png"
            over = seg_dir / f"{prefix}_overlay.png"

            if orig.exists():
                if use_scores:
                    content.append({
                        "type": "text",
                        "text": f"Frame {idx} (score {score:.2f}) - original:",
                    })
                else:
                    content.append({
                        "type": "text",
                        "text": f"Frame {idx} - original:",
                    })
                content.append(make_image_block(str(orig)))
                loaded += 1

            if use_heatmap and over.exists():
                content.append({
                    "type": "text",
                    "text": f"Frame {idx} - anomaly heatmap overlay:",
                })
                content.append(make_image_block(str(over)))

        return content, loaded, sampled_indices, True

    # Fallback: no pipeline segment, use uniform raw frames.
    indices = sample_indices(nframes, k_fallback)
    content = [{
        "type": "text",
        "text": (
            f"Here are {len(indices)} sampled frames from a {nframes}-frame video. "
            "Determine whether there is anomaly in the video."
        ),
    }]
    loaded = 0
    for idx in indices:
        raw = resolve_raw_frame_path(video_id, idx)
        if raw is None:
            continue
        if use_scores:
            content.append({
                "type": "text",
                "text": f"Frame {idx}/{nframes - 1} (score unavailable):",
            })
        else:
            content.append({
                "type": "text",
                "text": f"Frame {idx}/{nframes - 1}:",
            })
        content.append(make_image_block(str(raw)))
        loaded += 1
    return content, loaded, indices, False


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
    except Exception as exc:
        return {"anomaly_detected": None, "explanation": f"ERROR: {exc}"}


def run_video_variant(video_id, nframes, is_anom_gt, variant, k_fallback, force=False):
    out_file = PER_VIDEO_DIR / f"{video_id}_{variant}.json"
    if out_file.exists() and not force:
        with open(out_file) as f:
            return json.load(f)

    content, loaded, sampled_indices, used_pipeline_segment = build_user_content(
        video_id, nframes, variant, k_fallback
    )
    if loaded == 0:
        result = {
            "video_id": video_id,
            "variant": variant,
            "is_anomalous_gt": is_anom_gt,
            "total_frames": nframes,
            "frames_sampled": 0,
            "sampled_indices": sampled_indices,
            "used_pipeline_segment": used_pipeline_segment,
            "anomaly_detected": None,
            "explanation": "ERROR: no frames could be loaded",
        }
    else:
        response = query_gpt4o(content)
        result = {
            "video_id": video_id,
            "variant": variant,
            "is_anomalous_gt": is_anom_gt,
            "total_frames": nframes,
            "frames_sampled": loaded,
            "sampled_indices": sampled_indices,
            "used_pipeline_segment": used_pipeline_segment,
            "anomaly_detected": response.get("anomaly_detected"),
            "explanation": response.get("explanation", ""),
        }

    PER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    return result


def parse_variants(variants_arg):
    if variants_arg.strip().lower() == "all":
        return list(VARIANTS.keys())
    requested = [v.strip() for v in variants_arg.split(",") if v.strip()]
    bad = [v for v in requested if v not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variant(s): {bad}. Valid: {list(VARIANTS.keys())}")
    return requested


def main():
    parser = argparse.ArgumentParser(description="LLM behavior on all test videos (3 variants)")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant list or 'all': frames_only,frames_plus_score,frames_score_heatmap",
    )
    parser.add_argument("--k-fallback", type=int, default=8, help="Raw-frame fallback sample count")
    parser.add_argument("--sleep", type=float, default=0.3, help="Delay between API calls")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY first")
        sys.exit(1)

    try:
        variants = parse_variants(args.variants)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    videos = get_all_test_videos()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        print(f"\n=== Variant: {variant} ===")
        all_results = []
        for i, (video_id, nframes, is_anom_gt) in enumerate(videos, 1):
            result = run_video_variant(
                video_id, nframes, is_anom_gt, variant,
                k_fallback=args.k_fallback, force=args.force
            )
            all_results.append(result)
            detected = result.get("anomaly_detected")
            tag = "ANOM" if detected else "NORM" if detected is False else "ERR"
            explanation = result.get("explanation", "")
            print(
                f"[{i}/{len(videos)}] {video_id} [{tag}] "
                f"\"{explanation[:70]}{'...' if len(explanation) > 70 else ''}\""
            )
            time.sleep(args.sleep)

        summary_path = RESULTS_DIR / f"llm_{variant}_all_test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
