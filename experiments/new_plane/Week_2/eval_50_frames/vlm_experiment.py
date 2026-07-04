"""
VLM Baseline Experiment: Send anomalous + normal ShanghaiTech frames to GPT-4o
and measure detection accuracy + collect explanations.

Usage:
    export OPENAI_API_KEY="sk-..."
    python vlm_experiment.py
"""

import os
import json
import base64
import random
import time
import sys
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
FRAMES_DIR = BASE / "SHANGHAI" / "SHANGHAI_Test" / "frames"
TEST_TXT = BASE / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
OUTPUT_PATH = EVAL_DIR / "vlm_experiment_results.json"

VLM_PROMPT = """You are analyzing a surveillance camera frame for anomalies.

Look at this frame carefully and answer:
1. Describe what you see in 1-2 sentences.
2. Is there anything anomalous or unusual happening? If so, describe what specifically is anomalous and why.
3. Rate the anomaly severity from 0 to 10 (0 = completely normal everyday scene, 10 = extremely anomalous/dangerous).

Respond ONLY in this JSON format:
{
  "description": "brief description of the scene",
  "is_anomalous": true or false,
  "anomaly_explanation": "what is anomalous and why" or null if normal,
  "anomaly_score": number from 0 to 10
}"""


def load_anomalous_frames(n=20):
    """Pick n anomalous frames from manifest, sorted by MAE score (highest first)."""
    with open(EVAL_DIR / "manifest.json") as f:
        manifest = json.load(f)

    sorted_by_score = sorted(manifest, key=lambda x: x["anomaly_score"], reverse=True)
    selected = sorted_by_score[:n]

    frames = []
    for entry in selected:
        img_path = EVAL_DIR / entry["folder"] / "original.png"
        if img_path.exists():
            frames.append({
                "image_path": str(img_path),
                "video": entry["video"],
                "scene": entry["scene"],
                "frame_idx": entry["frame_idx"],
                "mae_anomaly_score": entry["anomaly_score"],
                "ground_truth": "anomalous",
                "source_folder": entry["folder"],
            })
    return frames


def load_normal_frames(n=20):
    """Sample n normal frames from fully-normal test videos, spread across scenes."""
    normal_videos = []
    with open(TEST_TXT) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            video_path = parts[0]
            num_frames = int(parts[1])
            has_anomaly = int(parts[2])
            if has_anomaly == 0:
                video_name = video_path.split("/")[-1]
                scene = video_name.split("_")[0]
                normal_videos.append({
                    "video": video_name,
                    "scene": scene,
                    "num_frames": num_frames,
                })

    scenes = {}
    for v in normal_videos:
        scenes.setdefault(v["scene"], []).append(v)

    candidates = []
    for scene_id, videos in scenes.items():
        sampled = random.sample(videos, min(3, len(videos)))
        for v in sampled:
            frame_dir = FRAMES_DIR / v["video"]
            if not frame_dir.exists():
                continue
            all_frames = sorted(os.listdir(frame_dir))
            if not all_frames:
                continue
            mid = len(all_frames) // 2
            frame_file = all_frames[mid]
            candidates.append({
                "image_path": str(frame_dir / frame_file),
                "video": v["video"],
                "scene": scene_id,
                "frame_idx": mid,
                "mae_anomaly_score": None,
                "ground_truth": "normal",
                "source_folder": None,
            })

    random.shuffle(candidates)
    return candidates[:n]


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(image_path):
    from openai import OpenAI

    client = OpenAI()

    ext = Path(image_path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    b64 = encode_image_b64(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "description": f"ERROR: {e}",
            "is_anomalous": None,
            "anomaly_explanation": None,
            "anomaly_score": None,
        }


def main():
    print("=== VLM Baseline Experiment ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print("Loading anomalous frames...")
    anomalous = load_anomalous_frames(20)
    print(f"  Selected {len(anomalous)} anomalous frames")

    print("Loading normal frames...")
    normal = load_normal_frames(20)
    print(f"  Selected {len(normal)} normal frames")

    all_frames = anomalous + normal
    print(f"\nTotal: {len(all_frames)} frames to process\n")

    results = []
    for i, frame in enumerate(all_frames):
        label = frame["ground_truth"]
        tag = f"[{label.upper():>9s}]"
        src = frame.get("source_folder") or f"{frame['video']}/frame{frame['frame_idx']}"
        print(f"({i+1}/{len(all_frames)}) {tag} {src} ...", end=" ", flush=True)

        vlm = query_vlm(frame["image_path"])
        print(f"score={vlm.get('anomaly_score')}  anomalous={vlm.get('is_anomalous')}")

        results.append({
            "video": frame["video"],
            "scene": frame["scene"],
            "frame_idx": frame["frame_idx"],
            "ground_truth": frame["ground_truth"],
            "mae_anomaly_score": frame["mae_anomaly_score"],
            "source_folder": frame.get("source_folder"),
            "image_path": frame["image_path"],
            "vlm_description": vlm.get("description"),
            "vlm_is_anomalous": vlm.get("is_anomalous"),
            "vlm_anomaly_explanation": vlm.get("anomaly_explanation"),
            "vlm_anomaly_score": vlm.get("anomaly_score"),
        })

        time.sleep(0.5)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    anom_results = [r for r in results if r["ground_truth"] == "anomalous"]
    norm_results = [r for r in results if r["ground_truth"] == "normal"]

    tp = sum(1 for r in anom_results if r["vlm_is_anomalous"] is True)
    fn = sum(1 for r in anom_results if r["vlm_is_anomalous"] is False)
    tn = sum(1 for r in norm_results if r["vlm_is_anomalous"] is False)
    fp = sum(1 for r in norm_results if r["vlm_is_anomalous"] is True)
    errs = sum(1 for r in results if r["vlm_is_anomalous"] is None)

    total_valid = tp + fn + tn + fp
    accuracy = (tp + tn) / total_valid if total_valid > 0 else 0

    avg_anom_vlm = (
        sum(r["vlm_anomaly_score"] for r in anom_results if r["vlm_anomaly_score"] is not None)
        / max(1, sum(1 for r in anom_results if r["vlm_anomaly_score"] is not None))
    )
    avg_norm_vlm = (
        sum(r["vlm_anomaly_score"] for r in norm_results if r["vlm_anomaly_score"] is not None)
        / max(1, sum(1 for r in norm_results if r["vlm_anomaly_score"] is not None))
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Anomalous frames detected:  {tp}/{len(anom_results)}  (sensitivity)")
    print(f"Normal frames correct:      {tn}/{len(norm_results)}  (specificity)")
    print(f"False positives:            {fp}")
    print(f"False negatives:            {fn}")
    print(f"Errors:                     {errs}")
    print(f"Accuracy:                   {accuracy:.1%}")
    print(f"Avg VLM score (anomalous):  {avg_anom_vlm:.1f}/10")
    print(f"Avg VLM score (normal):     {avg_norm_vlm:.1f}/10")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
