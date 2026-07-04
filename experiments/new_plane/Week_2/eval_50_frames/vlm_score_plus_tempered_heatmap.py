"""
Variant D: Frame + MAE score + tempered heatmap.

The heatmap is presented as an imperfect spatial hint, not ground truth.
The VLM is told the heatmap is noisy and approximate — it MAY help
locate the anomaly but should not be relied upon as definitive.

Runs on the same 20 anomalous + 20 normal frames for comparison.

Results saved to: results_score_plus_tempered_heatmap.json

Usage:
    export OPENAI_API_KEY="sk-..."
    python vlm_score_plus_tempered_heatmap.py
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
OUTPUT_PATH = EVAL_DIR / "results_score_plus_tempered_heatmap.json"

PROMPT_ANOMALOUS = """You are analyzing a surveillance camera frame for anomalies.

CONTEXT: An anomaly detection model trained on thousands of normal frames from this 
exact camera has assigned this frame an anomaly score of {mae_score:.3f}. 
Normal frames typically score below 0.25; scores above 0.35 indicate statistical 
deviation from learned normal patterns.

You are also given a reconstruction-error heatmap. IMPORTANT: This heatmap is noisy 
and approximate — it is NOT perfectly accurate. It may sometimes highlight irrelevant 
regions (textures, shadows, edges) and miss the actual anomaly. Treat it only as a 
rough spatial hint that MAY help you locate the anomaly, not as definitive evidence. 
Focus primarily on the original frame and use the heatmap only if it points to 
something that genuinely looks unusual.

Answer:
1. Describe what you see in the original frame in 1-2 sentences.
2. Given the elevated anomaly score, what might be anomalous? Does the heatmap 
   point to anything that looks genuinely unusual?
3. Rate anomaly severity from 0 to 10.

Respond ONLY in this JSON format:
{{"description": "...", "is_anomalous": true/false, "anomaly_explanation": "..." or null, "anomaly_score": 0-10}}"""

PROMPT_NORMAL_SCORE_ONLY = """You are analyzing a surveillance camera frame for anomalies.

CONTEXT: An anomaly detection model trained on thousands of normal frames from this 
exact camera has assigned this frame an anomaly score of {mae_score:.3f}. 
Normal frames typically score below 0.25; scores above 0.35 indicate statistical 
deviation from learned normal patterns.

Given this information, look at the frame carefully and answer:
1. Describe what you see in 1-2 sentences.
2. Is there anything anomalous or unusual?
3. Rate anomaly severity from 0 to 10.

Respond ONLY in this JSON format:
{{"description": "...", "is_anomalous": true/false, "anomaly_explanation": "..." or null, "anomaly_score": 0-10}}"""


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_image_content(path):
    ext = Path(path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    b64 = encode_image_b64(path)
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}


def query_vlm(prompt_text, image_paths):
    from openai import OpenAI
    client = OpenAI()

    content = [{"type": "text", "text": prompt_text}]
    for p in image_paths:
        content.append(make_image_content(p))

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"description": f"ERROR: {e}", "is_anomalous": None,
                "anomaly_explanation": None, "anomaly_score": None}


def get_anomalous_frames():
    with open(EVAL_DIR / "manifest.json") as f:
        manifest = json.load(f)
    sorted_by_score = sorted(manifest, key=lambda x: x["anomaly_score"], reverse=True)[:20]
    frames = []
    for entry in sorted_by_score:
        folder = EVAL_DIR / entry["folder"]
        original = folder / "original.png"
        overlay = folder / "overlay.png"
        if original.exists():
            frames.append({
                "video": entry["video"],
                "scene": entry["scene"],
                "frame_idx": entry["frame_idx"],
                "mae_score": entry["anomaly_score"],
                "ground_truth": "anomalous",
                "source_folder": entry["folder"],
                "original_path": str(original),
                "overlay_path": str(overlay) if overlay.exists() else None,
            })
    return frames


def get_normal_frames():
    with open(EVAL_DIR / "results_frame_only.json") as f:
        baseline = json.load(f)
    frames = []
    for entry in baseline:
        if entry["ground_truth"] == "normal" and entry.get("image_path"):
            frames.append({
                "video": entry["video"],
                "scene": entry["scene"],
                "frame_idx": entry["frame_idx"],
                "mae_score": 0.15,
                "ground_truth": "normal",
                "source_folder": None,
                "original_path": entry["image_path"],
                "overlay_path": None,
            })
    return frames


def main():
    print("=== Variant D: Score + Tempered Heatmap ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    anomalous = get_anomalous_frames()
    normal = get_normal_frames()
    print(f"Frames: {len(anomalous)} anomalous, {len(normal)} normal\n")

    results = []

    # Anomalous frames: score + tempered heatmap
    print("--- Anomalous frames (score + tempered heatmap) ---")
    for i, f in enumerate(anomalous):
        src = f["source_folder"]
        print(f"  ({i+1}/{len(anomalous)}) {src}  mae={f['mae_score']:.3f} ...", end=" ", flush=True)

        prompt = PROMPT_ANOMALOUS.format(mae_score=f["mae_score"])
        images = [f["original_path"]]
        if f["overlay_path"]:
            images.append(f["overlay_path"])

        vlm = query_vlm(prompt, images)
        print(f"score={vlm.get('anomaly_score')}  anom={vlm.get('is_anomalous')}")

        results.append({
            "video": f["video"], "scene": f["scene"], "frame_idx": f["frame_idx"],
            "ground_truth": "anomalous", "mae_score": f["mae_score"],
            "source_folder": f["source_folder"],
            "variant": "D_score_plus_tempered_heatmap",
            "vlm_description": vlm.get("description"),
            "vlm_is_anomalous": vlm.get("is_anomalous"),
            "vlm_anomaly_explanation": vlm.get("anomaly_explanation"),
            "vlm_anomaly_score": vlm.get("anomaly_score"),
        })
        time.sleep(0.3)

    # Normal frames: score only (no heatmap available)
    print("\n--- Normal frames (score only, low score) ---")
    for i, f in enumerate(normal):
        src = f"{f['video']}/frame{f['frame_idx']}"
        print(f"  ({i+1}/{len(normal)}) {src}  mae={f['mae_score']:.2f} ...", end=" ", flush=True)

        prompt = PROMPT_NORMAL_SCORE_ONLY.format(mae_score=f["mae_score"])
        vlm = query_vlm(prompt, [f["original_path"]])
        print(f"score={vlm.get('anomaly_score')}  anom={vlm.get('is_anomalous')}")

        results.append({
            "video": f["video"], "scene": f["scene"], "frame_idx": f["frame_idx"],
            "ground_truth": "normal", "mae_score": f["mae_score"],
            "source_folder": None,
            "variant": "D_score_plus_tempered_heatmap",
            "vlm_description": vlm.get("description"),
            "vlm_is_anomalous": vlm.get("is_anomalous"),
            "vlm_anomaly_explanation": vlm.get("anomaly_explanation"),
            "vlm_anomaly_score": vlm.get("anomaly_score"),
        })
        time.sleep(0.3)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    anom = [r for r in results if r["ground_truth"] == "anomalous"]
    norm = [r for r in results if r["ground_truth"] == "normal"]

    tp = sum(1 for r in anom if r["vlm_is_anomalous"] is True)
    fn = sum(1 for r in anom if r["vlm_is_anomalous"] is False)
    tn = sum(1 for r in norm if r["vlm_is_anomalous"] is False)
    fp = sum(1 for r in norm if r["vlm_is_anomalous"] is True)
    errs_a = sum(1 for r in anom if r["vlm_is_anomalous"] is None)
    errs_n = sum(1 for r in norm if r["vlm_is_anomalous"] is None)

    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)

    avg_anom = sum(r["vlm_anomaly_score"] for r in anom if r["vlm_anomaly_score"] is not None) / max(1, sum(1 for r in anom if r["vlm_anomaly_score"] is not None))
    avg_norm = sum(r["vlm_anomaly_score"] for r in norm if r["vlm_anomaly_score"] is not None) / max(1, sum(1 for r in norm if r["vlm_anomaly_score"] is not None))

    print("\n" + "=" * 65)
    print("  VARIANT D: Score + Tempered Heatmap — RESULTS")
    print("=" * 65)
    print(f"  Anomalous detected:   {tp}/{len(anom)}  ({sens:.0%} sensitivity)  [errors: {errs_a}]")
    print(f"  Normal correct:       {tn}/{len(norm)}  ({spec:.0%} specificity)  [errors: {errs_n}]")
    print(f"  False positives:      {fp}")
    print(f"  False negatives:      {fn}")
    print(f"  Avg VLM score (anom): {avg_anom:.1f}/10")
    print(f"  Avg VLM score (norm): {avg_norm:.1f}/10")

    print(f"\n  --- COMPARISON ---")
    print(f"  {'Variant':<45} {'Sens':>6} {'Spec':>6} {'FP':>4}")
    print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*4}")
    print(f"  {'Baseline (frame only)':<45} {'35%':>6} {'100%':>6} {'0':>4}")
    print(f"  {'A: frame + MAE score':<45} {'70%':>6} {'100%':>6} {'0':>4}")
    print(f"  {'B: frame + heatmap':<45} {'~60%':>6} {'n/a':>6} {'n/a':>4}")
    print(f"  {'C: score + heatmap (no disclaimer)':<45} {'100%':>6} {'n/a':>6} {'n/a':>4}")
    print(f"  {'D: score + tempered heatmap (THIS)':<45} {f'{sens:.0%}':>6} {f'{spec:.0%}':>6} {fp:>4}")

    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
