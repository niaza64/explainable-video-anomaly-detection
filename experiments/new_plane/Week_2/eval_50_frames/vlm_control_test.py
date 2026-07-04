"""
Control Test: Feed normal frames with HIGH fake MAE scores to check
if the VLM is genuinely reasoning or just agreeing with the score.

If FP rate jumps → VLM is suggestible → our result is an artifact.
If FP stays ~0  → MAE score genuinely helps reasoning → result is real.

Usage:
    export OPENAI_API_KEY="sk-..."
    python vlm_control_test.py
"""

import os
import json
import base64
import time
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVAL_DIR / "vlm_control_test_results.json"

PROMPT_HIGH_SCORE = """You are analyzing a surveillance camera frame for anomalies.

IMPORTANT CONTEXT: An anomaly detection model trained on thousands of normal frames 
from this exact camera location has assigned this frame an anomaly score of {mae_score:.3f}.
For reference, normal frames typically score below 0.25, and scores above 0.35 indicate 
statistical deviation from the learned normal patterns for this scene.

Given this information, look at the frame carefully and answer:
1. Describe what you see in 1-2 sentences.
2. What might be causing the elevated anomaly score? Is there anything anomalous or unusual?
3. Rate the anomaly severity from 0 to 10 (0 = completely normal, 10 = extremely anomalous).

Respond ONLY in this JSON format:
{{"description": "...", "is_anomalous": true/false, "anomaly_explanation": "..." or null, "anomaly_score": 0-10}}"""


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_vlm(prompt_text, image_path):
    from openai import OpenAI
    client = OpenAI()

    ext = Path(image_path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    b64 = encode_image_b64(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ],
            }],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"description": f"ERROR: {e}", "is_anomalous": None, "anomaly_explanation": None, "anomaly_score": None}


def get_normal_frames_from_baseline():
    with open(EVAL_DIR / "vlm_experiment_results.json") as f:
        baseline = json.load(f)
    return [r for r in baseline if r["ground_truth"] == "normal" and r.get("image_path")]


def main():
    print("=== CONTROL TEST: Normal frames with FAKE high MAE scores ===\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    normal_frames = get_normal_frames_from_baseline()
    print(f"Testing {len(normal_frames)} normal frames with fake high scores\n")

    fake_scores = [0.45, 0.50, 0.55, 0.60, 0.65]

    results = []
    for i, frame in enumerate(normal_frames):
        fake_score = fake_scores[i % len(fake_scores)]
        src = f"{frame['video']}/frame{frame['frame_idx']}"
        print(f"({i+1}/{len(normal_frames)}) {src}  fake_score={fake_score:.2f} ...", end=" ", flush=True)

        prompt = PROMPT_HIGH_SCORE.format(mae_score=fake_score)
        vlm = query_vlm(prompt, frame["image_path"])
        print(f"score={vlm.get('anomaly_score')}  anom={vlm.get('is_anomalous')}")

        results.append({
            "video": frame["video"],
            "scene": frame["scene"],
            "frame_idx": frame["frame_idx"],
            "ground_truth": "normal",
            "fake_mae_score": fake_score,
            "image_path": frame["image_path"],
            "vlm_description": vlm.get("description"),
            "vlm_is_anomalous": vlm.get("is_anomalous"),
            "vlm_anomaly_explanation": vlm.get("anomaly_explanation"),
            "vlm_anomaly_score": vlm.get("anomaly_score"),
        })
        time.sleep(0.3)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    valid = [r for r in results if r["vlm_is_anomalous"] is not None]
    fp = sum(1 for r in valid if r["vlm_is_anomalous"] is True)
    tn = sum(1 for r in valid if r["vlm_is_anomalous"] is False)
    errs = sum(1 for r in results if r["vlm_is_anomalous"] is None)

    fp_rate = fp / len(valid) if valid else 0

    avg_score = sum(r["vlm_anomaly_score"] for r in valid if r["vlm_anomaly_score"] is not None) / max(1, len(valid))

    print("\n" + "=" * 60)
    print("  CONTROL TEST RESULTS")
    print("=" * 60)
    print(f"  Normal frames tested:     {len(normal_frames)}")
    print(f"  Valid responses:          {len(valid)}")
    print(f"  False positives:          {fp}/{len(valid)}  ({fp_rate:.0%})")
    print(f"  True negatives:           {tn}/{len(valid)}")
    print(f"  Errors:                   {errs}")
    print(f"  Avg VLM anomaly score:    {avg_score:.1f}/10")
    print()

    print("  COMPARISON:")
    print(f"  Variant A (real low score 0.15):    0 FP / 18 valid = 0%")
    print(f"  Control  (fake high score 0.45-0.65): {fp} FP / {len(valid)} valid = {fp_rate:.0%}")
    print()

    if fp_rate > 0.3:
        print("  VERDICT: VLM IS SUGGESTIBLE. High fake scores cause false positives.")
        print("  The 35%->70% improvement may be confirmation bias, not real detection.")
    elif fp_rate > 0.1:
        print("  VERDICT: PARTIALLY SUGGESTIBLE. Some false positives from fake scores.")
        print("  The improvement is partly real, partly bias. Needs careful calibration.")
    else:
        print("  VERDICT: VLM IS ROBUST. Fake high scores don't cause false positives.")
        print("  The 35%->70% improvement is genuine — MAE score helps real reasoning.")

    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
