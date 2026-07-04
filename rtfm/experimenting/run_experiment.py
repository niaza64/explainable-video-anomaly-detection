#!/usr/bin/env python3
"""
RTFM-based Anomaly Explanation Pipeline

For each of the 44 annotated anomalous test videos:
  1. Load pre-extracted I3D features
  2. Run RTFM forward pass → per-snippet anomaly scores
  3. If max score > threshold → video is anomalous
  4. Take top-K snippets by score
  5. Sample 1 frame per snippet (middle frame of the 16-frame window)
  6. Send frames + scores to GPT-4o for explanation
  7. Judge explanation against human ground truth
  8. Save all results

Supports two configs:
  - Config A: No guidance — uniform 8 frames from full video, no scores, GPT-4o
  - Config D: RTFM guidance — top-5 snippet frames + scores, GPT-4o

Usage:
    export OPENAI_API_KEY="sk-..."
    python run_experiment.py --config D --threshold 0.5
    python run_experiment.py --config A
    python run_experiment.py --config all
"""

import os
import sys
import json
import time
import base64
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
RTFM_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RTFM_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "SHANGHAI"
ANNOTATIONS_PATH = DATA_DIR / "anomalous_videos" / "annotations.json"
FRAMES_DIR = DATA_DIR / "SHANGHAI_Test" / "frames"
TEST_FEAT_DIR = RTFM_DIR / "data" / "SH_Test_ten_crop_i3d"
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(RTFM_DIR))

# ─── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_GUIDED = """\
You are a surveillance video anomaly analyst. You will be shown frames \
from a video segment flagged as anomalous by an automated detector. Each \
frame has an associated anomaly score where higher values indicate greater \
deviation from normal patterns. Provide a concise explanation of the \
anomalous activity suitable for a surveillance operator.

Respond with ONLY a JSON object: {"explanation": "..."}\
"""

SYSTEM_PROMPT_UNGUIDED = """\
You are analyzing a surveillance video. You will be shown a set of frames \
sampled uniformly from a surveillance camera recording.

Your task: Examine all the frames carefully. Is there any anomalous or \
unusual activity visible? Normal activity for these scenes is pedestrians \
walking calmly on walkways and paths.

If you detect an anomaly, provide a concise explanation (1-2 sentences) of \
WHAT is unusual and WHY it deviates from normal pedestrian behaviour.

If everything appears normal, say so.

Respond with ONLY a JSON object: {"anomaly_detected": true/false, "explanation": "..."}\
"""

JUDGE_PROMPT = """\
You are an impartial judge evaluating the quality of an AI-generated \
explanation of an anomalous event in a surveillance video.

You will be given:
1. A HUMAN ground-truth explanation written by someone who watched the video.
2. An AI-GENERATED explanation produced by a vision-language model that only \
saw a few sampled frames.

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


# ─── RTFM Model ──────────────────────────────────────────────────────────────

def load_rtfm_model(checkpoint_path, device):
    import torch
    from model import Model

    model = Model(2048, batch_size=1)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def rtfm_score_video(model, feat_path, device):
    """
    Run RTFM on a single video's I3D features.
    Returns per-snippet anomaly scores (T,).
    """
    import torch
    from utils import process_feat

    features = np.load(feat_path, allow_pickle=True)
    features = np.array(features, dtype=np.float32)  # (10, T_raw, 2048)

    num_snippets_raw = features.shape[1]

    processed = []
    for crop in features:
        processed.append(process_feat(crop, 32))
    processed = np.array(processed, dtype=np.float32)  # (10, 32, 2048)

    inputs = torch.from_numpy(processed).unsqueeze(0).to(device)  # (1, 10, 32, 2048)
    inputs = inputs.permute(0, 2, 1, 3)  # (1, 32, 10, 2048) — what RTFM test expects

    with torch.no_grad():
        _, _, _, _, _, _, logits, _, _, _ = model(inputs=inputs)
        logits = torch.squeeze(logits, 1)
        logits = torch.mean(logits, 0)  # (32,)
        scores = logits.cpu().numpy()

    return scores, num_snippets_raw


# ─── Frame Sampling ──────────────────────────────────────────────────────────

def get_top_snippet_frames(scores, num_snippets_raw, total_frames, top_k=5):
    """
    Config D: Get frames from top-K anomalous snippets.
    Returns list of (frame_idx, snippet_score) tuples.
    """
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_indices = sorted(top_indices)

    scale = num_snippets_raw / 32.0
    frames_and_scores = []
    for idx in top_indices:
        raw_idx = int(idx * scale)
        frame_idx = min(raw_idx * 16 + 8, total_frames - 1)
        frames_and_scores.append((frame_idx, float(scores[idx])))

    return frames_and_scores


def get_uniform_frames(total_frames, num_frames=8):
    """
    Config A: Uniform sampling across full video.
    Returns list of (frame_idx, None) tuples.
    """
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return [(int(idx), None) for idx in indices]


# ─── Image Encoding ──────────────────────────────────────────────────────────

def encode_frame(frame_dir, frame_idx):
    for ext in ['.jpg', '.png']:
        path = frame_dir / f"{frame_idx:03d}{ext}"
        if path.exists():
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            media = "image/jpeg" if ext == ".jpg" else "image/png"
            return b64, media
    return None, None


def build_guided_message(frame_dir, frames_and_scores):
    """Config D: Frames + anomaly scores."""
    score_text = ", ".join(f"{s:.2f}" for _, s in frames_and_scores)
    text = (
        f"These {len(frames_and_scores)} frames are from a surveillance video "
        f"segment flagged as anomalous. The anomaly scores for these frames are: "
        f"[{score_text}]. Normal activity in this scene typically scores below 0.3. "
        f"Describe the anomalous activity observed."
    )

    content = [{"type": "text", "text": text}]
    for frame_idx, score in frames_and_scores:
        b64, media = encode_frame(frame_dir, frame_idx)
        if b64:
            content.append({
                "type": "text",
                "text": f"Frame {frame_idx} (anomaly score: {score:.2f}):"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{b64}", "detail": "high"}
            })

    return content


def build_unguided_message(frame_dir, frames):
    """Config A: Just frames, no scores."""
    content = [{
        "type": "text",
        "text": f"Here are {len(frames)} frames sampled uniformly from a surveillance video. "
                f"Examine them and describe any anomalous activity you observe."
    }]

    for frame_idx, _ in frames:
        b64, media = encode_frame(frame_dir, frame_idx)
        if b64:
            content.append({
                "type": "text",
                "text": f"Frame {frame_idx}:"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{b64}", "detail": "high"}
            })

    return content


# ─── GPT-4o Calls ────────────────────────────────────────────────────────────

def call_gpt4o(system_prompt, user_content):
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
        print(f"    GPT-4o ERROR: {e}")
        return {"explanation": f"ERROR: {e}"}


def judge_explanation(human_explanation, ai_explanation):
    user_msg = (
        f"HUMAN ground-truth explanation:\n\"{human_explanation}\"\n\n"
        f"AI-generated explanation:\n\"{ai_explanation}\""
    )
    return call_gpt4o(JUDGE_PROMPT, user_msg)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_config_d(video_id, annotation, model, device, threshold=0.5, top_k=5):
    """Config D: RTFM guidance + GPT-4o"""
    feat_path = TEST_FEAT_DIR / f"{video_id}_i3d.npy"
    if not feat_path.exists():
        return None, "no_features"

    scores, num_raw = rtfm_score_video(model, feat_path, device)
    max_score = float(scores.max())

    if max_score < threshold:
        return {
            "video_id": video_id,
            "config": "D",
            "max_score": max_score,
            "detected": False,
            "explanation": "Normal video — no anomaly detected by RTFM.",
            "scores": None,
        }, "below_threshold"

    frame_dir = FRAMES_DIR / video_id
    total_frames = len([f for f in frame_dir.iterdir() if f.suffix in ['.jpg', '.png']])
    frames_and_scores = get_top_snippet_frames(scores, num_raw, total_frames, top_k)

    user_content = build_guided_message(frame_dir, frames_and_scores)
    response = call_gpt4o(SYSTEM_PROMPT_GUIDED, user_content)
    explanation = response.get("explanation", "")

    judge_scores = judge_explanation(annotation["explanation"], explanation)

    return {
        "video_id": video_id,
        "config": "D",
        "max_score": max_score,
        "detected": True,
        "rtfm_scores": [float(s) for s in scores],
        "selected_frames": [(int(f), float(s)) for f, s in frames_and_scores],
        "explanation": explanation,
        "human_explanation": annotation["explanation"],
        "judge": judge_scores,
    }, "ok"


def run_config_a(video_id, annotation, num_frames=8):
    """Config A: No guidance, uniform frames, GPT-4o"""
    frame_dir = FRAMES_DIR / video_id
    total_frames = len([f for f in frame_dir.iterdir() if f.suffix in ['.jpg', '.png']])
    frames = get_uniform_frames(total_frames, num_frames)

    user_content = build_unguided_message(frame_dir, frames)
    response = call_gpt4o(SYSTEM_PROMPT_UNGUIDED, user_content)

    explanation = response.get("explanation", "")
    detected = response.get("anomaly_detected", True)

    judge_scores = judge_explanation(annotation["explanation"], explanation)

    return {
        "video_id": video_id,
        "config": "A",
        "detected": detected,
        "sampled_frames": [int(f) for f, _ in frames],
        "explanation": explanation,
        "human_explanation": annotation["explanation"],
        "judge": judge_scores,
    }, "ok"


def aggregate_results(results):
    metrics = {"correctness": [], "specificity": [], "completeness": [], "fluency": []}
    for r in results:
        if r.get("judge"):
            for key in metrics:
                val = r["judge"].get(key)
                if val is not None:
                    metrics[key].append(val)

    summary = {}
    for key, vals in metrics.items():
        if vals:
            summary[key] = {"mean": round(np.mean(vals), 2), "n": len(vals)}
    return summary


def main():
    parser = argparse.ArgumentParser(description="RTFM-based Anomaly Explanation Experiment")
    parser.add_argument("--config", type=str, default="D", choices=["A", "D", "all"])
    parser.add_argument("--checkpoint", type=str, default=str(RTFM_DIR / "ckpt" / "rtfm_best.pkl"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--video", type=str, default=None, help="Run on single video")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    with open(ANNOTATIONS_PATH) as f:
        annotations = {a["video_id"]: a for a in json.load(f)}
    print(f"Loaded {len(annotations)} annotated videos\n")

    configs = ["A", "D"] if args.config == "all" else [args.config]

    if args.video:
        video_ids = [args.video]
    else:
        video_ids = sorted(annotations.keys())

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rtfm_model = None
    if "D" in configs:
        print(f"Loading RTFM checkpoint: {args.checkpoint}")
        rtfm_model = load_rtfm_model(args.checkpoint, device)
        print("RTFM model loaded.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for config in configs:
        print(f"\n{'='*60}")
        print(f"  CONFIG {config}: {'RTFM guided + GPT-4o' if config == 'D' else 'Unguided + GPT-4o'}")
        print(f"{'='*60}\n")

        all_results = []

        for i, vid_id in enumerate(video_ids):
            ann = annotations[vid_id]
            print(f"[{i+1}/{len(video_ids)}] {vid_id}...", end=" ", flush=True)

            if config == "D":
                result, status = run_config_d(vid_id, ann, rtfm_model, device,
                                               threshold=args.threshold, top_k=args.top_k)
            else:
                result, status = run_config_a(vid_id, ann)

            if result is None:
                print(f"SKIPPED ({status})")
                continue

            expl = result.get("explanation", "")[:80]
            judge = result.get("judge", {})
            corr = judge.get("correctness", "?")
            print(f"corr={corr} | \"{expl}...\"")

            all_results.append(result)
            time.sleep(0.5)

        summary = aggregate_results(all_results)

        output = {
            "metadata": {
                "config": config,
                "timestamp": datetime.now().isoformat(),
                "num_videos": len(all_results),
                "threshold": args.threshold if config == "D" else None,
                "top_k": args.top_k if config == "D" else None,
            },
            "aggregate_scores": summary,
            "per_video": all_results,
        }

        out_path = RESULTS_DIR / f"config_{config}_results.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n--- Config {config} Summary ---")
        for key, val in summary.items():
            print(f"  {key}: {val['mean']:.2f} (n={val['n']})")
        print(f"Saved to: {out_path}\n")


if __name__ == "__main__":
    main()
