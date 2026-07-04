#!/usr/bin/env python3
"""
RTFM-based anomaly detection pipeline.

1. Run all 199 test videos through the trained RTFM model → per-snippet scores
2. Video-Level Gate: flag videos where score_abnormal > GATE_THRESHOLD
3. For flagged videos: find contiguous anomalous segments (score > segment_threshold)
4. Smart frame sampling from anomalous segments (onset/resolution + score-weighted)
5. Extract actual frames from video / frame directory
6. Save everything for downstream VLM explanation

Usage:
    python run_rtfm_pipeline.py                          # all test videos
    python run_rtfm_pipeline.py --video 01_0015          # single video
    python run_rtfm_pipeline.py --gate-threshold 0.2     # custom gate
    python run_rtfm_pipeline.py --segment-threshold 0.5  # custom segment threshold
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import json
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RTFM_DIR = BASE / "rtfm"
sys.path.insert(0, str(RTFM_DIR))

from model import Model
from option import parser as rtfm_parser

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = RTFM_DIR / "data" / "SH_Test_ten_crop_i3d"
TEST_LIST = RTFM_DIR / "list" / "shanghai-i3d-test-10crop.list"
CHECKPOINT = RTFM_DIR / "ckpt" / "rtfm_best.pkl"
FRAMES_DIR = BASE / "data" / "SHANGHAI" / "SHANGHAI_Test" / "frames"
VIDEOS_DIR = BASE / "data" / "SHANGHAI" / "anomalous_videos"
OUTPUT_BASE = BASE / "pipeline" / "rtfm_outputs"

# ── Defaults ─────────────────────────────────────────────────────────────────
GATE_THRESHOLD = 0.2
SEGMENT_THRESHOLD = 0.3
TOTAL_FRAME_BUDGET = 8
MIN_GAP = 2
SNIPPETS_PER_SEGMENT = 16


def load_rtfm_model():
    """Load the trained RTFM model."""
    args = rtfm_parser.parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = Model(args.feature_size, args.batch_size)
    net.load_state_dict(torch.load(str(CHECKPOINT), map_location=device))
    net = net.to(device).eval()
    print(f"RTFM model loaded on {device}  (k_abn={net.k_abn})")
    return net, device


def load_test_paths():
    """Load the test feature file list."""
    with open(str(TEST_LIST)) as f:
        return [l.strip() for l in f if l.strip()]


def video_name_from_path(path):
    return os.path.basename(path).replace(".npy", "").replace("_i3d", "")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1-2: RTFM Inference + Video-Level Gate
# ─────────────────────────────────────────────────────────────────────────────

def score_video(net, device, feature_path):
    """Run RTFM on one video. Returns (gate_score, segment_scores)."""
    features = np.load(feature_path, allow_pickle=True).astype(np.float32)
    inp = torch.from_numpy(features).unsqueeze(0).to(device)  # [1, T, 10, 2048]
    inp = inp.permute(0, 2, 1, 3)                              # [1, 10, T, 2048]

    with torch.no_grad():
        score_abnormal, _, _, _, _, _, logits, _, _, _ = net(inputs=inp)

    logits = torch.squeeze(logits, 1)
    seg_scores = torch.mean(logits, 0).squeeze().cpu().numpy()
    if seg_scores.ndim == 0:
        seg_scores = seg_scores.reshape(1)

    gate_score = float(score_abnormal.squeeze().cpu())
    return gate_score, seg_scores


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Find Contiguous Anomalous Segments
# ─────────────────────────────────────────────────────────────────────────────

def find_anomalous_segments(seg_scores, threshold):
    """Find contiguous runs where score > threshold. Returns list of (start, end) snippet indices."""
    segments = []
    in_seg = False
    start = 0

    for i, s in enumerate(seg_scores):
        if not in_seg and s > threshold:
            in_seg = True
            start = i
        elif in_seg and s <= threshold:
            segments.append((start, i - 1))
            in_seg = False

    if in_seg:
        segments.append((start, len(seg_scores) - 1))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Smart Snippet Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_snippets_from_segments(segments, seg_scores, total_budget, min_gap):
    """
    Allocate frame budget proportionally by segment length, then within each
    segment: always include first/last (onset + resolution), fill remaining
    budget with score-weighted selection respecting min_gap spacing.
    """
    if not segments:
        return []

    total_length = sum(e - s + 1 for s, e in segments)
    all_selected = []

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start + 1
        budget = max(2, round(total_budget * seg_len / total_length))

        selected = set()
        selected.add(seg_start)
        selected.add(seg_end)

        remaining_budget = budget - len(selected)
        if remaining_budget > 0 and seg_len > 2:
            candidates = []
            for i in range(seg_start + 1, seg_end):
                candidates.append((i, seg_scores[i]))
            candidates.sort(key=lambda x: x[1], reverse=True)

            for idx, score in candidates:
                if remaining_budget <= 0:
                    break
                too_close = any(abs(idx - s) < min_gap for s in selected)
                if too_close:
                    continue
                selected.add(idx)
                remaining_budget -= 1

        seg_selected = sorted(selected)
        all_selected.append({
            "segment": (seg_start, seg_end),
            "selected_snippets": [
                {"snippet_idx": idx, "score": float(seg_scores[idx])}
                for idx in seg_selected
            ],
        })

    return all_selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Frame Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_from_video(video_path, frame_num):
    """Extract a single frame from an MP4 file. Returns BGR numpy array or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_frame_from_dir(frames_dir, frame_num):
    """Extract a frame from the frames directory. Supports 3-digit and 4-digit naming."""
    for fmt in [f"{frame_num:03d}.jpg", f"{frame_num:04d}.jpg"]:
        path = frames_dir / fmt
        if path.exists():
            return cv2.imread(str(path))
    return None


def snippet_to_frame_num(snippet_idx, total_frames, n_segments):
    """Map a snippet index back to the middle frame of that snippet's range."""
    frames_per_snippet = total_frames / n_segments
    start_frame = int(snippet_idx * frames_per_snippet)
    mid_frame = start_frame + int(frames_per_snippet / 2)
    return min(mid_frame, total_frames - 1)


def extract_frames_for_video(video_id, selected_segments, n_segments, total_frames=None):
    """
    Extract actual frames for all selected snippets.
    Tries frame directory first, then MP4.
    Returns list of (frame_num, bgr_image) tuples, or None for missing frames.
    """
    frame_dir = FRAMES_DIR / video_id
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    use_video = not frame_dir.exists() and video_path.exists()

    if total_frames is None:
        if frame_dir.exists():
            total_frames = len(list(frame_dir.glob("*.jpg")))
        elif video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            return []

    extracted = []
    for seg_data in selected_segments:
        for snip in seg_data["selected_snippets"]:
            frame_num = snippet_to_frame_num(
                snip["snippet_idx"], total_frames, n_segments
            )
            if use_video:
                img = extract_frame_from_video(video_path, frame_num)
            else:
                img = extract_frame_from_dir(frame_dir, frame_num)

            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = None

            extracted.append({
                "snippet_idx": snip["snippet_idx"],
                "score": snip["score"],
                "frame_num": frame_num,
                "image": rgb,
            })

    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Saving Outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline_outputs(video_id, gate_score, seg_scores, segments_data,
                          extracted_frames, out_dir):
    """Save all pipeline outputs: metadata, scores, frames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ef in extracted_frames:
        if ef["image"] is not None:
            fname = f"snippet_{ef['snippet_idx']:03d}_frame_{ef['frame_num']:04d}.jpg"
            rgb = ef["image"]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / fname), bgr)

    metadata = {
        "video_id": video_id,
        "gate_score": gate_score,
        "n_segments": len(seg_scores),
        "segment_scores": [float(s) for s in seg_scores],
        "anomalous_segments": [
            {
                "start_snippet": sd["segment"][0],
                "end_snippet": sd["segment"][1],
                "selected_snippets": sd["selected_snippets"],
            }
            for sd in segments_data
        ],
        "extracted_frames": [
            {
                "snippet_idx": ef["snippet_idx"],
                "score": ef["score"],
                "frame_num": ef["frame_num"],
                "file": f"snippet_{ef['snippet_idx']:03d}_frame_{ef['frame_num']:04d}.jpg"
                        if ef["image"] is not None else None,
            }
            for ef in extracted_frames
        ],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_single_video(net, device, feature_path, video_id,
                         gate_threshold, segment_threshold,
                         total_budget, min_gap):
    """Full pipeline for one video. Returns metadata dict or None if gated out."""
    gate_score, seg_scores = score_video(net, device, feature_path)

    if gate_score <= gate_threshold:
        return {
            "video_id": video_id,
            "gate_score": gate_score,
            "flagged": False,
            "n_segments": len(seg_scores),
        }

    segments = find_anomalous_segments(seg_scores, segment_threshold)

    if not segments:
        # Fallback: select the top-k highest-scoring snippets as a single pseudo-segment
        ranked = sorted(range(len(seg_scores)), key=lambda i: seg_scores[i], reverse=True)
        top_k = min(total_budget, len(ranked))
        top_indices = sorted(ranked[:top_k])
        if top_indices:
            segments = [(top_indices[0], top_indices[-1])]

    segments_data = sample_snippets_from_segments(
        segments, seg_scores, total_budget, min_gap
    )

    extracted = extract_frames_for_video(
        video_id, segments_data, len(seg_scores)
    )

    out_dir = OUTPUT_BASE / video_id
    metadata = save_pipeline_outputs(
        video_id, gate_score, seg_scores, segments_data, extracted, out_dir
    )
    metadata["flagged"] = True
    return metadata


def main():
    parser = argparse.ArgumentParser(description="RTFM anomaly detection pipeline")
    parser.add_argument("--video", type=str, default=None,
                        help="Single video ID (e.g. 01_0015). Omit for all test videos.")
    parser.add_argument("--gate-threshold", type=float, default=GATE_THRESHOLD,
                        help=f"Video-level gate threshold (default: {GATE_THRESHOLD})")
    parser.add_argument("--segment-threshold", type=float, default=SEGMENT_THRESHOLD,
                        help=f"Segment-level anomaly threshold (default: {SEGMENT_THRESHOLD})")
    parser.add_argument("--frame-budget", type=int, default=TOTAL_FRAME_BUDGET,
                        help=f"Total frames to sample per video (default: {TOTAL_FRAME_BUDGET})")
    parser.add_argument("--min-gap", type=int, default=MIN_GAP,
                        help=f"Minimum snippet gap between samples (default: {MIN_GAP})")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    net, device = load_rtfm_model()
    test_paths = load_test_paths()

    if args.video:
        matches = [p for p in test_paths if args.video in os.path.basename(p)]
        if not matches:
            print(f"ERROR: video '{args.video}' not found in test list")
            sys.exit(1)
        test_paths = matches

    print(f"\nConfig:")
    print(f"  Gate threshold    : {args.gate_threshold}")
    print(f"  Segment threshold : {args.segment_threshold}")
    print(f"  Frame budget      : {args.frame_budget}")
    print(f"  Min gap           : {args.min_gap}")
    print(f"  Videos to process : {len(test_paths)}")
    print()

    all_results = []
    n_flagged = 0
    n_skipped = 0
    n_with_segments = 0

    for i, path in enumerate(test_paths):
        video_id = video_name_from_path(path)
        result = process_single_video(
            net, device, path, video_id,
            args.gate_threshold, args.segment_threshold,
            args.frame_budget, args.min_gap,
        )

        is_flagged = result.get("flagged", False)
        has_segments = len(result.get("anomalous_segments", [])) > 0

        if is_flagged:
            n_flagged += 1
            if has_segments:
                n_with_segments += 1
                n_frames = len(result.get("extracted_frames", []))
                print(f"  [{i+1:3d}/{len(test_paths)}] {video_id}  "
                      f"gate={result['gate_score']:.3f}  "
                      f"segments={len(result['anomalous_segments'])}  "
                      f"frames={n_frames}")
            else:
                print(f"  [{i+1:3d}/{len(test_paths)}] {video_id}  "
                      f"gate={result['gate_score']:.3f}  "
                      f"flagged but no segments > {args.segment_threshold}")
        else:
            n_skipped += 1

        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"  Total: {len(all_results)}  "
          f"Flagged: {n_flagged}  "
          f"With segments: {n_with_segments}  "
          f"Skipped: {n_skipped}")
    print(f"{'='*60}")

    summary = {
        "gate_threshold": args.gate_threshold,
        "segment_threshold": args.segment_threshold,
        "frame_budget": args.frame_budget,
        "min_gap": args.min_gap,
        "total_videos": len(all_results),
        "flagged": n_flagged,
        "with_segments": n_with_segments,
        "skipped": n_skipped,
        "videos": all_results,
    }
    summary_path = OUTPUT_BASE / "pipeline_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
