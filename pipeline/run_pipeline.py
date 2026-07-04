#!/usr/bin/env python3
"""
End-to-end anomaly detection + explanation pipeline.

1. Run AED-MAE on all video frames -> per-frame anomaly scores
2. Smooth + normalize scores
3. Hysteresis thresholding -> anomalous temporal segments
4. Sample K frames per segment
5. Generate heatmap overlays for sampled frames
6. Save everything for downstream VLM explanation

Usage:
    python run_pipeline.py --video 01_0027
    python run_pipeline.py --video 01_0027 --tau-high 0.4 --tau-low 0.2
    python run_pipeline.py --batch
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import json
import argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models" / "aed-mae"
sys.path.insert(0, str(MODEL_DIR))

from model.model_factory import mae_cvt_patch8
from util.abnormal_utils import filt

FRAMES_DIR = BASE / "SHANGHAI" / "SHANGHAI_Test" / "frames"
GRADS_DIR = BASE / "SHANGHAI" / "SHANGHAI_Test" / "gradients2"
LABEL_DIR = BASE / "SHANGHAI" / "SHANGHAI_Test" / "label"
TEST_TXT = BASE / "SHANGHAI" / "SHANGHAI_Test" / "SHANGHAI_test.txt"
OUTPUT_BASE = BASE / "pipeline" / "outputs"

TEACHER_CKPT = MODEL_DIR / "experiments" / "shanghai" / "checkpoint-best.pth"
STUDENT_CKPT = MODEL_DIR / "experiments" / "shanghai" / "checkpoint-best-student.pth"

H, W = 160, 320
MASK_RATIO = 0.5
DIRECTION = 3

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    print("Loading AED-MAE model ...")
    model = mae_cvt_patch8(
        norm_pix_loss=False,
        img_size=(H, W),
        use_only_masked_tokens_ab=False,
        abnormal_score_func=["L1", "L1"],
        masking_method="random_masking",
        grad_weighted_loss=True,
    ).float()

    teacher_sd = torch.load(str(TEACHER_CKPT), map_location="cpu", weights_only=False)["model"]
    student_sd = torch.load(str(STUDENT_CKPT), map_location="cpu", weights_only=False)["model"]
    for key in student_sd:
        if "student" in key:
            teacher_sd[key] = student_sd[key]
    model.load_state_dict(teacher_sd, strict=False)

    model.eval()
    model.train_TS = True
    model.abnormal_score_func_TS = "L1"
    print("Model loaded.\n")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (matches official test_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

def _read_indexed_jpg(video_dir, idx):
    """Read frame by index, supporting both 3-digit and 4-digit naming."""
    if idx < 0:
        return None

    candidates = [
        video_dir / f"{str(idx).zfill(3)}.jpg",
        video_dir / f"{str(idx).zfill(4)}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return cv2.imread(str(path))
    return None


def _read_frame(video_dir, idx):
    return _read_indexed_jpg(video_dir, idx)


def load_frame(video_id, frame_idx):
    """Load a single frame with its neighbours and gradient.
    Returns (samples, grads, targets, curr_rgb) tensors ready for model."""
    frame_dir = FRAMES_DIR / video_id
    grad_dir = GRADS_DIR / video_id

    curr = _read_indexed_jpg(frame_dir, frame_idx)
    if curr is None:
        raise FileNotFoundError(
            f"Missing frame for video={video_id}, idx={frame_idx} "
            f"(tried 3-digit and 4-digit JPG names)"
        )

    prev = _read_frame(frame_dir, frame_idx - DIRECTION)
    if prev is None:
        prev = curr.copy()

    nxt = _read_frame(frame_dir, frame_idx + DIRECTION)
    if nxt is None:
        nxt = curr.copy()

    # Try pre-computed gradient, else compute on-the-fly
    grad = _read_indexed_jpg(grad_dir, frame_idx)
    if grad is None:
        grad = np.abs(prev.astype(np.float32) - nxt.astype(np.float32)).astype(np.uint8)

    # Resize
    curr = cv2.resize(curr, (W, H))
    prev = cv2.resize(prev, (W, H))
    nxt = cv2.resize(nxt, (W, H))
    grad = cv2.resize(grad, (W, H))

    # Keep original RGB for visualization before normalization
    curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)

    # 9-channel input: prev + curr + next
    img = np.concatenate([prev, curr, nxt], axis=-1)

    # Target: curr RGB + zero anomaly channel
    mask_ch = np.zeros((H, W, 1), dtype=np.uint8)
    target = np.concatenate([curr, mask_ch], axis=-1)

    # Normalize
    img = (img.astype(np.float32) - 127.5) / 127.5
    target = (target.astype(np.float32) - 127.5) / 127.5
    grad = grad.astype(np.float32)

    # (H,W,C) -> (C,H,W)
    img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
    target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
    grad = np.swapaxes(grad, 0, 1).swapaxes(0, -1)

    samples = torch.from_numpy(img).unsqueeze(0)
    grads = torch.from_numpy(grad).unsqueeze(0)
    targets = torch.from_numpy(target).unsqueeze(0)

    return samples, grads, targets, curr_rgb

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Score all frames
# ──────────────────────────────────────────────────────────────────────────────

def score_all_frames(model, video_id, num_frames):
    """Run AED-MAE on every processable frame, return raw teacher scores."""
    raw_scores = []
    start_idx = DIRECTION
    end_idx = num_frames - DIRECTION

    for idx in range(start_idx, end_idx):
        if idx % 50 == 0:
            print(f"  Scoring frame {idx}/{num_frames} ...", flush=True)

        samples, grads, targets, _ = load_frame(video_id, idx)

        with torch.no_grad():
            _, _, _, scores = model(
                samples, targets=targets, grad_mask=grads, mask_ratio=MASK_RATIO
            )
            raw_scores.append(scores[1].item())

    return np.array(raw_scores)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Smooth, normalize, detect segments
# ──────────────────────────────────────────────────────────────────────────────

def smooth_and_normalize(raw_scores):
    smoothed = filt(raw_scores, dim=9, range=900, mu=282)
    smin, smax = smoothed.min(), smoothed.max()
    if smax - smin < 1e-8:
        return smoothed
    return (smoothed - smin) / (smax - smin)


def find_anomaly_segments(scores, tau_high=0.5, tau_low=0.3, min_len=8):
    """Hysteresis thresholding: start when score > tau_high, end when < tau_low."""
    segments = []
    in_segment = False
    start = 0

    for i, s in enumerate(scores):
        if not in_segment and s > tau_high:
            in_segment = True
            start = i
        elif in_segment and s < tau_low:
            if i - start >= min_len:
                segments.append((start, i))
            in_segment = False

    if in_segment and len(scores) - start >= min_len:
        segments.append((start, len(scores)))

    return segments


def sample_frame_indices(segment_start, segment_end, K=5):
    """Uniformly sample K frame indices from a segment."""
    length = segment_end - segment_start
    if length <= K:
        return list(range(segment_start, segment_end))
    step = length / (K + 1)
    return [segment_start + int(step * (i + 1)) for i in range(K)]

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Generate heatmaps for sampled frames
# ──────────────────────────────────────────────────────────────────────────────

def generate_heatmap(model, video_id, frame_idx):
    """Run encoder + teacher decoder to get pixel-level anomaly map.
    Returns (anomaly_map_normalized, curr_rgb)."""
    samples, grads, targets, curr_rgb = load_frame(video_id, frame_idx)

    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(samples, MASK_RATIO, grads)
        _, pred_teacher = model.forward_decoder_TS(latent, ids_restore)
        recon = model.unpatchify(pred_teacher)

    amap = torch.abs(targets[:, :3] - recon[:, :3]).mean(dim=1)
    amap = amap[0].cpu().numpy()

    amin, amax = amap.min(), amap.max()
    amap_norm = (amap - amin) / (amax - amin + 1e-8)

    return amap_norm, curr_rgb

# ──────────────────────────────────────────────────────────────────────────────
# Saving outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_frame_outputs(out_dir, frame_idx, amap_norm, curr_rgb):
    """Save original frame, heatmap, and overlay for one sampled frame."""
    out_dir = Path(out_dir)
    prefix = f"frame_{str(frame_idx).zfill(4)}"

    # Original
    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(curr_rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_dir / f"{prefix}_original.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Heatmap
    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(amap_norm, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_dir / f"{prefix}_heatmap.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    # Overlay
    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    plt.imshow(curr_rgb)
    plt.imshow(amap_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_dir / f"{prefix}_overlay.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def save_score_plot(out_dir, video_id, smoothed_scores, segments, gt=None):
    """Save a timeline plot with detected segments highlighted."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(smoothed_scores, linewidth=1.5, color="steelblue", label="Anomaly score")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="tau_high")
    ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.4, label="tau_low")

    for i, (s, e) in enumerate(segments):
        ax.axvspan(s, e, alpha=0.25, color="red",
                   label="Detected segment" if i == 0 else None)

    if gt is not None:
        gt_sub = gt[DIRECTION: DIRECTION + len(smoothed_scores)]
        ax.fill_between(
            range(len(gt_sub)), 0, 1,
            where=gt_sub == 1, alpha=0.10, color="green", label="Ground truth"
        )

    ax.set_xlabel("Frame index (offset by 3)")
    ax.set_ylabel("Normalized score")
    ax.set_title(f"Video {video_id}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "scores_plot.png", dpi=150)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline for one video
# ──────────────────────────────────────────────────────────────────────────────

def process_video(model, video_id, num_frames, K=5,
                  tau_high=0.5, tau_low=0.3, min_len=8):
    out_dir = OUTPUT_BASE / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Processing video: {video_id}  ({num_frames} frames)")
    print(f"{'='*60}")

    # --- Phase 1: Score all frames ---
    print("\n[Phase 1] Scoring all frames ...")
    raw_scores = score_all_frames(model, video_id, num_frames)
    print(f"  Raw scores: min={raw_scores.min():.4f}  max={raw_scores.max():.4f}")

    # --- Phase 2: Smooth + normalize + segment ---
    print("\n[Phase 2] Smoothing, normalizing, detecting segments ...")
    smoothed = smooth_and_normalize(raw_scores)
    segments = find_anomaly_segments(smoothed, tau_high, tau_low, min_len)

    # Map segment indices back to actual frame numbers (offset by DIRECTION)
    segments_absolute = [(s + DIRECTION, e + DIRECTION) for s, e in segments]

    print(f"  Normalized range: [{smoothed.min():.4f}, {smoothed.max():.4f}]")
    print(f"  Detected {len(segments)} anomaly segment(s):")
    for i, (s, e) in enumerate(segments_absolute):
        print(f"    Segment {i}: frames {s}-{e} ({e - s} frames)")

    # Save scores
    np.save(out_dir / "raw_scores.npy", raw_scores)
    np.save(out_dir / "smoothed_scores.npy", smoothed)

    # Load ground truth if available (for plot only)
    gt = None
    gt_path = LABEL_DIR / f"{video_id}.npy"
    if gt_path.exists():
        gt = np.load(str(gt_path))

    save_score_plot(out_dir, video_id, smoothed, segments, gt)

    # --- Phase 3: Sample frames + generate heatmaps per segment ---
    print(f"\n[Phase 3] Generating heatmaps for sampled frames ...")
    all_segment_data = []

    for seg_i, (seg_s, seg_e) in enumerate(segments):
        abs_s, abs_e = segments_absolute[seg_i]
        seg_dir = out_dir / f"segment_{seg_i}"
        seg_dir.mkdir(exist_ok=True)

        sampled_relative = sample_frame_indices(seg_s, seg_e, K)
        sampled_absolute = [idx + DIRECTION for idx in sampled_relative]

        print(f"  Segment {seg_i} (frames {abs_s}-{abs_e}): "
              f"sampled {len(sampled_absolute)} frames {sampled_absolute}")

        frame_data = []
        for frame_idx in sampled_absolute:
            amap, rgb = generate_heatmap(model, video_id, frame_idx)
            save_frame_outputs(seg_dir, frame_idx, amap, rgb)
            frame_data.append({
                "frame_idx": frame_idx,
                "anomaly_score": float(smoothed[frame_idx - DIRECTION]),
            })

        seg_meta = {
            "segment_index": seg_i,
            "start_frame": abs_s,
            "end_frame": abs_e,
            "duration_frames": abs_e - abs_s,
            "sampled_frames": frame_data,
        }
        all_segment_data.append(seg_meta)

        # Save per-segment metadata
        with open(seg_dir / "metadata.json", "w") as f:
            json.dump(seg_meta, f, indent=2)

    # --- Save top-level metadata ---
    meta = {
        "video_id": video_id,
        "total_frames": num_frames,
        "processable_frames": len(raw_scores),
        "tau_high": tau_high,
        "tau_low": tau_low,
        "min_segment_length": min_len,
        "K_sampled": K,
        "num_segments_detected": len(segments),
        "segments": all_segment_data,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Outputs saved to: {out_dir}")
    return meta

# ──────────────────────────────────────────────────────────────────────────────
# Batch mode helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_anomalous_videos():
    """Parse SHANGHAI_test.txt and return list of (video_id, num_frames) for anomalous videos."""
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

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anomaly detection + explanation pipeline")
    parser.add_argument("--video", type=str, help="Single video ID, e.g. 01_0027")
    parser.add_argument("--batch", action="store_true", help="Process all 44 anomalous videos")
    parser.add_argument("--K", type=int, default=5, help="Number of frames to sample per segment")
    parser.add_argument("--tau-high", type=float, default=0.5, help="Hysteresis high threshold")
    parser.add_argument("--tau-low", type=float, default=0.3, help="Hysteresis low threshold")
    parser.add_argument("--min-len", type=int, default=8, help="Minimum segment length in frames")
    args = parser.parse_args()

    if not args.video and not args.batch:
        parser.error("Specify --video VIDEO_ID or --batch")

    model = load_model()

    if args.batch:
        videos = get_anomalous_videos()
        print(f"Batch mode: {len(videos)} anomalous videos\n")
        all_results = []
        for i, (vid, nframes) in enumerate(videos):
            print(f"\n[{i+1}/{len(videos)}] ", end="")
            meta = process_video(
                model, vid, nframes,
                K=args.K, tau_high=args.tau_high,
                tau_low=args.tau_low, min_len=args.min_len
            )
            all_results.append(meta)

        summary_path = OUTPUT_BASE / "batch_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nBatch complete. Summary: {summary_path}")
    else:
        frame_dir = FRAMES_DIR / args.video
        if not frame_dir.exists():
            print(f"ERROR: frames directory not found: {frame_dir}")
            sys.exit(1)
        nframes = len(list(frame_dir.glob("*.jpg")))
        process_video(
            model, args.video, nframes,
            K=args.K, tau_high=args.tau_high,
            tau_low=args.tau_low, min_len=args.min_len
        )


if __name__ == "__main__":
    main()
