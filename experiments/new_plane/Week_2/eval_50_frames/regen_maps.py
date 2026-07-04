"""
Re-generates heatmap.png and overlay.png for every frame in eval_50_frames.

Inference is done exactly as the official code:
  - test_dataset.py  : data loading (prev/next ±3, normalize, swapaxes)
  - inference.py L56 : for ShanghaiTech → predictions_teacher only
                       (reconstruction error, NOT teacher-student discrepancy)
  - Spatial anomaly map = |target_rgb − recon_teacher_rgb|  (L1, per pixel)
    This is the spatial version of the scalar score used in inference.py
"""

import os, sys
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae")
from model.model_factory import mae_cvt_patch8

# ── paths ──────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/SHANGHAI/SHANGHAI_Test/frames"
GRADS_DIR  = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/SHANGHAI/SHANGHAI_Test/gradients2"
TEACHER_CKPT = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/shanghai/checkpoint-best.pth"
STUDENT_CKPT = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/shanghai/checkpoint-best-student.pth"

INPUT_SIZE  = (160, 320)   # (H, W) — Shanghai config
MASK_RATIO  = 0.5          # Shanghai config
DIRECTION   = 3            # ±3 as in test_dataset.py
DIGIT_PAD   = 3            # filenames are 000.jpg … 999.jpg

# ── load model ─────────────────────────────────────────────────────────────
print("Loading model …")
model = mae_cvt_patch8(
    norm_pix_loss=False,
    img_size=INPUT_SIZE,
    use_only_masked_tokens_ab=False,
    abnormal_score_func=["L1", "L1"],
    masking_method="random_masking",
    grad_weighted_loss=True,
).float()

teacher_sd = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)["model"]
student_sd = torch.load(STUDENT_CKPT, map_location="cpu", weights_only=False)["model"]
for key in student_sd:
    if "student" in key:
        teacher_sd[key] = student_sd[key]
model.load_state_dict(teacher_sd, strict=False)
model.eval()

# set exactly as inference.py lines 33-37
model.train_TS = True
model.abnormal_score_func_TS = "L1"
print("Model ready.\n")

# ── helpers ────────────────────────────────────────────────────────────────
def read_frame(video_dir, idx):
    """Load frame; fall back to idx itself when neighbour is missing."""
    path = os.path.join(video_dir, f"{str(idx).zfill(DIGIT_PAD)}.jpg")
    if os.path.exists(path):
        return cv2.imread(path)
    return None

def load_sample(video_id, frame_idx):
    """
    Replicates test_dataset.AbnormalDatasetGradientsTest.__getitem__ exactly.
    Returns (samples, grads, targets) as batched float32 tensors on CPU.
    """
    frame_dir = os.path.join(FRAMES_DIR, video_id)
    grad_dir  = os.path.join(GRADS_DIR,  video_id)

    curr = cv2.imread(os.path.join(frame_dir, f"{str(frame_idx).zfill(DIGIT_PAD)}.jpg"))

    # prev / next with fallback (test_dataset.read_prev_next_frame_if_exists)
    prev_path = os.path.join(frame_dir, f"{str(frame_idx - DIRECTION).zfill(DIGIT_PAD)}.jpg")
    prev = cv2.imread(prev_path) if os.path.exists(prev_path) else curr.copy()

    next_path = os.path.join(frame_dir, f"{str(frame_idx + DIRECTION).zfill(DIGIT_PAD)}.jpg")
    nxt  = cv2.imread(next_path) if os.path.exists(next_path) else curr.copy()

    grad = cv2.imread(os.path.join(grad_dir, f"{str(frame_idx).zfill(DIGIT_PAD)}.jpg"))

    # resize if needed (test_dataset line 61-64)
    H, W = INPUT_SIZE
    if curr.shape[:2] != (H, W):
        curr = cv2.resize(curr, (W, H))
        prev = cv2.resize(prev, (W, H))
        nxt  = cv2.resize(nxt,  (W, H))
        grad = cv2.resize(grad, (W, H))

    # 9-channel input (test_dataset line 58)
    img = np.concatenate([prev, curr, nxt], axis=-1)

    # target = current_img + zero anomaly channel (test_dataset lines 65-66)
    mask   = np.zeros((H, W, 1), dtype=np.uint8)
    target = np.concatenate([curr, mask], axis=-1)

    # normalize (test_dataset lines 67-71)
    img    = (img.astype(np.float32)    - 127.5) / 127.5
    target = (target.astype(np.float32) - 127.5) / 127.5
    grad   = grad.astype(np.float32)

    # (H,W,C) → (C,H,W) via swapaxes (test_dataset lines 72-74)
    img    = np.swapaxes(img,    0, -1).swapaxes(1, -1)   # (9,H,W)
    target = np.swapaxes(target, 0, -1).swapaxes(1, -1)   # (4,H,W)
    grad   = np.swapaxes(grad,   0,  1).swapaxes(0, -1)   # (3,H,W)

    samples = torch.from_numpy(img).unsqueeze(0)     # (1,9,H,W)
    grads   = torch.from_numpy(grad).unsqueeze(0)    # (1,3,H,W)
    targets = torch.from_numpy(target).unsqueeze(0)  # (1,4,H,W)
    return samples, grads, targets

# ── process all 50 folders ─────────────────────────────────────────────────
folders = sorted([
    d for d in os.listdir(BASE)
    if os.path.isdir(os.path.join(BASE, d))
])

print(f"Found {len(folders)} frame folders.\n")

for folder in folders:
    # parse  e.g. "47_03_0031_frame372"  →  video_id="03_0031"  frame=372
    left, frame_str = folder.rsplit("_frame", 1)
    video_id  = "_".join(left.split("_")[1:])   # drop leading seq index
    frame_idx = int(frame_str)

    out_dir = os.path.join(BASE, folder)

    print(f"[{folder}]  video={video_id}  frame={frame_idx}")

    # ── inference — encoder + teacher decoder only (Shanghai: predictions_teacher)
    samples, grads, targets = load_sample(video_id, frame_idx)
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(samples, MASK_RATIO, grads)
        _, pred_teacher = model.forward_decoder_TS(latent, ids_restore)

    recon_teacher = model.unpatchify(pred_teacher)   # (1,4,H,W)

    # L1 reconstruction error on RGB channels — matches inference.py L364:
    #   torch.abs(imgs - pred_teacher).mean((2))  →  spatial version
    amap = torch.abs(targets[:, :3] - recon_teacher[:, :3]).mean(dim=1)  # (1,H,W)
    amap = amap[0].cpu().numpy()   # (H,W)

    # normalise to [0,1] for display
    amin, amax = amap.min(), amap.max()
    amap_norm = (amap - amin) / (amax - amin + 1e-8)

    # ── load original frame for overlay (RGB) ────────────────────────────
    orig_path = os.path.join(out_dir, "original.png")
    orig_bgr  = cv2.imread(orig_path)
    orig_rgb  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # ── save heatmap.png  (matplotlib grayscale, smooth rendering) ────────
    fig, ax = plt.subplots(1, 1, figsize=(orig_rgb.shape[1]/100, orig_rgb.shape[0]/100), dpi=100)
    ax.imshow(amap_norm, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_dir, "heatmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # ── save overlay.png  (original + jet colormap at alpha=0.5) ─────────
    fig, ax = plt.subplots(1, 1, figsize=(orig_rgb.shape[1]/100, orig_rgb.shape[0]/100), dpi=100)
    ax.imshow(orig_rgb)
    ax.imshow(amap_norm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_dir, "overlay.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print("\nDone. heatmap.png and overlay.png updated for all frames.")
