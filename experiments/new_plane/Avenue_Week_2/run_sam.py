"""
SAM-enhanced anomaly localization for Avenue eval_50_frames.

Two-phase pipeline:
  Phase 1: Re-run Avenue anomaly model → save raw anomaly maps (.npy)
  Phase 2: Run SAM → segment each frame → score segments by anomaly map
           → output clean sam_overlay.png per frame

Adds to each eval_50_frames subfolder:
  - anomaly_raw.npy   (raw anomaly map from model)
  - sam_overlay.png    (original + top anomalous SAM segments highlighted)
"""

import os, sys, glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae")

# ── paths ──────────────────────────────────────────────────────────────────
EVAL_DIR     = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/new_plane/Avenue_Week_2/eval_50_frames"
FRAMES_ROOT  = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/Avenue_dataset_related_Work/Avenue Dataset/test/frames"
GRADS_ROOT   = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/Avenue_dataset_related_Work/Avenue Dataset/test/gradients2"
TEACHER_CKPT = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/avenue/checkpoint-best.pth"
STUDENT_CKPT = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/avenue/checkpoint-best-student.pth"
SAM_CKPT     = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/sam/sam_vit_b_01ec64.pth"

INPUT_SIZE = (320, 640)   # (H, W) — Avenue config
MASK_RATIO = 0.5
DIRECTION  = 3

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Anomaly model: compute and save raw anomaly maps
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("PHASE 1: Computing raw anomaly maps …")
print("="*60)

from model.model_factory import mae_cvt_patch16

model = mae_cvt_patch16(
    norm_pix_loss=False,
    img_size=INPUT_SIZE,
    use_only_masked_tokens_ab=False,
    abnormal_score_func=["L2", "L2"],
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
model.train_TS = True
model.abnormal_score_func_TS = "L2"

def load_sample(vid_name, frame_idx):
    frame_dir = os.path.join(FRAMES_ROOT, vid_name)
    grad_dir  = os.path.join(GRADS_ROOT,  vid_name)
    H, W = INPUT_SIZE

    def read_frame(idx):
        path = os.path.join(frame_dir, f"{idx:04d}.jpg")
        if os.path.exists(path):
            return cv2.imread(path)
        return cv2.imread(os.path.join(frame_dir, f"{frame_idx:04d}.jpg"))

    curr = read_frame(frame_idx)
    prev = read_frame(max(0, frame_idx - DIRECTION))
    nxt  = read_frame(frame_idx + DIRECTION) if os.path.exists(
              os.path.join(frame_dir, f"{frame_idx+DIRECTION:04d}.jpg")) else curr.copy()
    grad = cv2.imread(os.path.join(grad_dir, f"{frame_idx:04d}.jpg"))

    if curr.shape[:2] != (H, W):
        curr = cv2.resize(curr, (W, H))
        prev = cv2.resize(prev, (W, H))
        nxt  = cv2.resize(nxt,  (W, H))
        grad = cv2.resize(grad, (W, H))

    img    = np.concatenate([prev, curr, nxt], axis=-1)
    mask   = np.zeros((H, W, 1), dtype=np.uint8)
    target = np.concatenate([curr, mask], axis=-1)

    img    = (img.astype(np.float32)    - 127.5) / 127.5
    target = (target.astype(np.float32) - 127.5) / 127.5
    grad   = grad.astype(np.float32)

    img    = np.swapaxes(img,    0, -1).swapaxes(1, -1)
    target = np.swapaxes(target, 0, -1).swapaxes(1, -1)
    grad   = np.swapaxes(grad,   0,  1).swapaxes(0, -1)

    return (torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(grad).unsqueeze(0),
            torch.from_numpy(target).unsqueeze(0))

folders = sorted([d for d in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, d))])

for folder in folders:
    npy_path = os.path.join(EVAL_DIR, folder, "anomaly_raw.npy")
    if os.path.exists(npy_path):
        print(f"  [{folder}] already computed, skipping")
        continue

    # parse folder name: "01_vid01_frame77" → vid="01", frame=77
    parts = folder.split("_vid")[1]
    vid_name, frame_str = parts.split("_frame")
    frame_idx = int(frame_str)

    samples, grads, targets = load_sample(vid_name, frame_idx)
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(samples, MASK_RATIO, grads)
        pred_stud, pred_teacher   = model.forward_decoder_TS(latent, ids_restore)

    recon_teacher = model.unpatchify(pred_teacher)
    recon_student = model.unpatchify(pred_stud)

    # Avenue: both terms combined (L2)
    recon_err = ((targets[:, :3] - recon_teacher[:, :3]) ** 2).mean(dim=1)
    ts_gap    = ((recon_teacher[:, :3] - recon_student[:, :3]) ** 2).mean(dim=1)
    amap      = (recon_err + ts_gap)[0].cpu().numpy()

    np.save(npy_path, amap)
    print(f"  [{folder}] saved anomaly_raw.npy")

# free anomaly model memory
del model, teacher_sd, student_sd
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc; gc.collect()

print("Phase 1 complete.\n")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — SAM: segment + score + visualize
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("PHASE 2: Running SAM segmentation …")
print("="*60)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)

print("SAM loaded.\n")

for folder in folders:
    out_path = os.path.join(EVAL_DIR, folder, "sam_overlay.png")
    orig_path = os.path.join(EVAL_DIR, folder, "original.png")
    npy_path  = os.path.join(EVAL_DIR, folder, "anomaly_raw.npy")

    print(f"  [{folder}]", end=" ", flush=True)

    # load original at model input resolution for SAM
    orig_bgr = cv2.imread(orig_path)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # load raw anomaly map
    amap = np.load(npy_path)  # (H, W) at model resolution

    # resize anomaly map to match original image if needed
    if amap.shape[:2] != orig_rgb.shape[:2]:
        amap = cv2.resize(amap, (orig_rgb.shape[1], orig_rgb.shape[0]))

    # run SAM
    masks = mask_generator.generate(orig_rgb)
    print(f"→ {len(masks)} segments", end=" ", flush=True)

    if len(masks) == 0:
        print("(no masks, skipping)")
        continue

    # score each segment by mean anomaly value
    for m in masks:
        seg_mask = m['segmentation']   # bool (H, W)
        m['anomaly_score'] = float(amap[seg_mask].mean())

    # sort by anomaly score, highest first
    masks_sorted = sorted(masks, key=lambda x: x['anomaly_score'], reverse=True)

    # pick top segments: all that are above the 90th percentile of segment scores
    scores = np.array([m['anomaly_score'] for m in masks_sorted])
    threshold = np.percentile(scores, 75)
    top_masks = [m for m in masks_sorted if m['anomaly_score'] >= threshold]
    # at least 1, at most 5
    top_masks = top_masks[:5] if len(top_masks) > 5 else top_masks
    if len(top_masks) == 0:
        top_masks = [masks_sorted[0]]

    # build combined mask of top anomalous segments
    combined = np.zeros(orig_rgb.shape[:2], dtype=bool)
    for m in top_masks:
        combined |= m['segmentation']

    # ── visualization ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(orig_rgb.shape[1]/100, orig_rgb.shape[0]/100), dpi=100)

    # dim the non-anomalous regions
    display = orig_rgb.copy().astype(np.float32)
    display[~combined] *= 0.3   # darken background

    # draw red contours around anomalous segments
    contour_img = display.astype(np.uint8)
    mask_uint8 = combined.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)

    # overlay anomaly heatmap only on the top segments
    amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    heatmap_rgba = plt.cm.jet(amap_norm)[:, :, :3]
    heatmap_rgba = (heatmap_rgba * 255).astype(np.uint8)

    overlay = contour_img.copy()
    alpha = 0.4
    overlay[combined] = (
        (1 - alpha) * contour_img[combined].astype(np.float32) +
        alpha * heatmap_rgba[combined].astype(np.float32)
    ).astype(np.uint8)

    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"→ {len(top_masks)} top segments → saved")

print("\nDone. sam_overlay.png added to each frame folder.")
