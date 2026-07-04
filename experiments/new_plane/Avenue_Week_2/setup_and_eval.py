"""
Avenue pipeline — end-to-end:
  1. Extract frames from testing_videos/*.avi  →  test/frames/{01..21}/
  2. Compute temporal gradients (step=1)       →  test/gradients2/{01..21}/
  3. Sample 50 frames across the 21 videos
  4. Run inference (official code, Avenue config)
  5. Save original.png / heatmap.png / overlay.png  →  eval_50_frames/

Avenue-specific differences from Shanghai:
  - Model: mae_cvt_patch16   (patch size 16, input 320×640)
  - Score: L2, both teacher-reconstruction + teacher-student combined
  - mask_ratio = 0.5, direction = ±3
"""

import os, sys, glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae")
from model.model_factory import mae_cvt_patch16

# ── paths ──────────────────────────────────────────────────────────────────
AVENUE_ROOT  = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/Avenue_dataset_related_Work/Avenue Dataset"
TEST_AVI_DIR = os.path.join(AVENUE_ROOT, "testing_videos")
FRAMES_ROOT  = os.path.join(AVENUE_ROOT, "test", "frames")
GRADS_ROOT   = os.path.join(AVENUE_ROOT, "test", "gradients2")
OUT_DIR      = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/new_plane/Avenue_Week_2/eval_50_frames"

TEACHER_CKPT = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/avenue/checkpoint-best.pth"
STUDENT_CKPT = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/avenue/checkpoint-best-student.pth"

INPUT_SIZE  = (320, 640)   # (H, W) — Avenue config
MASK_RATIO  = 0.5
DIRECTION   = 3
GRAD_STEP   = 1            # extract_gradients.py uses step=1 for Avenue
GT_DIR           = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/new_plane/Avenue_Week_2/Avenue_gt"
FRAMES_PER_VIDEO = 2       # 21 videos × 2-3 = ~50 frames

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract frames from AVIs
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("STEP 1: Extracting frames from test AVIs …")
print("="*60)

avis = sorted(glob.glob(os.path.join(TEST_AVI_DIR, "*.avi")))
video_frame_counts = {}

for avi_path in avis:
    vid_name = os.path.splitext(os.path.basename(avi_path))[0]   # "01", "02", …
    out_folder = os.path.join(FRAMES_ROOT, vid_name)
    os.makedirs(out_folder, exist_ok=True)

    cap = cv2.VideoCapture(avi_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_counts[vid_name] = total

    # skip if already extracted
    existing = glob.glob(os.path.join(out_folder, "*.jpg"))
    if len(existing) == total:
        print(f"  {vid_name}: already extracted ({total} frames), skipping")
        cap.release()
        continue

    print(f"  {vid_name}: extracting {total} frames …", end="", flush=True)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_folder, f"{idx:04d}.jpg"), frame)
        idx += 1
    cap.release()
    video_frame_counts[vid_name] = idx
    print(f" done ({idx})")

print("Frame extraction complete.\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute temporal gradients  (mirrors extract_gradients.py)
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("STEP 2: Computing gradients …")
print("="*60)

for vid_name in sorted(video_frame_counts.keys()):
    grad_folder = os.path.join(GRADS_ROOT, vid_name)
    os.makedirs(grad_folder, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(FRAMES_ROOT, vid_name, "*.jpg")),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    n = len(img_paths)

    existing_grads = glob.glob(os.path.join(grad_folder, "*.jpg"))
    if len(existing_grads) == n:
        print(f"  {vid_name}: gradients already exist, skipping")
        continue

    print(f"  {vid_name}: computing {n} gradients …", end="", flush=True)
    for i, img_path in enumerate(img_paths):
        prev_idx = max(0, i - GRAD_STEP)
        next_idx = min(n - 1, i + GRAD_STEP)
        prev_img = cv2.imread(img_paths[prev_idx]).astype(np.int32)
        next_img = cv2.imread(img_paths[next_idx]).astype(np.int32)
        grad = np.abs(prev_img - next_img).astype(np.uint8)
        grad_rgb = cv2.cvtColor(grad, cv2.COLOR_BGR2RGB)
        fname = os.path.basename(img_path)   # same filename as frame
        cv2.imwrite(os.path.join(grad_folder, fname),
                    cv2.cvtColor(grad_rgb, cv2.COLOR_RGB2BGR))
    print(" done")

print("Gradient computation complete.\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Load model
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("STEP 3: Loading Avenue model …")
print("="*60)

model = mae_cvt_patch16(
    norm_pix_loss=False,
    img_size=INPUT_SIZE,
    use_only_masked_tokens_ab=False,
    abnormal_score_func=["L2", "L2"],    # Avenue uses L2
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
model.abnormal_score_func_TS = "L2"    # Avenue: inference.py line 35
print("Model ready.\n")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Data loading  (mirrors test_dataset.py exactly)
# ═══════════════════════════════════════════════════════════════════════════
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

    # resize (test_dataset lines 61-64)
    if curr.shape[:2] != (H, W):
        curr = cv2.resize(curr, (W, H))
        prev = cv2.resize(prev, (W, H))
        nxt  = cv2.resize(nxt,  (W, H))
        grad = cv2.resize(grad, (W, H))

    img    = np.concatenate([prev, curr, nxt], axis=-1)                     # (H,W,9)
    mask   = np.zeros((H, W, 1), dtype=np.uint8)
    target = np.concatenate([curr, mask], axis=-1)                          # (H,W,4)

    img    = (img.astype(np.float32)    - 127.5) / 127.5
    target = (target.astype(np.float32) - 127.5) / 127.5
    grad   = grad.astype(np.float32)

    img    = np.swapaxes(img,    0, -1).swapaxes(1, -1)   # (9,H,W)
    target = np.swapaxes(target, 0, -1).swapaxes(1, -1)   # (4,H,W)
    grad   = np.swapaxes(grad,   0,  1).swapaxes(0, -1)   # (3,H,W)

    return (torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(grad).unsqueeze(0),
            torch.from_numpy(target).unsqueeze(0),
            cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(frame_dir, f"{frame_idx:04d}.jpg")),
                                    (W, H)), cv2.COLOR_BGR2RGB))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Sample 50 ANOMALOUS frames using GT labels and run inference
# ═══════════════════════════════════════════════════════════════════════════
print("="*60)
print("STEP 4/5: Running inference on 50 anomalous frames …")
print("="*60)

# Load GT labels and pick anomalous frames only
selected = []
vid_names = sorted(video_frame_counts.keys())
for i, vid_name in enumerate(vid_names):
    gt_path = os.path.join(GT_DIR, f"{vid_name}.txt")
    labels  = np.loadtxt(gt_path)                        # 0=normal, 1=anomalous

    # valid anomalous frame indices (needs ±DIRECTION neighbours)
    n = video_frame_counts[vid_name]
    anom_idx = np.where(labels == 1)[0]
    anom_idx = anom_idx[(anom_idx >= DIRECTION) & (anom_idx < n - DIRECTION)]

    if len(anom_idx) == 0:
        continue

    # pick 2 evenly-spaced frames; last few videos get 3 to reach ~50
    picks = 3 if i >= len(vid_names) - 8 else 2
    picks = min(picks, len(anom_idx))
    indices = np.linspace(0, len(anom_idx) - 1, picks, dtype=int)
    for idx in indices:
        selected.append((vid_name, int(anom_idx[idx])))

selected = selected[:50]
print(f"Selected {len(selected)} frames across {len(vid_names)} videos.\n")

os.makedirs(OUT_DIR, exist_ok=True)

for seq, (vid_name, frame_idx) in enumerate(selected, 1):
    folder_name = f"{seq:02d}_vid{vid_name}_frame{frame_idx}"
    out_folder  = os.path.join(OUT_DIR, folder_name)
    os.makedirs(out_folder, exist_ok=True)

    print(f"[{folder_name}]")

    samples, grads, targets, orig_rgb = load_sample(vid_name, frame_idx)

    # inference — same as inference.py line 38
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(samples, MASK_RATIO, grads)
        pred_stud, pred_teacher   = model.forward_decoder_TS(latent, ids_restore)

    recon_teacher = model.unpatchify(pred_teacher)   # (1,4,H,W)
    recon_student = model.unpatchify(pred_stud)      # (1,4,H,W)

    # Avenue: both terms combined (L2), matching inference.py lines 51-54
    recon_err = ((targets[:, :3] - recon_teacher[:, :3]) ** 2).mean(dim=1)  # (1,H,W)
    ts_gap    = ((recon_teacher[:, :3] - recon_student[:, :3]) ** 2).mean(dim=1)  # (1,H,W)
    amap      = (recon_err + ts_gap)[0].cpu().numpy()   # (H,W)

    amin, amax = amap.min(), amap.max()
    amap_norm  = (amap - amin) / (amax - amin + 1e-8)

    H_out, W_out = orig_rgb.shape[:2]

    # original.png
    fig, ax = plt.subplots(figsize=(W_out/100, H_out/100), dpi=100)
    ax.imshow(orig_rgb); ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_folder, "original.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # heatmap.png
    fig, ax = plt.subplots(figsize=(W_out/100, H_out/100), dpi=100)
    ax.imshow(amap_norm, cmap='gray', vmin=0, vmax=1); ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_folder, "heatmap.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # overlay.png
    fig, ax = plt.subplots(figsize=(W_out/100, H_out/100), dpi=100)
    ax.imshow(orig_rgb)
    ax.imshow(amap_norm, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(out_folder, "overlay.png"), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

print("\nDone. All frames saved to:", OUT_DIR)
