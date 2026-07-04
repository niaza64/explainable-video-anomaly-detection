#!/usr/bin/env python3
"""
Extract and run video-level evaluation (Cells 7-8 from Freeman_style.ipynb)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
sys.path.append("/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae")

import torch
import cv2 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from model.model_factory import mae_cvt_patch8
from util.abnormal_utils import filt
from sklearn.metrics import roc_auc_score

# Load model (from Cell 1)
print("Loading model...")
teacher_ckpt = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/shanghai/checkpoint-best.pth"
student_ckpt = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/models/aed-mae/experiments/shanghai/checkpoint-best-student.pth"

model = mae_cvt_patch8(
    norm_pix_loss=False,
    img_size=(160, 320),
    use_only_masked_tokens_ab=False,
    abnormal_score_func=["l1", "L1"],
    masking_method="random_masking",
    grad_weighted_loss=True
).float()

teacher = torch.load(teacher_ckpt, map_location='cpu', weights_only=False)["model"]
student = torch.load(student_ckpt, map_location='cpu', weights_only=False)["model"]

for key in student.keys():
    if "student" in key:
        teacher[key] = student[key]

model.load_state_dict(teacher, strict=False)
model.eval()
model.train_TS = True
model.abnormal_score_func = "L1"

print("Model loaded!\n")

# Preprocessing function (from Cell 4)
def load_and_prcoess_frame(frame_idx, video_dir):
    prev_idx = frame_idx - 3
    next_idx = frame_idx + 3

    prev_frame = cv2.imread(os.path.join(video_dir, f"{prev_idx:03d}.jpg"))
    frame = cv2.imread(os.path.join(video_dir, f"{frame_idx:03d}.jpg"))
    next_frame = cv2.imread(os.path.join(video_dir, f"{next_idx:03d}.jpg"))

    # Resize
    prev_frame = cv2.resize(prev_frame, (320, 160))
    frame = cv2.resize(frame, (320, 160))
    next_frame = cv2.resize(next_frame, (320, 160))

    # BGR to RGB
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    next_frame_rgb = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

    # Normalize
    prev_frame = (prev_frame_rgb.astype(np.float32) - 127.5) / 127.5
    frame = (frame_rgb.astype(np.float32) - 127.5) / 127.5
    next_frame = (next_frame_rgb.astype(np.float32) - 127.5) / 127.5

    # Motion gradient
    motion_grad =  np.abs(prev_frame - next_frame)
    motion_grad_avg = np.mean(motion_grad, axis=2, keepdims=True).astype(np.float32)

    # To tensors
    prev_frame = torch.from_numpy(prev_frame).unsqueeze(0).permute(0, 3, 1, 2)
    frame = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)
    next_frame = torch.from_numpy(next_frame).unsqueeze(0).permute(0, 3, 1, 2)
    motion_grad_avg = torch.from_numpy(motion_grad_avg).unsqueeze(0).permute(0, 3, 1, 2)

    frames = torch.cat((prev_frame, frame, next_frame), dim=1)
  
    return {
        'frames': frames,
        'motion_grad_avg': motion_grad_avg,
        'curr_rgb': frame_rgb,
    }

# CELL 7: Process entire video
print("="*70)
print("STEP 5/7: PROPER VIDEO-LEVEL EVALUATION")
print("="*70)

video_dir = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/SHANGHAI/SHANGHAI_Test/frames/01_0025"
gt_path = "/Users/niazahmad/Desktop/Research/explainable-video-anomaly-detection/SHANGHAI/SHANGHAI_Test/label/01_0025.npy"

gt = np.load(gt_path)
num_frames = len(gt)

print(f"\nVideo: 01_0025")
print(f"  Total frames: {num_frames}")
print(f"  Anomaly frames: {gt.sum()} ({gt.sum()/num_frames*100:.1f}%)")
print(f"  Processable frames: {num_frames-6} (skipping first/last 3 for neighbors)")

all_scores_teacher = []

print("\n🔄 Processing frames (this will take 1-2 minutes)...")
for frame_idx in range(3, num_frames - 3):
    if frame_idx % 50 == 0:
        print(f"  Progress: {frame_idx}/{num_frames} frames...")
    
    data = load_and_prcoess_frame(frame_idx, video_dir)
    
    with torch.no_grad():
        model.eval()
        frame = data['frames'][:, 3:6, :, :]
        zeror_anamoly_target = torch.zeros(1, 1, 160, 320)
        target = torch.cat([frame, zeror_anamoly_target], dim=1)
        
        loss, pred_teacher, mask, scores = model(
            data['frames'],
            target,
            data['motion_grad_avg'],
            mask_ratio=0.75,
        )
        
        all_scores_teacher.append(scores[1].item())

scores_teacher = np.array(all_scores_teacher)

print(f"\n✅ Collected scores for {len(scores_teacher)} frames")
print(f"   Raw score range: [{scores_teacher.min():.4f}, {scores_teacher.max():.4f}]")

# CELL 8: Apply temporal smoothing and normalization
print("\n" + "="*70)
print("STEP 6/7: APPLYING PAPER'S EVALUATION METHOD")
print("="*70)

print("\n📊 Step 1: Temporal smoothing with filt(range=900, mu=282)...")
smoothed_scores = filt(scores_teacher, dim=9, range=900, mu=282)

print("📊 Step 2: Min-max normalization...")
smoothed_scores = (smoothed_scores - smoothed_scores.min()) / (smoothed_scores.max() - smoothed_scores.min())

print(f"\n   Smoothed score range: [{smoothed_scores.min():.4f}, {smoothed_scores.max():.4f}]")

# Check specific frames
print("\n" + "="*70)
print("FRAME-LEVEL SCORES (After Proper Processing):")
print("="*70)
frame_130_score = smoothed_scores[130-3]
frame_175_score = smoothed_scores[175-3]
print(f"Frame 130 (NORMAL):  {frame_130_score:.4f}")
print(f"Frame 175 (ANOMALY): {frame_175_score:.4f}")
print(f"Difference: {frame_175_score - frame_130_score:.4f} ({(frame_175_score/frame_130_score - 1)*100:+.1f}%)")

# Calculate AUC
gt_subset = gt[3:num_frames-3]
auc = roc_auc_score(gt_subset, smoothed_scores)

print(f"\n📈 VIDEO-LEVEL PERFORMANCE:")
print(f"   AUC: {auc:.4f}")
print("="*70)

# Visualize
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(smoothed_scores, linewidth=2, color='blue', label='Anomaly Score')
plt.axvline(x=130-3, color='green', linestyle='--', alpha=0.7, label='Frame 130 (Normal)')
plt.axvline(x=175-3, color='red', linestyle='--', alpha=0.7, label='Frame 175 (Anomaly)')

anomaly_indices = np.where(gt_subset == 1)[0]
if len(anomaly_indices) > 0:
    plt.fill_between(range(len(smoothed_scores)), 0, 1, 
                      where=gt_subset==1, alpha=0.2, color='red', label='Ground Truth Anomalies')

plt.xlabel('Frame Number', fontsize=12)
plt.ylabel('Normalized Anomaly Score', fontsize=12)
plt.title(f'Video 01_0025 - Full Timeline (AUC: {auc:.4f})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
zoom_start = max(0, 148-3-20)
zoom_end = min(len(smoothed_scores), 201-3+20)
plt.plot(range(zoom_start, zoom_end), smoothed_scores[zoom_start:zoom_end], 
         linewidth=2, color='blue')
plt.axvline(x=130-3, color='green', linestyle='--', alpha=0.7, label='Frame 130 (Normal)')
plt.axvline(x=175-3, color='red', linestyle='--', alpha=0.7, label='Frame 175 (Anomaly)')
plt.fill_between(range(zoom_start, zoom_end), 0, 1, 
                  where=gt_subset[zoom_start:zoom_end]==1, alpha=0.2, color='red')
plt.xlabel('Frame Number', fontsize=12)
plt.ylabel('Normalized Anomaly Score', fontsize=12)
plt.title('Zoomed View: Anomaly Region', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('day1_video_level_scores.png', dpi=150, bbox_inches='tight')
print("\n💾 Saved: day1_video_level_scores.png")

print("\n✅ STEP 6 COMPLETE")
print("="*70)

print("\n🎯 DAY 1 COMPLETE!")
print("Check 'day1_video_level_scores.png' for the visualization")
