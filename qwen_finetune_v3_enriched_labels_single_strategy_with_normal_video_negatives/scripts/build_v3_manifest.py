#!/usr/bin/env python3
"""
v3 manifest builder.

Combines TWO sources of training rows:
  (A) 63 anomalous videos × ONE frame-sampling strategy (human_span_smart)
      using the enriched 2-3 sentence labels from v2.
      → exactly 63 rows, one per anomalous video.

  (B) 30 normal training videos (sampled deterministically from the 175 normal
      videos in the rtfm train list), scored with RTFM locally on CPU, with
      ~5 frames sampled uniformly across the video. Assistant label is a
      canonical "no anomalous activity visible" sentence.
      → exactly 30 rows.

Total: 93 rows. Images are extracted from the per-frame JPEG dump at
data/SHANGHAI/SHANGHAI_TRAIN/frames/<vid>/0001.jpg .. NNNN.jpg.

Output:
  out_v3/manifest.jsonl
  out_v3/images/anom/<vid>/...jpg
  out_v3/images/normal/<vid>/...jpg
  out_v3/build_summary.json
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# ── paths ───────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
V3   = Path(__file__).resolve().parents[1]
V2   = REPO / "qwen_finetune_v2"
SFT_V1_OUT = REPO / "qwen_finetune_colab" / "out"     # has manifest + extracted anom frames

ANOM_ANNOT_FILE   = REPO / "data/SHANGHAI/SHANGHAI_TRAIN/videos_anomalous_train_with_human_annotations/Anomalous_train_annotations.json"
ENRICHED_FILE     = V3   / "data/enriched_train_annotations.json"
TRAIN_FRAMES_DIR  = REPO / "data/SHANGHAI/SHANGHAI_TRAIN/frames"
RTFM_TRAIN_LIST   = REPO / "rtfm/list/shanghai-i3d-train-10crop.list"
RTFM_FEATURES_DIR = REPO / "rtfm/data/SH_Train_ten_crop_i3d"
RTFM_CKPT         = REPO / "rtfm/ckpt/rtfm_best.pkl"

OUT_DIR        = V3 / "out_v3"
OUT_IMG_DIR    = OUT_DIR / "images"
OUT_MANIFEST   = OUT_DIR / "manifest.jsonl"
OUT_SUMMARY    = OUT_DIR / "build_summary.json"

SEED = 42
N_NORMAL = 30
FRAMES_PER_NORMAL = 5  # uniform sampling
ANOM_STRATEGY = "human_span_smart"  # single strategy

# Make RTFM imports work
sys.path.insert(0, str(REPO / "rtfm"))
sys.path.insert(0, str(REPO / "qwen_finetune_colab"))

# Same system prompt as v1/v2 — we keep it stable across versions
SYSTEM_PROMPT = """You are a surveillance video anomaly analyst. You will be shown a set of \
frames sampled from a surveillance video that has been flagged as anomalous \
by a weakly-supervised anomaly detection model (RTFM).

The frames are ordered temporally. Each frame comes from a specific temporal \
snippet of the video, and you are given the anomaly score for that snippet \
(0 = normal, 1 = highly anomalous).

The frames were specifically selected from the anomalous portions of the \
video — they represent the onset, peak, and resolution of the detected anomaly.

Your task: Based on ALL the frames and their anomaly scores together, provide \
a single concise explanation (2-3 sentences) of what anomalous activity is \
happening. Focus on:
- WHAT is happening (the specific anomalous activity)
- WHO/WHAT is involved (people, vehicles, objects — describe appearance)
- WHEN in the sequence it starts and ends
- WHY it is anomalous (how it deviates from normal pedestrian behaviour)

Respond with ONLY a JSON object in this exact format:
{"explanation": "..."}"""

NORMAL_LABEL_TEMPLATES = [
    "No anomalous activity is visible in this video. All pedestrians are walking normally in the pedestrian area, and there are no signs of cycling, skateboarding, fighting, or any other restricted behaviour.",
    "There is no anomaly in this video. Pedestrians are walking calmly through the area in a manner consistent with normal pedestrian flow, with no vehicles, cyclists, or aggressive interactions present.",
    "The video shows only normal pedestrian activity. People are walking at typical speeds through the public space, and no behaviour deviating from normal pedestrian use of the area is observed.",
]


def user_prompt_from_scores(scores):
    s = ", ".join(f"{x:.4f}" for x in scores)
    return (f"Anomaly scores for each shown frame (temporal order, RTFM logits): [{s}]\n"
            "Describe the anomalous activity.")


def uniform_sample_indices(n_total, k):
    if n_total <= k:
        return list(range(n_total))
    if k == 1:
        return [n_total // 2]
    return sorted({int(round(i * (n_total - 1) / (k - 1))) for i in range(k)})


def load_rtfm():
    """Load RTFM model on CPU."""
    from model import Model
    from option import parser as rtfm_parser
    args = rtfm_parser.parse_args([])
    device = torch.device("cpu")
    net = Model(args.feature_size, args.batch_size)
    net.load_state_dict(torch.load(str(RTFM_CKPT), map_location=device))
    net.eval().to(device)
    return net, device


def score_video(net, device, feat_path: Path):
    x = np.load(str(feat_path), allow_pickle=True).astype(np.float32)
    t = torch.from_numpy(x).unsqueeze(0).to(device).permute(0, 2, 1, 3)
    with torch.no_grad():
        out = net(inputs=t)
    logits = torch.squeeze(out[6], 1)
    seg = torch.mean(logits, 0).squeeze().cpu().numpy()
    if seg.ndim == 0:
        seg = seg.reshape(1)
    return [float(s) for s in seg]


def feature_for(vid: str) -> Path | None:
    p = RTFM_FEATURES_DIR / f"{vid}_i3d.npy"
    return p if p.is_file() else None


def jpeg_for_frame(vid: str, frame_num: int) -> Path | None:
    # Frame files are 0001.jpg, 0002.jpg, ... (1-indexed)
    fname = f"{frame_num:04d}.jpg"
    p = TRAIN_FRAMES_DIR / vid / fname
    return p if p.is_file() else None


def count_frames(vid: str) -> int:
    d = TRAIN_FRAMES_DIR / vid
    if not d.is_dir():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() == ".jpg")


# ── main ────────────────────────────────────────────────────────────────────

def build_anom_rows() -> list[dict]:
    """Reuse v1 manifest rows for the chosen single strategy, swap in v2 enriched labels."""
    enriched = {r["video_id"]: r["explanation"] for r in json.loads(ENRICHED_FILE.read_text())}
    print(f"Loaded {len(enriched)} enriched labels")

    src_manifest = SFT_V1_OUT / "manifest.jsonl"
    rows = []
    with open(src_manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("strategy") != ANOM_STRATEGY:
                continue
            vid = r["video_id"]
            new_label = enriched.get(vid)
            if new_label is None:
                print(f"  WARN: no enriched label for {vid}, skipping")
                continue
            # Copy images from v1 out/images/<vid>/<strategy>/ to v3 out/images/anom/<vid>/
            new_image_paths = []
            for rel in r["images"]:
                src = SFT_V1_OUT / rel
                if not src.is_file():
                    print(f"  WARN: missing {src}")
                    continue
                dest_rel = Path("images") / "anom" / vid / Path(rel).name
                dest_abs = OUT_DIR / dest_rel
                dest_abs.parent.mkdir(parents=True, exist_ok=True)
                if not dest_abs.exists():
                    shutil.copy(src, dest_abs)
                new_image_paths.append(str(dest_rel).replace("\\", "/"))

            row = {
                "id":           f"anom_{vid}",
                "video_id":     vid,
                "video_type":   "anomalous",
                "strategy":     ANOM_STRATEGY,
                "system":       SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value":
                        "\n".join("<image>" for _ in new_image_paths) + "\n" +
                        user_prompt_from_scores(r["scores"])},
                    {"from": "assistant", "value":
                        json.dumps({"explanation": new_label}, ensure_ascii=False)},
                ],
                "images":         new_image_paths,
                "frame_indices":  r.get("frame_indices", []),
                "scores":         r["scores"],
            }
            rows.append(row)
    return rows


def build_normal_rows(net, device) -> list[dict]:
    """Score 30 normal videos with RTFM, sample frames uniformly, write rows."""
    anom_ids = {r["video_id"] for r in json.loads(ANOM_ANNOT_FILE.read_text())}

    # Read train list, take only normal videos that have BOTH features and frames
    candidates = []
    with open(RTFM_TRAIN_LIST) as f:
        for line in f:
            vid = Path(line.strip()).name.replace("_i3d.npy", "")
            if vid in anom_ids:
                continue
            if not feature_for(vid):
                continue
            n = count_frames(vid)
            if n < 50:  # need enough frames to subsample
                continue
            candidates.append((vid, n))

    print(f"Normal candidates with features + ≥50 frames: {len(candidates)}")

    rnd = random.Random(SEED)
    rnd.shuffle(candidates)
    picked = candidates[:N_NORMAL]
    print(f"Picked {len(picked)} normal videos for training negatives")

    rows = []
    for i, (vid, n_frames) in enumerate(picked, 1):
        scores = score_video(net, device, feature_for(vid))
        T = len(scores)
        # Sample frame indices uniformly across the video length
        frame_idx_set = uniform_sample_indices(n_frames, FRAMES_PER_NORMAL)
        # Map each frame index to a snippet index for the score
        # RTFM uses 16-frame snippets evenly distributed over T snippets
        # snippet_for_frame ≈ frame_num / n_frames * T
        sample_scores = []
        copied_images = []
        for fnum in frame_idx_set:
            jpeg = jpeg_for_frame(vid, fnum + 1)  # 1-indexed jpegs
            if not jpeg:
                continue
            dest_rel = Path("images") / "normal" / vid / f"f{fnum:05d}.jpg"
            dest_abs = OUT_DIR / dest_rel
            dest_abs.parent.mkdir(parents=True, exist_ok=True)
            if not dest_abs.exists():
                shutil.copy(jpeg, dest_abs)
            copied_images.append(str(dest_rel).replace("\\", "/"))
            snip = min(T - 1, int(fnum / max(n_frames, 1) * T))
            sample_scores.append(scores[snip])

        if not copied_images:
            print(f"  [{i}/{N_NORMAL}] {vid}: no jpegs found, skipping")
            continue

        # Pick a label template — rotate so we get all 3 in roughly equal mix
        label = NORMAL_LABEL_TEMPLATES[i % len(NORMAL_LABEL_TEMPLATES)]
        row = {
            "id":           f"normal_{vid}",
            "video_id":     vid,
            "video_type":   "normal",
            "strategy":     "uniform_sample_for_normal",
            "system":       SYSTEM_PROMPT,
            "conversations": [
                {"from": "human", "value":
                    "\n".join("<image>" for _ in copied_images) + "\n" +
                    user_prompt_from_scores(sample_scores)},
                {"from": "assistant", "value":
                    json.dumps({"explanation": label}, ensure_ascii=False)},
            ],
            "images":         copied_images,
            "frame_indices":  list(frame_idx_set),
            "scores":         sample_scores,
        }
        rows.append(row)
        max_score = max(sample_scores) if sample_scores else 0.0
        print(f"  [{i}/{N_NORMAL}] {vid}: T={T}, n_frames={n_frames}, max_score={max_score:.3f}, kept {len(copied_images)} imgs")
    return rows


def main():
    print("=== Step A: building anomalous rows (single strategy, enriched labels) ===")
    anom_rows = build_anom_rows()
    print(f"  → {len(anom_rows)} anomalous rows")

    print()
    print("=== Step B: loading RTFM model for normal-video scoring ===")
    net, device = load_rtfm()

    print()
    print("=== Step C: building normal rows ===")
    normal_rows = build_normal_rows(net, device)
    print(f"  → {len(normal_rows)} normal rows")

    all_rows = anom_rows + normal_rows
    rnd = random.Random(SEED)
    rnd.shuffle(all_rows)  # interleave anom and normal in the manifest

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_rows":         len(all_rows),
        "anomalous_rows":     len(anom_rows),
        "normal_rows":        len(normal_rows),
        "anom_strategy":      ANOM_STRATEGY,
        "frames_per_normal":  FRAMES_PER_NORMAL,
        "seed":               SEED,
        "anom_video_ids":     sorted(r["video_id"] for r in anom_rows),
        "normal_video_ids":   sorted(r["video_id"] for r in normal_rows),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print()
    print(f"Wrote manifest: {OUT_MANIFEST}  ({len(all_rows)} rows = {len(anom_rows)} anom + {len(normal_rows)} normal)")
    print(f"Wrote summary:  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
