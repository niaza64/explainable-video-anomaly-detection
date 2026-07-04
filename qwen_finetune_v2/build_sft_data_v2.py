#!/usr/bin/env python3
"""
Build SFT pack: RTFM scores + pooled frames + annotations → images/ + manifest.jsonl + summary.json.

Run from anywhere:
  python path/to/explainable-video-anomaly-detection/qwen_finetune_colab/build_sft_data.py --out-dir ./out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

COLAB = Path(__file__).resolve().parent
REPO = COLAB.parent
sys.path.insert(0, str(COLAB))
sys.path.insert(0, str(REPO / "rtfm"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from model import Model  # noqa: E402
from option import parser as rtfm_parser  # noqa: E402

from pool import (  # noqa: E402
    STRATEGIES,
    fallback_snippet_closest_to_interval,
    snippets_in_human_window,
)

DEFAULT_VIDEO_ROOT = (
    REPO
    / "data"
    / "SHANGHAI"
    / "SHANGHAI_TRAIN"
    / "videos_anomalous_train_with_human_annotations"
)
DEFAULT_ANNOTATIONS = DEFAULT_VIDEO_ROOT / "Anomalous_train_annotations.json"
DEFAULT_VIDEOS_DIR = DEFAULT_VIDEO_ROOT
DEFAULT_TRAIN_LIST = REPO / "rtfm" / "list" / "shanghai-i3d-train-10crop.list"
DEFAULT_FEATURES = REPO / "rtfm" / "data" / "SH_Train_ten_crop_i3d"
DEFAULT_CKPT = REPO / "rtfm" / "ckpt" / "rtfm_best.pkl"

# Same text as pipeline/local_qwen_caller.py SYSTEM_PROMPT (L48–68); keep in sync manually.
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


def user_prompt_from_scores(scores: list[float]) -> str:
    s = ", ".join(f"{x:.4f}" for x in scores)
    return (
        f"Anomaly scores for each shown frame (temporal order, RTFM logits): [{s}]\n"
        "Describe the anomalous activity."
    )


def subsample_pairs(pairs: list[tuple[int, float]], max_n: int) -> list[tuple[int, float]]:
    """Uniformly subsample in list order (temporal) for VLM token / VRAM limits."""
    n = len(pairs)
    if n <= max_n:
        return pairs
    if max_n == 1:
        idx = [0]
    else:
        idx = sorted({int(round(i * (n - 1) / (max_n - 1))) for i in range(max_n)})
    return [pairs[i] for i in idx]


def write_manifest_train_val_splits(
    manifest_path: Path,
    val_frac: float,
    seed: int,
) -> None:
    """Split manifest.jsonl by video_id; write manifest_train.jsonl, manifest_eval.jsonl, manifest_split.json."""
    rows: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        print("write_manifest_train_val_splits: empty manifest, skipping")
        return
    by_vid: dict[str, list[dict]] = {}
    for r in rows:
        by_vid.setdefault(r["video_id"], []).append(r)
    vids = sorted(by_vid.keys())
    rnd = random.Random(seed)
    rnd.shuffle(vids)
    n_val = max(1, int(round(len(vids) * val_frac)))
    val_set = set(vids[:n_val])
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for vid, rs in by_vid.items():
        (eval_rows if vid in val_set else train_rows).extend(rs)
    out_dir = manifest_path.parent
    mt = out_dir / "manifest_train.jsonl"
    me = out_dir / "manifest_eval.jsonl"
    ms = out_dir / "manifest_split.json"
    with open(mt, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(me, "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ms.write_text(
        json.dumps(
            {
                "val_frac": val_frac,
                "seed": seed,
                "n_videos": len(vids),
                "n_val_videos": len(val_set),
                "val_video_ids": sorted(val_set),
                "train_video_ids": sorted(set(vids) - val_set),
                "n_train_rows": len(train_rows),
                "n_eval_rows": len(eval_rows),
                "manifest_train": str(mt),
                "manifest_eval": str(me),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote train/val manifests → {mt.name}, {me.name}, {ms.name}")


def load_rtfm(ckpt: Path):
    args = rtfm_parser.parse_args([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Model(args.feature_size, args.batch_size)
    net.load_state_dict(torch.load(str(ckpt), map_location=device))
    net.eval().to(device)
    return net, device


def score_video(net, device: torch.device, feat_path: Path) -> list[float]:
    x = np.load(str(feat_path), allow_pickle=True).astype(np.float32)
    t = torch.from_numpy(x).unsqueeze(0).to(device).permute(0, 2, 1, 3)
    with torch.no_grad():
        out = net(inputs=t)
    logits = torch.squeeze(out[6], 1)
    seg = torch.mean(logits, 0).squeeze().cpu().numpy()
    if seg.ndim == 0:
        seg = seg.reshape(1)
    return [float(s) for s in seg]


def resolve_feature_path(line: str, root: Path) -> Path | None:
    line = line.strip()
    if not line:
        return None
    p = Path(line)
    if p.is_file():
        return p
    c = root / p.name
    return c if c.is_file() else None


def feature_index(train_list: Path, root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with open(train_list, encoding="utf-8") as f:
        for line in f:
            p = resolve_feature_path(line, root)
            if p is None:
                continue
            vid = p.name.replace("_i3d.npy", "").replace(".npy", "")
            out[vid] = p
    return out


def frame_count(mp4: Path) -> int:
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, n)


def save_frame(mp4: Path, frame_idx: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), bgr)


def run_strategy(
    name: str,
    I: list[int],
    scores: list[float],
    F: int,
    T: int,
    *,
    delta: int,
    snippet_budget: int,
    min_gap: int,
) -> list[tuple[int, float]]:
    fn = STRATEGIES[name]
    if name == "human_span_smart":
        return fn(I, scores, F, T, budget=snippet_budget, min_gap=min_gap)
    if name in ("every_snippet_mid_frame_band", "top3_snippets_mid_frame_band"):
        return fn(I, scores, F, T, delta=delta)
    return fn(I, scores, F, T)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build SFT data from SFT_FRAME_POOLING.md strategies")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument(
        "--videos-dir",
        type=Path,
        default=DEFAULT_VIDEOS_DIR,
        help="Directory of train .mp4 files (default: …/SHANGHAI_TRAIN/videos_anomalous_train_with_human_annotations)",
    )
    ap.add_argument("--train-list", type=Path, default=DEFAULT_TRAIN_LIST)
    ap.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES.keys()),
        help="Subset of: " + " ".join(STRATEGIES.keys()),
    )
    ap.add_argument("--delta", type=int, default=2, help="±δ frames around snippet mid (strategies 3 and 5)")
    ap.add_argument("--snippet-budget", type=int, default=8, help="Max snippets selected for human_span_smart")
    ap.add_argument("--min-gap", type=int, default=2, help="Min snippet-index gap for human_span_smart")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Cap frames per row after pooling (uniform subsample); use for VLM training memory limits",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Only re-run pooling + RTFM + manifest.jsonl / summary.json; do not write JPEGs (use after a full build)",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=None,
        metavar="F",
        help=(
            "If set (e.g. 0.15), after building manifest.jsonl write manifest_train.jsonl, "
            "manifest_eval.jsonl, and manifest_split.json by video_id (same schema as manifest rows)"
        ),
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for --val-frac video shuffle",
    )
    args = ap.parse_args()

    for n in args.strategies:
        if n not in STRATEGIES:
            sys.exit(f"Unknown strategy: {n}. Choose from {list(STRATEGIES)}")

    idx = feature_index(args.train_list, args.features_root)
    with open(args.annotations, encoding="utf-8") as f:
        rows = json.load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    net, device = load_rtfm(args.checkpoint)

    written = 0
    skipped: list[str] = []

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for rec in rows:
            vid = rec["video_id"]
            text = (rec.get("explanation") or "").strip()
            if not text:
                skipped.append(f"{vid}: empty explanation")
                continue
            fp = idx.get(vid)
            if fp is None:
                skipped.append(f"{vid}: no feature file")
                continue
            mp4 = args.videos_dir / f"{vid}.mp4"
            if not mp4.is_file():
                skipped.append(f"{vid}: missing mp4")
                continue
            F = frame_count(mp4)
            if F <= 0:
                skipped.append(f"{vid}: no frames")
                continue
            a0, a1 = int(rec["anomaly_start_frame"]), int(rec["anomaly_end_frame"])

            seg = score_video(net, device, fp)
            T = len(seg)

            I = snippets_in_human_window(a0, a1, F, T)
            if not I:
                I = fallback_snippet_closest_to_interval(a0, a1, F, T)

            seen_lists: set[tuple[int, ...]] = set()
            for strat in args.strategies:
                pairs = run_strategy(
                    strat,
                    I,
                    seg,
                    F,
                    T,
                    delta=args.delta,
                    snippet_budget=args.snippet_budget,
                    min_gap=args.min_gap,
                )
                if not pairs:
                    continue
                if args.max_frames is not None and len(pairs) > args.max_frames:
                    pairs = subsample_pairs(pairs, args.max_frames)
                key = tuple(f for f, _ in pairs)
                if key in seen_lists:
                    continue
                seen_lists.add(key)

                rel_imgs: list[str] = []
                sc_list: list[float] = []
                for j, (fi, sc) in enumerate(pairs):
                    rel = Path("images") / vid / strat / f"f{j:03d}_frm{fi:05d}.jpg"
                    dest = args.out_dir / rel
                    if args.no_images:
                        rel_imgs.append(str(rel).replace("\\", "/"))
                        sc_list.append(sc)
                        continue
                    if not save_frame(mp4, fi, dest):
                        skipped.append(f"{vid}: extract fail frame={fi}")
                        break
                    rel_imgs.append(str(rel).replace("\\", "/"))
                    sc_list.append(sc)
                else:
                    ph = "\n".join("<image>" for _ in rel_imgs)
                    record = {
                        "id": f"{vid}_{strat}",
                        "video_id": vid,
                        "strategy": strat,
                        "system": SYSTEM_PROMPT,
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"{ph}\n{user_prompt_from_scores(sc_list)}",
                            },
                            {
                                "from": "assistant",
                                "value": json.dumps({"explanation": text}, ensure_ascii=False),
                            },
                        ],
                        "images": rel_imgs,
                        "frame_indices": [p[0] for p in pairs],
                        "scores": sc_list,
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    summary = {
        "written_rows": written,
        "skipped": skipped,
        "manifest": str(manifest_path),
        "no_images": args.no_images,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {written} rows → {manifest_path}")
    if skipped:
        print(f"Skipped / warnings: {len(skipped)} (see summary.json)")
    if args.val_frac is not None:
        if not 0 < args.val_frac < 1:
            sys.exit("--val-frac must be in (0, 1)")
        write_manifest_train_val_splits(manifest_path, args.val_frac, args.split_seed)


if __name__ == "__main__":
    main()
