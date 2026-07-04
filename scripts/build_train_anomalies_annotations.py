#!/usr/bin/env python3
"""
Build train_annotations.json with the same schema as test anomalous_videos/annotations.json:

  [
    {
      "video_id": "01_0014",
      "explanation": "",
      "anomaly_start_frame": 12,
      "anomaly_end_frame": 198
    },
    ...
  ]

- Includes only training clips flagged anomalous in SHANGHAI_train.txt (last column == 1).
- anomaly_* are derived from SHANGHAI_TRAIN/label/<video_id>.npy (first / last index where label > 0).
- explanation is left empty for you to fill (test file uses human-written strings).

Writes the canonical file under data/SHANGHAI/anomalous_videos/ and mirrors the same JSON next to
your train anomaly MP4s:
  .../videos_anomalous_train_with_human_annotations/Anomalous_train_annotations.json
  .../videos_anomalous_train_with_human_annotations/Anomalous_train_annotations

Usage:
  python build_train_anomalies_annotations.py
  python build_train_anomalies_annotations.py --out /path/to/train_annotations.json
  python build_train_anomalies_annotations.py --no-copy-beside-videos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TXT = PROJECT_ROOT / "data" / "SHANGHAI" / "SHANGHAI_TRAIN" / "SHANGHAI_train.txt"
DEFAULT_LABEL_DIR = PROJECT_ROOT / "data" / "SHANGHAI" / "SHANGHAI_TRAIN" / "label"
DEFAULT_OUT = (
    PROJECT_ROOT / "data" / "SHANGHAI" / "anomalous_videos" / "train_annotations.json"
)
# Same schema, kept next to train anomaly MP4s for convenience
DEFAULT_OUT_ALONGSIDE_VIDEOS = (
    PROJECT_ROOT
    / "data"
    / "SHANGHAI"
    / "SHANGHAI_TRAIN"
    / "videos_anomalous_train_with_human_annotations"
    / "Anomalous_train_annotations.json"
)


def anomaly_span_from_label(label_path: Path) -> tuple[int | None, int | None]:
    if not label_path.is_file():
        return None, None
    arr = np.load(label_path)
    if arr.size == 0:
        return None, None
    pos = np.flatnonzero(arr > 0)
    if pos.size == 0:
        return None, None
    return int(pos.min()), int(pos.max())


def parse_train_txt(path: Path) -> list[tuple[str, int, int]]:
    """Return list of (video_id, num_frames_from_txt, is_anomaly)."""
    rows: list[tuple[str, int, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                print(f"skip malformed line: {line}", file=sys.stderr)
                continue
            rel = parts[0]
            vid = rel.rstrip("/").split("/")[-1]
            num_frames = int(parts[2])
            is_anom = int(parts[3])
            rows.append((vid, num_frames, is_anom))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-txt", type=Path, default=DEFAULT_TRAIN_TXT)
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--no-copy-beside-videos",
        action="store_true",
        help="Do not also write alongside train anomaly MP4s",
    )
    args = p.parse_args()

    if not args.train_txt.is_file():
        print(f"Not found: {args.train_txt}", file=sys.stderr)
        sys.exit(1)

    entries: list[dict] = []
    for vid, _num_txt, is_anom in parse_train_txt(args.train_txt):
        if is_anom != 1:
            continue
        label_path = args.label_dir / f"{vid}.npy"
        start, end = anomaly_span_from_label(label_path)
        entries.append(
            {
                "video_id": vid,
                "explanation": "",
                "anomaly_start_frame": start,
                "anomaly_end_frame": end,
            }
        )

    entries.sort(key=lambda e: e["video_id"])
    text = json.dumps(entries, indent=2) + "\n"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    if not args.no_copy_beside_videos:
        alongside = DEFAULT_OUT_ALONGSIDE_VIDEOS
        alongside.parent.mkdir(parents=True, exist_ok=True)
        alongside.write_text(text, encoding="utf-8")
        # Extension-less name you use in Finder / editors
        alongside.with_name("Anomalous_train_annotations").write_text(
            text, encoding="utf-8"
        )

    n_span = sum(1 for e in entries if e["anomaly_start_frame"] is not None)
    print(f"Wrote {len(entries)} train anomaly entries -> {args.out}")
    if not args.no_copy_beside_videos:
        print(f"  mirrored -> {DEFAULT_OUT_ALONGSIDE_VIDEOS}")
        print(f"  mirrored -> {DEFAULT_OUT_ALONGSIDE_VIDEOS.with_name('Anomalous_train_annotations')}")
    print(f"  ({n_span} with non-null anomaly span from label .npy)")


if __name__ == "__main__":
    main()
