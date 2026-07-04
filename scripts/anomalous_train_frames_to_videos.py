#!/usr/bin/env python3
"""
Encode ShanghaiTech *training* clips to MP4, using **folder names only**.

Select subfolders of `frames/` whose names match `NN_MMMM`: two digits, an
underscore, then **exactly four** digits (e.g. `01_0014`). Skip folders with
only three digits after the underscore (e.g. `01_002`). This matches the
training anomaly list in SHANGHAI_train.txt (last column 1).

For each selected folder, **every** image file is included exactly once, in order.
Encoding uses ffmpeg's concat demuxer (not `%03d` patterns), so gaps in numbering
or mixed padding cannot drop frames.

Usage:
  python anomalous_train_frames_to_videos.py [--dry-run] [--fps 25] \\
      [--frames-dir PATH] [--out-dir PATH]

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Folder name: two digits, underscore, then exactly four digits (anomalous train clip).
ANOMALOUS_CLIP_RE = re.compile(r"^\d{2}_\d{4}$")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_anomalous_clip_dir(name: str) -> bool:
    return bool(ANOMALOUS_CLIP_RE.match(name))


def _frame_sort_key(p: Path) -> tuple:
    stem = p.stem
    if stem.isdigit():
        return (0, int(stem), p.suffix.lower(), p.name)
    return (1, p.name.lower())


def list_frames_sorted(clip_dir: Path) -> list[Path]:
    frames = [p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(frames, key=_frame_sort_key)


def _ffconcat_quote_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def encode_clip(clip_dir: Path, out_path: Path, fps: float, dry_run: bool) -> bool:
    frames = list_frames_sorted(clip_dir)
    if not frames:
        print(f"  skip (no images): {clip_dir.name}", file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / fps

    if dry_run:
        print(
            f"# {clip_dir.name}: {len(frames)} frames -> concat demuxer (each file = 1 output frame)"
        )
        print(
            "ffmpeg ... -f concat -safe 0 -i <temp.ffconcat> -c:v libx264 -pix_fmt yuv420p",
            str(out_path),
        )
        return True

    # One file + duration per image (no trailing duplicate file line — that adds an extra frame).
    concat_lines = ["ffconcat version 1.0"]
    for p in frames:
        q = _ffconcat_quote_path(p)
        concat_lines.append(f"file '{q}'")
        concat_lines.append(f"duration {duration}")

    fd, concat_path = tempfile.mkstemp(suffix=".ffconcat", prefix="frames2vid_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(concat_lines) + "\n")

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ffmpeg not found; install ffmpeg and ensure it is on PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg failed for {clip_dir.name}: {e}", file=sys.stderr)
        return False
    finally:
        try:
            os.unlink(concat_path)
        except OSError:
            pass

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1] / "data" / "SHANGHAI" / "SHANGHAI_TRAIN" / "frames"
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=default_root,
        help=f"Root folder of per-clip frame directories (default: {default_root})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for MP4s (default: <frames-dir>/../videos_anomalous)",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Output frame rate (default: 25)")
    parser.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands only")
    args = parser.parse_args()

    frames_dir: Path = args.frames_dir
    if not frames_dir.is_dir():
        print(f"frames dir not found: {frames_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = frames_dir.parent / "videos_anomalous"

    clips = sorted(d for d in frames_dir.iterdir() if d.is_dir() and is_anomalous_clip_dir(d.name))
    if not clips:
        print(
            f"No folders matching NN_MMMM (e.g. 01_0014) under {frames_dir}.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(clips)} folders (4-digit suffix rule) under {frames_dir}")
    print(f"Writing videos to {out_dir} @ {args.fps} fps")

    ok = 0
    for clip in clips:
        out_mp4 = out_dir / f"{clip.name}.mp4"
        if encode_clip(clip, out_mp4, args.fps, args.dry_run):
            ok += 1
            if not args.dry_run:
                print(f"  {clip.name} -> {out_mp4.name}")

    print(f"Done: {ok}/{len(clips)} clips")
    if args.dry_run:
        print("(dry-run: no files written)")


if __name__ == "__main__":
    main()
