#!/usr/bin/env python3
"""
Once UCF I3D features are unpacked, inspect their layout so we can wire up
the rest of the pipeline (RTFM inference, frame extraction, etc.).

Run on cluster after download completes:
  python inspect_ucf_features.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/scratch/svc_td_ppml/qrx527/niaz_research_ucf_crime_separate_workspace/data")

def inspect_dir(d: Path, label: str):
    print(f"\n=== {label}: {d} ===")
    if not d.is_dir():
        print("  (does not exist)")
        return
    files = sorted(d.iterdir())
    print(f"  total entries: {len(files)}")
    # Show first 5 of each extension
    by_ext = {}
    for f in files:
        by_ext.setdefault(f.suffix.lower(), []).append(f)
    for ext, items in by_ext.items():
        print(f"  {ext or '(no ext)'}: {len(items)} files, first 3 = {[i.name for i in items[:3]]}")

    # If we have npy files, show shape of first one
    npys = by_ext.get(".npy", [])
    if npys:
        f = npys[0]
        try:
            x = np.load(str(f), allow_pickle=True)
            print(f"  Sample .npy: {f.name}  shape={x.shape}  dtype={x.dtype}")
        except Exception as e:
            print(f"  Could not load {f.name}: {e}")

inspect_dir(ROOT / "i3d_test",  "TEST FEATURES")
inspect_dir(ROOT / "i3d_train", "TRAIN FEATURES")
