#!/usr/bin/env python3
"""
v2 step 2 — produce manifest_v2.jsonl by swapping each row's assistant label
with the enriched 2-3 sentence label for that video. Everything else
(images, scores, system prompt, user message) stays identical to the
original manifest, so the only changed variable is the training label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2   = Path(__file__).resolve().parent
DEFAULT_MANIFEST_IN  = REPO / "qwen_finetune_colab" / "out" / "manifest.jsonl"
DEFAULT_ENRICHED     = V2 / "data" / "enriched_train_annotations.json"
DEFAULT_OUT_DIR      = V2 / "out_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-in", type=Path, default=DEFAULT_MANIFEST_IN)
    ap.add_argument("--enriched",    type=Path, default=DEFAULT_ENRICHED)
    ap.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    enriched = {r["video_id"]: r["explanation"] for r in json.loads(args.enriched.read_text())}
    print(f"Loaded {len(enriched)} enriched labels")

    rows_in, rows_out = 0, 0
    no_label = []
    out_path = args.out_dir / "manifest.jsonl"
    with open(args.manifest_in) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rows_in += 1
            r = json.loads(line)
            vid = r["video_id"]
            new = enriched.get(vid)
            if new is None:
                no_label.append(vid)
                continue
            for turn in r["conversations"]:
                if turn["from"] == "assistant":
                    turn["value"] = json.dumps({"explanation": new}, ensure_ascii=False)
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows_out += 1

    print(f"Wrote {rows_out}/{rows_in} rows → {out_path}")
    if no_label:
        print(f"  WARNING: {len(set(no_label))} videos missing enriched label: {sorted(set(no_label))[:5]}")

    # Word-count check
    wcs = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            a = next(c["value"] for c in r["conversations"] if c["from"] == "assistant")
            try:
                e = json.loads(a)["explanation"]
            except Exception:
                e = a
            wcs.append(len(e.split()))
    print(f"v2 labels: mean={sum(wcs)/len(wcs):.1f}w  min={min(wcs)}w  max={max(wcs)}w")


if __name__ == "__main__":
    main()
