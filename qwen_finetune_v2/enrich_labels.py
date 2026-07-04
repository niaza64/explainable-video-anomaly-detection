#!/usr/bin/env python3
"""
v2 step 1 — enrich the short 1-sentence training labels into 2–3 sentence
descriptions that match the test-annotation style, using GPT-4o on a small
set of frames from the human-annotated anomaly window of each training video.

Input:
  - data/short_train_annotations.json   (the original ~20-word labels)
  - frames already extracted under qwen_finetune_colab/out/images/<vid>/human_span_smart/*.jpg

Output:
  - data/enriched_train_annotations.json
      [{video_id, original_explanation, enriched_explanation,
        anomaly_start_frame, anomaly_end_frame, frames_used: [paths]}, ...]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
V2   = Path(__file__).resolve().parent
IMG_ROOT = REPO / "qwen_finetune_colab" / "out" / "images"
DEFAULT_IN  = V2 / "data" / "short_train_annotations.json"
DEFAULT_OUT = V2 / "data" / "enriched_train_annotations.json"

SYSTEM = (
    "You are an expert annotator for a video anomaly explanation dataset. "
    "You will be shown a SHORT one-sentence anomaly description written by a "
    "human, plus several frames sampled from the anomalous portion of that "
    "surveillance video.\n\n"
    "Your job: rewrite the description as 2 to 3 sentences (about 35 to 60 "
    "words total) that match this reference style:\n"
    "  'A person in a red shirt is skateboarding in a pedestrian-only area "
    "  where skateboarding is not allowed, followed by another person in a "
    "  white shirt who is also skateboarding.'\n\n"
    "Rules:\n"
    "- DO NOT change the type of anomaly named in the short label. If it "
    "  says 'cycling', the anomaly is cycling. Trust the short label.\n"
    "- DO add concrete visual detail you can see in the frames: clothing "
    "  colour, accessories carried, direction of travel, number of people "
    "  involved, or another pedestrian visible.\n"
    "- DO say briefly WHY it is anomalous (e.g. cycling/skateboarding is not "
    "  allowed in a pedestrian-only area).\n"
    "- DO NOT mention frame indices, snippet numbers, anomaly scores, or "
    "  surveillance system reasoning.\n"
    "- DO NOT speculate about intent (no 'suspicious', 'possibly stealing').\n"
    "- Respond with ONLY a JSON object: {\"explanation\": \"...\"}"
)


def pick_frames(video_id: str, k: int = 5) -> list[Path]:
    """Pick up to k frames from the human_span_smart strategy (in anomaly window)."""
    base = IMG_ROOT / video_id
    candidates = []
    pref = base / "human_span_smart"
    if pref.is_dir():
        candidates = sorted(pref.glob("*.jpg"))
    if not candidates and base.is_dir():
        for strat_dir in sorted(base.iterdir()):
            if strat_dir.is_dir():
                candidates = sorted(strat_dir.glob("*.jpg"))
                if candidates:
                    break
    if not candidates:
        return []
    n = len(candidates)
    if n <= k:
        return candidates
    idx = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
    return [candidates[i] for i in idx]


def call_gpt4o(client, short_label: str, frames: list[Path]) -> str:
    content = [
        {"type": "text",
         "text": f"Short label (the anomaly type you must preserve):\n\"{short_label}\"\n\n"
                 f"Frames from the anomalous portion of the video:"},
    ]
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode("utf-8")
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": content},
        ],
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content)
    return parsed["explanation"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", dest="outp", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--frames-per-video", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap N videos (debug)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    rows = json.loads(args.inp.read_text())
    if args.limit:
        rows = rows[: args.limit]

    # Resume support
    done = {}
    if args.outp.exists():
        done = {r["video_id"]: r for r in json.loads(args.outp.read_text())}
        print(f"Resuming: {len(done)} already enriched")

    enriched = list(done.values())
    for i, r in enumerate(rows, 1):
        vid = r["video_id"]
        if vid in done:
            continue
        short = r["explanation"].strip()
        frames = pick_frames(vid, args.frames_per_video)
        if not frames:
            print(f"  [{i}/{len(rows)}] {vid}: NO FRAMES — keeping short label")
            new = short
        else:
            try:
                new = call_gpt4o(client, short, frames)
                print(f"  [{i}/{len(rows)}] {vid}: {len(new.split())}w  {new[:100]}...")
            except Exception as e:
                print(f"  [{i}/{len(rows)}] {vid}: ERROR {e} — keeping short")
                new = short
            time.sleep(0.3)

        enriched.append({
            "video_id":              vid,
            "original_explanation":  short,
            "explanation":           new,            # used by build_sft_data_v2
            "anomaly_start_frame":   r.get("anomaly_start_frame"),
            "anomaly_end_frame":     r.get("anomaly_end_frame"),
            "frames_used":           [str(f.relative_to(REPO)) for f in frames],
        })
        # Incremental save (cheap insurance)
        args.outp.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))

    # Word-count diagnostics
    src_wc = [len(r["original_explanation"].split()) for r in enriched]
    new_wc = [len(r["explanation"].split())          for r in enriched]
    print()
    print(f"Enriched {len(enriched)} videos → {args.outp}")
    print(f"  orig labels   mean={sum(src_wc)/len(src_wc):.1f}w  max={max(src_wc)}w")
    print(f"  new  labels   mean={sum(new_wc)/len(new_wc):.1f}w  max={max(new_wc)}w")


if __name__ == "__main__":
    main()
