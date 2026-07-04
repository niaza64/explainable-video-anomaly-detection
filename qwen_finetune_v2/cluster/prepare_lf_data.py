#!/usr/bin/env python3
"""
Cluster prep: take the v2 manifest.jsonl produced locally and convert it to
LLaMA-Factory's sharegpt-style train/eval JSON files. Mirrors the logic in
the original Colab training notebook (cell 3) so the only difference vs.
the v1 run is the label content.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/svc_td_ppml/qrx527/niaz_research_v2/data/out_v2")
WORK      = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/scratch/svc_td_ppml/qrx527/niaz_research_v2/lf_data")
NAME      = "rtfm_qwen_sft_v2"
VAL_FRAC, SEED = 0.15, 42
MAX_IMAGES = 16

random.seed(SEED)
WORK.mkdir(parents=True, exist_ok=True)
MANIFEST = DATA_ROOT / "manifest.jsonl"
assert MANIFEST.is_file(), MANIFEST

rows = []
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

# Verify every image exists
miss = [rel for r in rows for rel in r["images"] if not (DATA_ROOT / rel).is_file()]
assert not miss, f"missing {len(miss)} images, first: {miss[:5]}"


def user_prompt_from_scores(scores):
    s = ", ".join(f"{x:.4f}" for x in scores)
    return (f"Anomaly scores for each shown frame (temporal order, RTFM logits): [{s}]\n"
            "Describe the anomalous activity.")


def subsample(rels, scores, k):
    n = len(rels)
    if n <= k:
        return rels, scores
    if k == 1:
        idx = [0]
    else:
        idx = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
    return [rels[i] for i in idx], [scores[i] for i in idx]


def to_lf(r):
    rels = list(r["images"])
    sc   = list(r.get("scores", []))
    assert len(sc) == len(rels), (len(sc), len(rels), r.get("id"))
    rels, sc = subsample(rels, sc, MAX_IMAGES)
    imgs = [str((DATA_ROOT / rel).resolve()) for rel in rels]
    ph   = "\n".join("<image>" for _ in imgs)
    u    = f"{ph}\n{user_prompt_from_scores(sc)}"
    a    = next(t["value"] for t in r["conversations"] if t["from"] == "assistant")
    return {
        "messages": [
            {"role": "system",    "content": r["system"]},
            {"role": "user",      "content": u},
            {"role": "assistant", "content": a},
        ],
        "images": imgs,
    }


by = {}
for r in rows:
    by.setdefault(r["video_id"], []).append(r)
vids = sorted(by)
random.shuffle(vids)
n_val = max(1, int(round(len(vids) * VAL_FRAC)))
val = set(vids[:n_val])

tr_r, ev_r = [], []
for vid, rs in by.items():
    (ev_r if vid in val else tr_r).extend(rs)

train_j = WORK / f"{NAME}_train.json"
eval_j  = WORK / f"{NAME}_eval.json"
train_j.write_text(json.dumps([to_lf(r) for r in tr_r], ensure_ascii=False))
eval_j.write_text(json.dumps([to_lf(r) for r in ev_r], ensure_ascii=False))

split = {
    "val_video_ids": sorted(val),
    "n_train_videos": len(vids) - n_val,
    "n_eval_videos": n_val,
    "n_train_rows": len(tr_r),
    "n_eval_rows": len(ev_r),
}
(WORK / "manifest_split.json").write_text(json.dumps(split, indent=2))
print(f"Wrote {len(tr_r)} train / {len(ev_r)} eval rows → {WORK}")
print(json.dumps(split, indent=2))
