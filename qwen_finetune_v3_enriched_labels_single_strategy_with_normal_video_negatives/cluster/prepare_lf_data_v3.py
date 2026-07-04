#!/usr/bin/env python3
"""
Cluster prep for v3: convert manifest.jsonl (anom + normal) into LLaMA-Factory
sharegpt-format train/eval JSON files. Splits by video_id so eval rows
contain unseen videos.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

DATA_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives/data/out_v3")
WORK      = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives/lf_data")
NAME      = "rtfm_qwen_sft_v3"
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

# Image presence check
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


# Split by video_id, stratified per video_type
anom_vids   = sorted({r["video_id"] for r in rows if r["video_type"] == "anomalous"})
normal_vids = sorted({r["video_id"] for r in rows if r["video_type"] == "normal"})
rnd = random.Random(SEED)
rnd.shuffle(anom_vids)
rnd.shuffle(normal_vids)
n_val_anom   = max(1, int(round(len(anom_vids)   * VAL_FRAC)))
n_val_normal = max(1, int(round(len(normal_vids) * VAL_FRAC)))
val_vids = set(anom_vids[:n_val_anom]) | set(normal_vids[:n_val_normal])

tr_r, ev_r = [], []
for r in rows:
    (ev_r if r["video_id"] in val_vids else tr_r).append(r)

train_j = WORK / f"{NAME}_train.json"
eval_j  = WORK / f"{NAME}_eval.json"
train_j.write_text(json.dumps([to_lf(r) for r in tr_r], ensure_ascii=False))
eval_j.write_text( json.dumps([to_lf(r) for r in ev_r], ensure_ascii=False))

split = {
    "val_video_ids":  sorted(val_vids),
    "n_train_videos": len(set(r["video_id"] for r in tr_r)),
    "n_eval_videos":  len(set(r["video_id"] for r in ev_r)),
    "n_train_rows":   len(tr_r),
    "n_eval_rows":    len(ev_r),
    "n_train_anom":   sum(1 for r in tr_r if r["video_type"] == "anomalous"),
    "n_train_normal": sum(1 for r in tr_r if r["video_type"] == "normal"),
    "n_eval_anom":    sum(1 for r in ev_r if r["video_type"] == "anomalous"),
    "n_eval_normal":  sum(1 for r in ev_r if r["video_type"] == "normal"),
}
(WORK / "manifest_split.json").write_text(json.dumps(split, indent=2))
print(f"Wrote {len(tr_r)} train / {len(ev_r)} eval rows → {WORK}")
print(json.dumps(split, indent=2))
