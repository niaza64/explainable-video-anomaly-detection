#!/usr/bin/env python3
"""Build the LoRA / RAG training manifest for UCF-Crime.

Same schema as the ShanghaiTech v3 manifest.jsonl. Targets are UCA (CVPR 2024)
human sentences, selected against **RTFM's own predicted anomalous window** -
not the ground-truth window - so the target describes exactly the interval whose
frames the model is shown.

No LLM writes any target here.
"""
import json, os, argparse, random
from collections import Counter

SYSTEM = (
 "You are a surveillance video anomaly analyst. You will be shown a set of frames sampled "
 "from a surveillance video that has been flagged as anomalous by a weakly-supervised anomaly "
 "detection model (RTFM).\n\n"
 "The frames are ordered temporally. Each frame comes from a specific temporal snippet of the "
 "video, and you are given the anomaly score for that snippet (0 = normal, 1 = highly "
 "anomalous).\n\n"
 "The frames were specifically selected from the anomalous portions of the video — they "
 "represent the onset, peak, and resolution of the detected anomaly.\n\n"
 "Your task: Based on ALL the frames and their anomaly scores together, provide a single "
 "concise explanation (2-3 sentences) of what anomalous activity is happening. Focus on:\n"
 "- WHAT is happening (the specific anomalous activity)\n"
 "- WHO/WHAT is involved (people, vehicles, objects — describe appearance)\n"
 "- WHEN in the sequence it starts and ends\n"
 "- WHY it is anomalous (how it deviates from ordinary, lawful activity in this setting)\n\n"
 'Respond with ONLY a JSON object in this exact format:\n{"explanation": "..."}'
)
USER_TAIL = "Describe the anomalous activity."


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def pick(u, wins, min_frac=0.30, max_keep=3):
    win_len = sum(b - a for a, b in wins) or 1e-9
    sc = []
    for (t0, t1), s in zip(u["timestamps"], u["sentences"]):
        ov = sum(overlap(t0, t1, a, b) for a, b in wins)
        if ov > 0:
            sc.append({"start": t0, "end": t1, "ov": ov,
                       "frac": ov / win_len, "text": " ".join(s.split())})
    sc.sort(key=lambda r: -r["ov"])
    keep = sc[:1] + [r for r in sc[1:] if r["frac"] >= min_frac]
    keep = keep[:max_keep]
    keep.sort(key=lambda r: r["start"])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uca-raw", required=True)
    ap.add_argument("--gated-train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsample", type=int, default=0,
                    help="also emit a size-matched pool of N videos (regime-matched ablation)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    uca = {}
    for sp in ("Train", "Val", "Test"):
        for k, v in json.load(open(os.path.join(a.uca_raw, f"UCA_{sp}.json"))).items():
            uca[k] = v

    recs, skipped = [], Counter()
    for vid in sorted(os.listdir(a.gated_train_dir)):
        mp = os.path.join(a.gated_train_dir, vid, "metadata.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        u = uca.get(vid)
        if u is None:
            skipped["no_uca"] += 1; continue
        frames = [e for e in m["extracted_frames"] if e.get("file")]
        if not frames:
            skipped["no_frames"] += 1; continue

        dur = u.get("duration") or 0
        tf = m.get("total_video_frames") or 0
        if not dur or not tf:
            skipped["no_duration"] += 1; continue
        fps = tf / dur
        T = m["n_segments"]
        fps_snip = tf / T                      # frames per snippet

        # RTFM's own predicted anomalous window(s), in seconds
        wins = [((s["start_snippet"] * fps_snip) / fps,
                 ((s["end_snippet"] + 1) * fps_snip) / fps)
                for s in m["anomalous_segments"]]
        if not wins:
            skipped["no_segments"] += 1; continue

        keep = pick(u, wins)
        if not keep:
            skipped["no_overlap"] += 1; continue
        target = " ".join(k["text"].rstrip(".") + "." for k in keep).strip()

        scores = [round(float(e["score"]), 4) for e in frames]
        imgs = [os.path.join(vid, e["file"]) for e in frames]
        user = ("<image>\n" * len(frames)
                + f"Anomaly scores for each shown frame (temporal order, RTFM logits): {scores}\n"
                + USER_TAIL)
        recs.append({
            "id": "anom_" + vid,
            "video_id": vid,
            "video_type": "anomalous",
            "strategy": "uca_rtfm_window",
            "system": SYSTEM,
            "conversations": [
                {"from": "human", "value": user},
                {"from": "assistant", "value": json.dumps({"explanation": target})},
            ],
            "images": imgs,
            "frame_indices": [e["frame_num"] for e in frames],
            "scores": [float(e["score"]) for e in frames],
            "uca_sentences_used": keep,
            "gate_score": m.get("gate_score"),
        })

    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("manifest records: %d  -> %s" % (len(recs), a.out))
    print("skipped:", dict(skipped))
    lens = [len(json.loads(r["conversations"][1]["value"])["explanation"]) for r in recs]
    if lens:
        import statistics
        print("target chars: mean %.0f median %.0f" % (statistics.mean(lens), statistics.median(lens)))

    if a.subsample and a.subsample < len(recs):
        random.Random(a.seed).shuffle(recs)
        sub = recs[:a.subsample]
        p2 = a.out.replace(".jsonl", f"_subsample{a.subsample}.jsonl")
        with open(p2, "w") as f:
            for r in sub:
                f.write(json.dumps(r) + "\n")
        print("regime-matched pool: %d -> %s" % (len(sub), p2))


if __name__ == "__main__":
    main()
