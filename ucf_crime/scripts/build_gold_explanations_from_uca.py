#!/usr/bin/env python3
"""Build gold reference explanations for the UCF-Crime test set.

Source of truth is HUMAN text only. For each anomalous test video we take the
UCA (CVPR 2024) human-written sentences whose timestamps overlap the official
UCF-Crime temporal anomaly window, and select them deterministically - no LLM
is involved anywhere in this file, so the references stay independent of the
GPT-4o judge.

Outputs (schema matches the ShanghaiTech annotations.json the pipeline reads):
    annotations.json          - the references, one record per video
    annotations_review.md     - human-readable sheet for spot-checking
    build_report.json         - QA counts + everything needing manual attention
"""
import json, os, argparse, statistics
from collections import Counter, defaultdict

CLASS_CUES = {
    "Abuse":         ["beat","hit","kick","punch","abus","attack","strike","slap","push","scold"],
    "Arrest":        ["police","arrest","handcuff","detain","officer","cop","pin","subdue","drag"],
    "Arson":         ["fire","burn","flame","ignit","torch","lit","blaze","smoke"],
    "Assault":       ["assault","attack","beat","hit","punch","kick","fight","stab","strike"],
    "Burglary":      ["break","kick","pry","enter","burglar","steal","door","window","climb","rob"],
    "Explosion":     ["explo","blast","fire","smoke","burn","flame","blew","boom"],
    "Fighting":      ["fight","beat","punch","kick","brawl","hit","knock","attack","struggle"],
    "RoadAccidents": ["hit","crash","collid","accident","knock","struck","overturn","ran over","run over"],
    "Robbery":       ["rob","gun","threat","snatch","grab","steal","take","kidnap","force","point"],
    "Shooting":      ["shoot","shot","gun","fire","pistol","weapon","bullet","aim"],
    "Shoplifting":   ["steal","hid","hide","pocket","conceal","took","take","shoplift","bag","stuff","put"],
    "Stealing":      ["steal","stole","took","take","grab","snatch","carry","remove","pocket","hid"],
    "Vandalism":     ["smash","break","damag","destroy","kick","hit","spray","vandal","throw","tear"],
}

NORMAL_REFERENCE = ("Nothing anomalous occurs in this video; it shows ordinary, "
                    "everyday activity with no unusual or criminal event.")


def load_uca(raw_dir):
    uca = {}
    for split in ("Train", "Val", "Test"):
        p = os.path.join(raw_dir, f"UCA_{split}.json")
        for k, v in json.load(open(p)).items():
            uca[k] = dict(v, _uca_split=split)
    return uca


def load_windows(path):
    """Temporal_Anomaly_Annotation_for_Testing_Videos.txt -> frames."""
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) < 6:
            continue
        wins = [(int(p[2]), int(p[3])), (int(p[4]), int(p[5]))]
        out[p[0].replace(".mp4", "")] = (p[1], [w for w in wins if w[0] >= 0 and w[1] >= 0])
    return out


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def pick_sentences(u, sec_wins, min_frac=0.30, max_keep=3):
    """Rank UCA sentences by how much of the anomaly window they cover.

    Keeps the best sentence always, then any other covering >= min_frac of the
    window, up to max_keep - this is what filters out the ambient-context
    sentences ("many vehicles driving on the road") that bare overlap pulls in.
    """
    win_len = sum(b - a for a, b in sec_wins) or 1e-9
    scored = []
    for (t0, t1), s in zip(u["timestamps"], u["sentences"]):
        ov = sum(overlap(t0, t1, a, b) for a, b in sec_wins)
        if ov > 0:
            scored.append({"start": round(t0, 2), "end": round(t1, 2),
                           "overlap_s": round(ov, 2),
                           "frac_of_window": round(ov / win_len, 3),
                           "text": " ".join(s.split())})
    scored.sort(key=lambda r: -r["overlap_s"])
    keep = scored[:1] + [r for r in scored[1:] if r["frac_of_window"] >= min_frac]
    keep = keep[:max_keep]
    # read in time order, not in overlap order - otherwise the resolution can be
    # narrated before the onset ("the dog ran off... the postman took out a stick")
    keep.sort(key=lambda r: r["start"])
    return scored, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uca-raw", required=True)
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--gated-dir", required=True)
    ap.add_argument("--test-list", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    uca = load_uca(a.uca_raw)
    win = load_windows(a.temporal)
    os.makedirs(a.out_dir, exist_ok=True)

    tests = [l.strip().replace("\\", "/").split("/")[-1].replace("_i3d.npy", "")
             for l in open(a.test_list) if l.strip()]

    records, report = [], {"needs_review": [], "no_uca": [], "no_window": [],
                           "no_overlap": [], "single_short_sentence": []}

    for vid in tests:
        is_norm = vid.startswith("Normal_Videos")
        meta_p = os.path.join(a.gated_dir, vid, "metadata.json")
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        rec = {"video_id": vid,
               "video_type": "normal" if is_norm else "anomalous",
               "class": win.get(vid, ("Normal", []))[0],
               "gate_score": meta.get("gate_score"),
               "total_frames": meta.get("total_video_frames"),
               "passes_gate": meta.get("passes_gate", {})}

        if is_norm:
            rec.update({"explanation": NORMAL_REFERENCE, "source": "fixed_normal_reference",
                        "anomaly_windows_frames": [], "uca_sentences_used": []})
            records.append(rec)
            continue

        u = uca.get(vid)
        cls, wins = win.get(vid, (rec["class"], []))
        rec["anomaly_windows_frames"] = wins
        if u is None:
            report["no_uca"].append(vid); report["needs_review"].append(vid)
            rec.update({"explanation": None, "source": "MISSING_UCA", "uca_sentences_used": []})
            records.append(rec); continue
        if not wins:
            report["no_window"].append(vid); report["needs_review"].append(vid)

        dur = u.get("duration") or 0
        tf = rec["total_frames"]
        fps = (tf / dur) if (tf and dur) else 30.0
        rec["fps_used"] = round(fps, 3)
        sec_wins = [(x / fps, y / fps) for x, y in wins] or [(0.0, dur)]
        rec["anomaly_windows_seconds"] = [[round(x, 2), round(y, 2)] for x, y in sec_wins]

        scored, keep = pick_sentences(u, sec_wins)
        if not scored:
            report["no_overlap"].append(vid); report["needs_review"].append(vid)
            # fall back to the whole-video sentences so a human has material to work from
            keep = [{"start": round(t0, 2), "end": round(t1, 2), "overlap_s": 0.0,
                     "frac_of_window": 0.0, "text": " ".join(s.split())}
                    for (t0, t1), s in list(zip(u["timestamps"], u["sentences"]))[:2]]
            rec["source"] = "uca_no_overlap_fallback_whole_video"
        else:
            rec["source"] = "uca_anomaly_window_overlap"

        text = " ".join(k["text"].rstrip(".") + "." for k in keep).strip()
        rec["explanation"] = text
        rec["uca_sentences_used"] = keep
        rec["uca_sentences_all_overlapping"] = scored
        rec["uca_split"] = u.get("_uca_split")
        rec["n_uca_sentences_total"] = len(u["sentences"])

        cues = CLASS_CUES.get(cls, [])
        rec["class_cue_matched"] = bool(cues) and any(c in text.lower() for c in cues)
        rec["window_coverage"] = round(sum(k["frac_of_window"] for k in keep), 3)

        reasons = []
        if len(text) < 60:
            reasons.append("very short")
            report["single_short_sentence"].append(vid)
        if cues and not rec["class_cue_matched"]:
            reasons.append("no %s cue word" % cls)
            report.setdefault("no_class_cue", []).append(vid)
        if rec["window_coverage"] < 0.25:
            reasons.append("low window coverage")
            report.setdefault("low_coverage", []).append(vid)
        rec["review_reasons"] = reasons
        if reasons:
            report["needs_review"].append(vid)
        records.append(rec)

    json.dump(records, open(os.path.join(a.out_dir, "annotations.json"), "w"), indent=1)

    anom = [r for r in records if r["video_type"] == "anomalous"]
    lens = [len(r["explanation"] or "") for r in anom]
    summary = {
        "total_test_videos": len(records),
        "anomalous": len(anom),
        "normal": len(records) - len(anom),
        "anomalous_with_explanation": sum(1 for r in anom if r["explanation"]),
        "explanation_chars_mean": round(statistics.mean(lens), 1) if lens else 0,
        "explanation_chars_median": statistics.median(lens) if lens else 0,
        "sentences_used_hist": dict(Counter(len(r["uca_sentences_used"]) for r in anom)),
        "by_class": dict(Counter(r["class"] for r in anom)),
        "needs_review_count": len(set(report["needs_review"])),
    }
    report["needs_review"] = sorted(set(report["needs_review"]))
    json.dump({"summary": summary, "flags": report},
              open(os.path.join(a.out_dir, "build_report.json"), "w"), indent=1)

    # review sheet
    with open(os.path.join(a.out_dir, "annotations_review.md"), "w") as f:
        f.write("# Gold explanations - review sheet\n\n")
        f.write("Human text from UCA (CVPR 2024). No LLM involved. Edit `annotations.json`\n")
        f.write("directly; this sheet is for reading.\n\n")
        f.write(f"- anomalous videos: **{summary['anomalous']}**\n")
        f.write(f"- with an explanation: **{summary['anomalous_with_explanation']}**\n")
        f.write(f"- flagged for review: **{summary['needs_review_count']}**\n\n")
        by = defaultdict(list)
        for r in anom:
            by[r["class"]].append(r)
        for cls in sorted(by):
            f.write(f"\n## {cls} ({len(by[cls])})\n\n")
            for r in sorted(by[cls], key=lambda x: x["video_id"]):
                rs = r.get("review_reasons") or []
                flag = (" ⚠️ " + "; ".join(rs)) if rs else ""
                w = r.get("anomaly_windows_seconds")
                f.write(f"**{r['video_id']}**{flag}  ")
                f.write(f"<sub>gate={r['gate_score']:.3f} · anomaly {w}s</sub>\n\n")
                f.write(f"> {r['explanation']}\n\n")
    print(json.dumps(summary, indent=1))
    print("flags:", {k: len(v) for k, v in report.items()})


if __name__ == "__main__":
    main()
