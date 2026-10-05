#!/usr/bin/env python3
"""Improved gold-reference extraction from UCA, v2.

v1 ranked candidate sentences by ABSOLUTE overlap seconds with the anomaly
window. Diagnosed failure: flagged videos have much longer windows (median 16.0s
vs 6.3s; one Assault window is 230s), so a sweeping ambient sentence ("The police
came to the scene...") accumulates more overlap than the short precise one that
actually names the assault. The ranking preferred the wrong sentence exactly when
the window was long.

v2 ranks by, in order:
  1. whether the sentence contains a cue word for the labelled anomaly class,
  2. CONCENTRATION - how much of the SENTENCE lies inside the window,
  3. absolute overlap.
Plus a marked fallback: if no overlapping sentence mentions the class at all,
search the whole video (measured: 8 of 19 such cases do describe the anomaly
elsewhere) and record source="uca_cue_outside_window" so it stays auditable.

Still no LLM anywhere in this file - references remain independent of the judge.
"""
import json, os, argparse, statistics
from collections import Counter, defaultdict

NORMAL_REFERENCE = ("Nothing anomalous occurs in this video; it shows ordinary, "
                    "everyday activity with no unusual or criminal event.")

CLASS_CUES = {
    "Abuse":         ["beat","hit","kick","punch","abus","attack","strike","slap","push","scold","throw"],
    "Arrest":        ["police","arrest","handcuff","detain","officer","cop","pin","subdue","drag","restrain"],
    "Arson":         ["fire","burn","flame","ignit","torch","lit","blaze","smoke","gasoline","petrol"],
    "Assault":       ["assault","attack","beat","hit","punch","kick","fight","stab","strike","knock"],
    "Burglary":      ["break","kick","pry","enter","burglar","steal","window","climb","rob","force","smash"],
    "Explosion":     ["explo","blast","fire","smoke","burn","flame","blew","boom","erupt"],
    "Fighting":      ["fight","beat","punch","kick","brawl","hit","knock","attack","struggle","wrestl"],
    "RoadAccidents": ["hit","crash","collid","accident","knock","struck","overturn","ran over","run over","rear-end","flip"],
    "Robbery":       ["rob","gun","threat","snatch","grab","steal","take","kidnap","force","point","demand"],
    "Shooting":      ["shoot","shot","gun","fire","pistol","weapon","bullet","aim","shoot"],
    "Shoplifting":   ["steal","hid","hide","pocket","conceal","took","take","shoplift","stuff","slip","put"],
    "Stealing":      ["steal","stole","took","take","grab","snatch","carry","remove","pocket","hid","load"],
    "Vandalism":     ["smash","break","damag","destroy","kick","hit","spray","vandal","throw","tear","scratch"],
}


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def has_cue(text, cls):
    return any(c in text.lower() for c in CLASS_CUES.get(cls, []))


def pick(u, wins, cls, max_keep=3, min_conc=0.25):
    win_len = sum(b - a for a, b in wins) or 1e-9
    cands = []
    for (t0, t1), s in zip(u["timestamps"], u["sentences"]):
        ov = sum(overlap(t0, t1, a, b) for a, b in wins)
        if ov <= 0:
            continue
        dur = max(t1 - t0, 1e-9)
        cands.append({"start": round(t0, 2), "end": round(t1, 2),
                      "overlap_s": round(ov, 2),
                      "frac_of_window": round(ov / win_len, 3),
                      "concentration": round(min(ov / dur, 1.0), 3),
                      "cue": has_cue(s, cls),
                      "text": " ".join(s.split())})
    if not cands:
        return [], "no_overlap"
    # cue first, then concentration, then absolute overlap
    cands.sort(key=lambda r: (not r["cue"], -r["concentration"], -r["overlap_s"]))
    src = "uca_window_cue_ranked"
    if not any(c["cue"] for c in cands):
        # nothing in the window names the anomaly - look across the whole video
        out = []
        for (t0, t1), s in zip(u["timestamps"], u["sentences"]):
            if has_cue(s, cls):
                out.append({"start": round(t0, 2), "end": round(t1, 2), "overlap_s": 0.0,
                            "frac_of_window": 0.0, "concentration": 0.0, "cue": True,
                            "outside_window": True, "text": " ".join(s.split())})
        if out:
            mid = sum((a + b) / 2 for a, b in wins) / len(wins)
            out.sort(key=lambda r: abs((r["start"] + r["end"]) / 2 - mid))
            keep = out[:1]
            keep.sort(key=lambda r: r["start"])
            return keep, "uca_cue_outside_window"
    keep = cands[:1] + [c for c in cands[1:] if c["concentration"] >= min_conc]
    keep = keep[:max_keep]
    keep.sort(key=lambda r: r["start"])
    return keep, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uca-raw", required=True); ap.add_argument("--temporal", required=True)
    ap.add_argument("--gated-dir", required=True); ap.add_argument("--test-list", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    uca = {}
    for sp in ("Train", "Val", "Test"):
        for k, v in json.load(open(os.path.join(a.uca_raw, f"UCA_{sp}.json"))).items():
            uca[k] = v
    win = {}
    for line in open(a.temporal):
        p = line.split()
        if len(p) < 6: continue
        ws = [(int(p[2]), int(p[3])), (int(p[4]), int(p[5]))]
        win[p[0].replace(".mp4", "")] = (p[1], [w for w in ws if w[0] >= 0 and w[1] >= 0])

    os.makedirs(a.out_dir, exist_ok=True)
    tests = [l.strip().replace("\\", "/").split("/")[-1].replace("_i3d.npy", "")
             for l in open(a.test_list) if l.strip()]
    recs, flags = [], Counter()
    for vid in tests:
        is_norm = vid.startswith("Normal_Videos")
        mp = os.path.join(a.gated_dir, vid, "metadata.json")
        md = json.load(open(mp)) if os.path.exists(mp) else {}
        cls = win.get(vid, ("Normal", []))[0]
        rec = {"video_id": vid, "video_type": "normal" if is_norm else "anomalous",
               "class": cls, "gate_score": md.get("gate_score"),
               "total_frames": md.get("total_video_frames")}
        if is_norm:
            rec.update({"explanation": NORMAL_REFERENCE, "source": "fixed_normal_reference",
                        "anomaly_windows_frames": [], "uca_sentences_used": []})
            recs.append(rec); continue
        u = uca.get(vid); _, wins = win.get(vid, (cls, []))
        rec["anomaly_windows_frames"] = wins
        if u is None or not wins:
            rec.update({"explanation": None, "source": "MISSING"}); recs.append(rec)
            flags["missing"] += 1; continue
        dur = u.get("duration") or 0; tf = rec["total_frames"] or 0
        fps = (tf / dur) if (tf and dur) else 30.0
        sec = [(x / fps, y / fps) for x, y in wins]
        keep, src = pick(u, sec, cls)
        rec["anomaly_windows_seconds"] = [[round(x, 2), round(y, 2)] for x, y in sec]
        rec["explanation"] = " ".join(k["text"].rstrip(".") + "." for k in keep).strip() if keep else None
        rec["source"] = src; rec["uca_sentences_used"] = keep
        rec["class_cue_matched"] = bool(keep) and any(k.get("cue") for k in keep)
        flags[src] += 1
        if not rec["class_cue_matched"]: flags["still_no_cue"] += 1
        recs.append(rec)

    json.dump(recs, open(os.path.join(a.out_dir, "annotations_v2.json"), "w"), indent=1)
    an = [r for r in recs if r["video_type"] == "anomalous"]
    cue = sum(1 for r in an if r.get("class_cue_matched"))
    print(f"anomalous: {len(an)}   with class cue: {cue} ({100*cue/len(an):.0f}%)")
    print("sources:", dict(flags))
    lens = [len(r["explanation"] or "") for r in an]
    print("explanation chars: mean %.0f median %.0f" % (statistics.mean(lens), statistics.median(lens)))


if __name__ == "__main__":
    main()
