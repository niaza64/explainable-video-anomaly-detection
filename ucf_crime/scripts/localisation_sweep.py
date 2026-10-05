#!/usr/bin/env python3
"""Can we localise UCF anomalies better by changing HOW we pick snippets?

Scores every anomalous test video once, then evaluates many selection
strategies against the ground-truth anomaly window. Metric = fraction of the
8 selected frames that land inside the GT window (chance ~= window/video).
"""
import json, os, sys, argparse
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--test-list", required=True)
ap.add_argument("--code-dir", required=True); ap.add_argument("--annotations", required=True)
ap.add_argument("--gated-dir", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
from model import Model

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Model(2048, 32); net.load_state_dict(torch.load(a.ckpt, map_location=dev))
net = net.to(dev).eval()

ann = {r["video_id"]: r for r in json.load(open(a.annotations))}
BUDGET = 8

def smooth(x, k):
    if k <= 1: return x
    ker = np.ones(k)/k
    return np.convolve(x, ker, mode="same")

def topk_idx(sig, k):
    return sorted(np.argsort(-sig)[:k].tolist())

def contiguous(sig, thr, k):
    segs, ins, st = [], False, 0
    for i, s in enumerate(sig):
        if not ins and s > thr: ins, st = True, i
        elif ins and s <= thr: segs.append((st, i-1)); ins = False
    if ins: segs.append((st, len(sig)-1))
    if not segs: return topk_idx(sig, k)
    tot = sum(e-s+1 for s, e in segs); out = []
    for s, e in segs:
        b = max(2, round(k*(e-s+1)/tot)); sel = {s, e}
        cand = sorted(range(s+1, e), key=lambda i: -sig[i])
        for i in cand:
            if len(sel) >= b: break
            if all(abs(i-j) >= 2 for j in sel): sel.add(i)
        out += sorted(sel)
    return sorted(set(out))[:k] or topk_idx(sig, k)

rows = []
paths = [l.strip() for l in open(a.test_list) if l.strip()]
for p in paths:
    vid = os.path.basename(p).replace(".npy", "").replace("_i3d", "")
    r = ann.get(vid)
    if not r or r["video_type"] != "anomalous": continue
    md_p = os.path.join(a.gated_dir, vid, "metadata.json")
    if not os.path.exists(md_p): continue
    md = json.load(open(md_p)); tf = md.get("total_video_frames") or 0
    wins = r.get("anomaly_windows_frames") or []
    if not tf or not wins: continue

    f = np.load(p, allow_pickle=True).astype(np.float32)
    inp = torch.from_numpy(f).unsqueeze(0).to(dev).permute(0, 2, 1, 3)
    with torch.no_grad():
        out = net(inputs=inp)
    logits = torch.mean(torch.squeeze(out[6], 1), 0).squeeze().cpu().numpy().reshape(-1)
    fmag = out[9].squeeze().detach().cpu().numpy()
    if fmag.ndim > 1: fmag = fmag.mean(axis=0)
    fmag = fmag.reshape(-1)
    T = len(logits)
    if len(fmag) != T: fmag = np.resize(fmag, T)
    fps_s = tf / T
    gt = np.array([any(x <= (i+0.5)*fps_s <= y for x, y in wins) for i in range(T)])
    if gt.sum() == 0: continue

    sigs = {"logits": logits, "featmag": fmag,
            "logits_z": (logits-logits.mean())/(logits.std()+1e-9),
            "featmag_z": (fmag-fmag.mean())/(fmag.std()+1e-9),
            "logits_sm5": smooth(logits, 5), "featmag_sm5": smooth(fmag, 5),
            "logits_x_featmag": (logits-logits.min())*(fmag-fmag.min())}
    rec = {"video_id": vid, "T": T, "chance": float(gt.mean())}
    for nm, s in sigs.items():
        for k in (1, 8):
            idx = topk_idx(s, k)
            rec[f"top{k}_{nm}"] = float(gt[idx].mean())
    for thr in (0.3, 0.5, 0.7, 0.9):
        rec[f"seg{thr}_logits"] = float(gt[contiguous(logits, thr, BUDGET)].mean())

    # peak-centred blocks: top-1 is the most reliable signal, so keep the
    # budget close to it instead of spreading across the video
    for nm, s_ in (("logits", logits), ("featmag_sm5", smooth(fmag, 5))):
        pk = int(np.argmax(s_))
        for half in (4, 8, 16):
            lo = max(0, pk-half); hi = min(T, pk+half+1)
            band = list(range(lo, hi))
            sel = sorted(band, key=lambda i: -s_[i])[:BUDGET]
            rec[f"peak{half}_{nm}"] = float(gt[sorted(sel)].mean())
        # contiguous block of BUDGET snippets centred on the peak
        lo = max(0, min(pk-BUDGET//2, T-BUDGET))
        rec[f"block_{nm}"] = float(gt[list(range(lo, min(lo+BUDGET, T)))].mean())
        # top-k spaced, but only within the top-scoring quartile
        thr_q = np.quantile(s_, 0.90)
        cand = [i for i in range(T) if s_[i] >= thr_q]
        rec[f"q90_{nm}"] = float(gt[sorted(sorted(cand, key=lambda i: -s_[i])[:BUDGET])].mean()) if cand else 0.0
    rows.append(rec)

json.dump(rows, open(a.out, "w"), indent=1)
keys = [k for k in rows[0] if k not in ("video_id", "T")]
print(f"anomalous videos: {len(rows)}\n")
print(f"{'strategy':<26} {'frames-in-GT-window':>20}")
print("-"*48)
for k in sorted(keys, key=lambda k: -np.mean([r[k] for r in rows])):
    v = np.mean([r[k] for r in rows])
    mark = "  <-- chance" if k == "chance" else ""
    print(f"{k:<26} {v:>18.3f}{mark}")
