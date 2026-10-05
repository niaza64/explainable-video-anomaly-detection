#!/usr/bin/env python3
"""Does the improved UCF frame sampler regress ShanghaiTech?

The paper must use ONE method on both datasets. The v4/v5 samplers are not a
different detector - same RTFM model, same checkpoint, same forward pass, only
the rule for choosing which snippets to show, using RTFM's own feat_magnitudes.
This checks that swapping the rule does not hurt ShanghaiTech.

Metric is identical to the UCF analysis: fraction of selected frames landing
inside the ground-truth anomalous frames, plus coverage.
"""
import json, os, sys, glob, argparse
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--rtfm-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=8)
a = ap.parse_args()
sys.path.insert(0, a.rtfm_dir)
from model import Model

dev = torch.device("cpu")
net = Model(2048, 32)
net.load_state_dict(torch.load(os.path.join(a.rtfm_dir, "rtfm_checkpoints", "rtfm_best.pkl"),
                               map_location=dev))
net = net.to(dev).eval()

MASK = os.path.join(a.rtfm_dir, "list", "test_frame_mask")
FEAT = os.path.join(a.rtfm_dir, "data", "SH_Test_ten_crop_i3d")


def med_sm(x, k=5):
    if len(x) < k: return x
    p = k//2; xp = np.pad(x, p, mode="edge")
    return np.array([np.median(xp[i:i+k]) for i in range(len(x))])


def sel_gap(s, k, gap):
    out = []
    for i in np.argsort(-s):
        i = int(i)
        if all(abs(i-j) >= gap for j in out): out.append(i)
        if len(out) >= k: break
    if len(out) < k:
        out += [int(i) for i in np.argsort(-s) if int(i) not in out][:k-len(out)]
    return sorted(out)


def sel_segments(seg, k, thr=0.3, min_gap=2):
    """v1: contiguous runs above threshold, first/last always kept."""
    segs, ins, st = [], False, 0
    for i, v in enumerate(seg):
        if not ins and v > thr: ins, st = True, i
        elif ins and v <= thr: segs.append((st, i-1)); ins = False
    if ins: segs.append((st, len(seg)-1))
    if not segs:
        return sorted(np.argsort(-seg)[:k].tolist())
    tot = sum(e-s+1 for s, e in segs); out = []
    for s, e in segs:
        b = max(2, round(k*(e-s+1)/tot)); sl = {s, e}
        for i in sorted(range(s+1, e), key=lambda j: -seg[j]):
            if len(sl) >= b: break
            if all(abs(i-j) >= min_gap for j in sl): sl.add(i)
        out += sorted(sl)
    return sorted(set(out))[:k]


rows = []
for f in sorted(glob.glob(os.path.join(FEAT, "*.npy"))):
    vid = os.path.basename(f).replace("_i3d.npy", "")
    mp = os.path.join(MASK, vid + ".npy")
    if not os.path.exists(mp): continue
    gtf = np.load(mp)
    if gtf.sum() == 0: continue                      # normal video, no anomaly frames
    x = np.load(f, allow_pickle=True).astype(np.float32)
    inp = torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1, 3)
    with torch.no_grad(): o = net(inputs=inp)
    logits = torch.mean(torch.squeeze(o[6], 1), 0).squeeze().numpy().reshape(-1).astype(np.float64)
    fm = o[9].squeeze().numpy()
    if fm.ndim > 1: fm = fm.mean(axis=0)
    fm = fm.reshape(-1).astype(np.float64)
    T = len(logits)
    if len(fm) != T: fm = np.resize(fm, T)
    nF = len(gtf); fps_s = nF / T
    gt = np.array([gtf[min(int((i+0.5)*fps_s), nF-1)] == 1 for i in range(T)])
    if gt.sum() == 0: continue
    prod = (logits-logits.min())*(fm-fm.min())
    strat = {
        "v1_segments_logits_k8":  sel_segments(logits, 8),
        "v4_featmag_gap3_k8":     sel_gap(fm, 8, 3),
        "v5_prod_med5_gap8_k12":  sel_gap(med_sm(prod, 5), 12, 8),
    }
    rec = {"video_id": vid, "T": T, "chance": float(gt.mean())}
    for n, idx in strat.items():
        rec[n] = float(gt[idx].mean()) if idx else 0.0
    rows.append(rec)

json.dump(rows, open(a.out, "w"), indent=1)
import statistics as st
ch = st.mean([r["chance"] for r in rows])
print(f"ShanghaiTech anomalous test videos: {len(rows)}   chance = {ch:.3f}\n")
print(f"{'sampler':<26} {'mean':>7} {'>=1':>7} {'>=2':>7} {'lift':>7}")
print("-"*58)
for k in ("v1_segments_logits_k8", "v4_featmag_gap3_k8", "v5_prod_med5_gap8_k12"):
    v = np.array([r[k] for r in rows])
    print(f"{k:<26} {v.mean():>7.3f} {100*(v>0).mean():>6.0f}% {100*(v>=0.25).mean():>6.0f}% {v.mean()/ch:>6.2f}x")
