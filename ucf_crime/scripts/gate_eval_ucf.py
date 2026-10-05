"""Video-level gate evaluation on the UCF-Crime test set (290 videos).

Mirrors pipeline/run_rtfm_pipeline.py: gate_score = RTFM's score_abnormal
(top-k mean over snippet scores, k selected by feature magnitude).
"""
import os, sys, json, argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--test-list', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--code-dir', required=True)
a = ap.parse_args()
sys.path.insert(0, a.code_dir)
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Model(2048, 32)
net.load_state_dict(torch.load(a.ckpt, map_location=device))
net = net.to(device).eval()
print("loaded %s  k_abn=%s  device=%s" % (os.path.basename(a.ckpt), net.k_abn, device), flush=True)

paths = [l.strip() for l in open(a.test_list) if l.strip()]
rows = []
for i, p in enumerate(paths, 1):
    name = os.path.basename(p).replace('.npy', '').replace('_i3d', '')
    f = np.load(p, allow_pickle=True).astype(np.float32)      # (T,10,2048)
    inp = torch.from_numpy(f).unsqueeze(0).to(device).permute(0, 2, 1, 3)
    with torch.no_grad():
        score_abnormal, _, _, _, _, _, logits, _, _, _ = net(inputs=inp)
    seg = torch.mean(torch.squeeze(logits, 1), 0).squeeze().cpu().numpy().reshape(-1)
    rows.append({"video": name,
                 "gate": float(score_abnormal.squeeze().cpu()),
                 "T": int(f.shape[0]),
                 "seg_max": float(seg.max()),
                 "seg_mean": float(seg.mean()),
                 "is_anomalous": 0 if name.startswith("Normal_Videos") else 1})
    if i % 50 == 0: print("  %d/%d" % (i, len(paths)), flush=True)

json.dump(rows, open(a.out, "w"), indent=1)
y = np.array([r["is_anomalous"] for r in rows])
print("\nvideos: %d   anomalous=%d  normal=%d" % (len(y), y.sum(), (1-y).sum()))

def report(score, label):
    print("\n================ gate = %s ================" % label)
    best = None
    print(" thr    TP  FN  FP  TN   prec   rec    F1    acc")
    for thr in [0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7]:
        pred = (score > thr).astype(int)
        tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
        fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        acc  = (tp+tn)/len(y)
        print(" %.2f  %4d %3d %3d %3d  %.3f  %.3f  %.3f  %.3f" % (thr,tp,fn,fp,tn,prec,rec,f1,acc))
        if best is None or f1 > best[1]: best = (thr,f1,tp,fn,fp,tn,prec,rec,acc)
    thr,f1,tp,fn,fp,tn,prec,rec,acc = best
    print(" BEST-F1 thr=%.2f -> %d/%d anomalous caught (recall %.1f%%), %d false alarms of %d normal, precision %.1f%%, accuracy %.1f%%"
          % (thr, tp, tp+fn, 100*rec, fp, fp+tn, 100*prec, 100*acc))
    # video-level AUC
    o = np.argsort(-score); ys = y[o]
    tps = np.cumsum(ys); fps = np.cumsum(1-ys)
    tpr = np.r_[0, tps/max(tps[-1],1)]; fpr = np.r_[0, fps/max(fps[-1],1)]
    vauc = float(np.trapezoid(tpr, fpr)) if hasattr(np,'trapezoid') else float(np.trapz(tpr,fpr))
    print(" video-level AUC = %.4f" % vauc)
    return best

g = np.array([r["gate"] for r in rows]); sm = np.array([r["seg_max"] for r in rows])
b1 = report(g, "score_abnormal (as in ShanghaiTech pipeline)")
b2 = report(sm, "max segment score")

# per-class recall at best-F1 gate threshold
thr = b1[0]
print("\nper-class recall at gate thr=%.2f:" % thr)
import re, collections
cls = collections.defaultdict(lambda: [0,0])
for r in rows:
    if not r["is_anomalous"]: continue
    c = re.match(r"([A-Za-z]+?)\d+_x264", r["video"])
    c = c.group(1) if c else r["video"]
    cls[c][1] += 1
    if r["gate"] > thr: cls[c][0] += 1
for c in sorted(cls): 
    hit,tot = cls[c]; print("   %-16s %2d/%2d  (%.0f%%)" % (c,hit,tot,100*hit/tot))
