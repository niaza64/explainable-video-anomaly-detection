"""Pre-pool train I3D features to RTFM's 32 segments.

dataset.py does, per training sample:
    f = np.load(...)            # (T,10,2048)
    f = f.transpose(1,0,2)      # (10,T,2048)
    [process_feat(c,32) for c in f]  -> (10,32,2048)
That is deterministic, so we do it once offline. 40 MB avg -> 2.6 MB fixed.
"""
import os, sys, argparse, numpy as np

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int32)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat

def pool(src):
    a = np.load(src, mmap_mode='r')          # (T,10,2048), never fully resident
    T = a.shape[0]
    out = np.zeros((10, 32, 2048), dtype=np.float32)
    r = np.linspace(0, T, 33, dtype=np.int32)
    for i in range(32):
        lo, hi = r[i], r[i + 1]
        chunk = np.asarray(a[lo:hi] if hi > lo else a[lo:lo + 1])   # (n,10,2048)
        out[:, i, :] = chunk.mean(axis=0) if hi > lo else chunk[0]
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--names', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--verify', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = [l.strip() for l in open(a.names) if l.strip()]
    names = names[a.shard::a.nshards]
    for i, n in enumerate(names, 1):
        src = os.path.join(a.in_dir, n + '_i3d.npy')
        dst = os.path.join(a.out_dir, n + '_i3d.npy')
        if os.path.exists(dst):
            continue
        o = pool(src)
        if a.verify and i <= a.verify:      # exact-equality check vs the original path
            f = np.load(src); f = f.transpose(1, 0, 2)
            ref = np.array([process_feat(c, 32) for c in f], dtype=np.float32)
            print("   verify %-28s identical=%s maxdiff=%.3e" %
                  (n, np.array_equal(o, ref), float(np.abs(o - ref).max())), flush=True)
        np.save(dst + '.part.npy', o)
        os.replace(dst + '.part.npy', dst)
        if i % 25 == 0:
            print("[%d/%d]" % (i, len(names)), flush=True)
    print("shard %d done (%d)" % (a.shard, len(names)), flush=True)
