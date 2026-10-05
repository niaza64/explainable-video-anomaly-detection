import av, os, glob, numpy as np, sys, math
W="/scratch/svc_td_ppml/qrx527/niaz_research_ucf_crime_separate_workspace"
vids={}
for p in glob.glob(W+"/data/videos_train/**/*.mp4",recursive=True)+glob.glob(W+"/data/videos_test/**/*.mp4",recursive=True):
    vids.setdefault(os.path.basename(p)[:-4], p)

def meta_frames(p):
    c=av.open(p)
    try:
        st=c.streams.video[0]; n=st.frames or 0
        if n<=0 and st.duration and st.average_rate:
            n=int(float(st.duration*st.time_base)*float(st.average_rate))
        return int(n)
    finally: c.close()

def check(split, featdir, names):
    bad=[]; ok=0; tot_snip=0
    for nm in names:
        base = nm if nm.endswith("_x264") else nm+"_x264"
        f=os.path.join(featdir, base+"_i3d.npy")
        if not os.path.exists(f): bad.append((nm,"MISSING FEATURE")); continue
        try:
            a=np.load(f, mmap_mode='r')
        except Exception as e:
            bad.append((nm,"UNREADABLE %s"%type(e).__name__)); continue
        if a.ndim!=3 or a.shape[1]!=10 or a.shape[2]!=2048:
            bad.append((nm,"BAD SHAPE %s"%(a.shape,))); continue
        if a.dtype!=np.float32: bad.append((nm,"DTYPE %s"%a.dtype)); continue
        vp=vids.get(nm) or vids.get(base)
        if vp:
            exp=math.ceil(meta_frames(vp)/16)
            if a.shape[0]!=exp:
                bad.append((nm,"T=%d expected ceil=%d"%(a.shape[0],exp))); continue
        tot_snip+=a.shape[0]; ok+=1
    print("[%s] ok=%d/%d  total_snippets=%d" % (split, ok, len(names), tot_snip))
    for n,r in bad[:15]: print("    BAD %-28s %s" % (n,r))
    return bad

tr=[l.strip() for l in open(W+"/data/train_names.txt") if l.strip()]
te=[l.strip() for l in open(W+"/data/test_names.txt") if l.strip()]
b1=check("TRAIN", W+"/data/i3d_train_ours", tr)
b2=check("TEST",  W+"/data/i3d_test_ours",  te)

# NaN/Inf spot check on a random sample
rng=np.random.default_rng(0)
samp=[(W+"/data/i3d_train_ours",n) for n in rng.choice(tr,12,replace=False)] + \
     [(W+"/data/i3d_test_ours", n) for n in rng.choice(te,8,replace=False)]
print("\nfinite/scale spot-check:")
for d,n in samp:
    b = n if n.endswith("_x264") else n+"_x264"
    a=np.load(os.path.join(d,b+"_i3d.npy"), mmap_mode='r')
    s=np.asarray(a[:min(20,a.shape[0])])
    print("   %-30s %-16s finite=%s min=%.3f max=%.3f L2/vec=%.2f" %
          (n, s.shape if a.shape[0]<=20 else a.shape, np.isfinite(s).all(),
           s.min(), s.max(), np.linalg.norm(s,axis=-1).mean()))
print("\nTOTAL BAD:", len(b1)+len(b2))
