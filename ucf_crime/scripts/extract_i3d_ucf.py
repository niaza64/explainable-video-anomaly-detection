"""
Extract I3D ResNet50 10-crop features from UCF-Crime .mp4 videos,
matching the RTFM authors' official feature format.

Output: .npy of shape (T, 10, 2048) where T = ceil(num_frames / 16)
(the official UCF test features use ceil, verified against 5 videos).
"""
import os, sys, argparse, glob
import numpy as np
import torch
import av
from PIL import Image
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pytorch-resnet3d'))
from models.resnet import I3Res50

PRETRAINED = os.path.join(HERE, 'pytorch-resnet3d', 'pretrained', 'i3d_r50_kinetics.pth')
CLIP_LEN, CROP, SHORT = 16, 224, 256
NORMS = {
    'slowfast': ([0.45,0.45,0.45], [0.225,0.225,0.225]),
    'imagenet': ([0.485,0.456,0.406], [0.229,0.224,0.225]),
    'none':     ([0.0,0.0,0.0], [1.0,1.0,1.0]),
    'raw255':   ([0.0,0.0,0.0], [1/255.,1/255.,1/255.]),
    'caffe':    ([0.485,0.456,0.406], [1/255.,1/255.,1/255.]),
}
CFG = {'norm':'slowfast','resize':'short256','bgr':False,'weights':None}
_norm = None
def set_norm(k):
    global _norm
    m,s = NORMS[k]; _norm = transforms.Normalize(mean=m, std=s)


def frame_iter(path):
    c = av.open(path)
    try:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24")
    finally:
        c.close()


def resize_short(img_np):
    if CFG['bgr']:
        img_np = img_np[:, :, ::-1].copy()
    im = Image.fromarray(img_np)
    w, h = im.size
    mode = CFG['resize']
    if mode == 'fixed256x340':
        return im.resize((340, 256), Image.BILINEAR)
    if mode == 'fixed224':
        return im.resize((224, 224), Image.BILINEAR)
    if mode == 'short256bicubic':
        if w < h: nw, nh = SHORT, int(round(h*SHORT/w))
        else:     nh, nw = SHORT, int(round(w*SHORT/h))
        return im.resize((nw, nh), Image.BICUBIC)
    if w < h: nw, nh = SHORT, int(round(h*SHORT/w))
    else:     nh, nw = SHORT, int(round(w*SHORT/h))
    return im.resize((nw, nh), Image.BILINEAR)


def ten_crop(t):
    """t: (T,C,H,W) -> list of 10 tensors (T,C,CROP,CROP)"""
    _, _, h, w = t.shape
    pos = [(0, 0), (0, w - CROP), (h - CROP, 0), (h - CROP, w - CROP),
           ((h - CROP) // 2, (w - CROP) // 2)]
    crops = []
    for top, left in pos:
        c = t[:, :, top:top + CROP, left:left + CROP]
        crops.append(c)
        crops.append(torch.flip(c, dims=[3]))
    return crops


def build_model(device):
    m = I3Res50(num_classes=400, use_nl=False)
    wp = CFG['weights'] or PRETRAINED
    sd = torch.load(wp, map_location='cpu')
    if not isinstance(sd, dict) or 'conv1.weight' not in sd:
        for k in ('state_dict','model_state','blobs'):
            if isinstance(sd, dict) and k in sd: sd = sd[k]; break
    missing = m.load_state_dict(sd, strict=False)
    print('   weights=%s missing=%d unexpected=%d' % (os.path.basename(wp),
          len(missing.missing_keys), len(missing.unexpected_keys)), flush=True)
    return m.to(device).eval()


@torch.no_grad()
def forward_clips(model, x):
    """x: (B,C,T,H,W) -> (B,2048)"""
    x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool1(x)
    x = model.layer1(x); x = model.maxpool2(x)
    x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
    x = model.avgpool(x)
    return x.view(x.shape[0], -1)


def n_frames_meta(path):
    c = av.open(path)
    try:
        st = c.streams.video[0]
        n = st.frames or 0
        if n <= 0 and st.duration and st.average_rate:
            n = int(float(st.duration * st.time_base) * float(st.average_rate))
        return int(n)
    finally:
        c.close()


def extract(model, path, device, pad="slide", out_path=None, memmap_min=2000):
    """Stream-decode. For long videos write straight into a .npy memmap so host
    RAM stays flat (Normal_Videos308 is 976503 frames -> 61032 snippets ~5 GB)."""
    from collections import deque
    import gc

    T_meta = n_frames_meta(path)
    T_est = int(np.ceil(T_meta / CLIP_LEN)) if T_meta > 0 else 0
    use_mm = out_path is not None and T_est >= memmap_min

    mm = None
    if use_mm:
        mm = np.lib.format.open_memmap(out_path + '.part', mode='w+',
                                       dtype=np.float32, shape=(T_est, 10, 2048))
        print('   memmap %s snippets (%.1f GB)' % (T_est, T_est*10*2048*4/1e9), flush=True)

    feats, buf, tail, n, t = [], [], deque(maxlen=CLIP_LEN), 0, 0

    def emit(frames16):
        nonlocal t
        clip = torch.stack(frames16)
        crops = ten_crop(clip)
        batch = torch.stack([c.permute(1, 0, 2, 3) for c in crops]).to(device)
        f = forward_clips(model, batch).cpu().numpy()
        if use_mm:
            if t < T_est:
                mm[t] = f
        else:
            feats.append(f)
        t += 1

    for fr in frame_iter(path):
        x = _norm(transforms.functional.to_tensor(resize_short(fr)))
        buf.append(x); tail.append(x); n += 1
        if len(buf) == CLIP_LEN:
            emit(buf); buf = []

    if n == 0:
        if mm is not None:
            del mm; os.remove(out_path + '.part')
        return None

    if buf:
        if pad == "slide" and n >= CLIP_LEN:
            emit(list(tail))
        else:
            emit(buf + [buf[-1]] * (CLIP_LEN - len(buf)))

    if not use_mm:
        return np.stack(feats, axis=0)

    mm.flush()
    if t == T_est:
        del mm; gc.collect()
        os.replace(out_path + '.part', out_path)
    else:  # metadata frame count was wrong -> rewrite at the true length
        print('   NOTE metadata said %d snippets, got %d; trimming' % (T_est, t), flush=True)
        src = mm
        dst = np.lib.format.open_memmap(out_path + '.fix', mode='w+',
                                        dtype=np.float32, shape=(t, 10, 2048))
        for a in range(0, t, 512):
            dst[a:a+512] = src[a:a+512]
        dst.flush()
        del dst, src, mm; gc.collect()
        os.remove(out_path + '.part')
        os.replace(out_path + '.fix', out_path)
    return 'WROTE'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', nargs='*', default=None, help='basenames without _x264')
    ap.add_argument('--video-root', default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pad', default='slide', choices=['slide', 'repeat'])
    ap.add_argument('--only-missing-vs', default=None,
                    help='dir of official features; skip videos already present there')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--norm', default='slowfast', choices=list(NORMS))
    ap.add_argument('--resize', default='short256',
                    choices=['short256','short256bicubic','fixed256x340','fixed224'])
    ap.add_argument('--bgr', action='store_true')
    ap.add_argument('--weights', default=None)
    args = ap.parse_args()
    CFG['norm'], CFG['resize'], CFG['bgr'], CFG['weights'] = args.norm, args.resize, args.bgr, args.weights
    set_norm(args.norm)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device, flush=True)
    model = build_model(device)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.video_root, '**', '*.mp4'), recursive=True))
    byname = {}
    for p in paths:
        b = os.path.basename(p)[:-4]
        byname.setdefault(b, p)
        byname.setdefault(b.replace('_x264',''), p)
    names = args.videos if args.videos else sorted(byname)

    if args.only_missing_vs:
        have = {f.replace('_x264_i3d.npy', '') for f in os.listdir(args.only_missing_vs)}
        names = [n for n in names if n not in have]
        print("skipping %d already-official; %d to extract" % (len(have), len(names)), flush=True)

    if args.limit:
        names = names[:args.limit]

    for i, n in enumerate(names, 1):
        base = n if n.endswith('_x264') else n + '_x264'
        out = os.path.join(args.out_dir, base + '_i3d.npy')
        if os.path.exists(out):
            print("[%d/%d] skip %s" % (i, len(names), n), flush=True); continue
        p = byname.get(n)
        if not p:
            print("[%d/%d] MISSING VIDEO %s" % (i, len(names), n), flush=True); continue
        f = extract(model, p, device, pad=args.pad, out_path=out)
        if f is None:
            print("[%d/%d] EMPTY %s" % (i, len(names), n), flush=True); continue
        if isinstance(f, str):
            print("[%d/%d] %s -> %s (memmap)" % (i, len(names), n, np.load(out, mmap_mode='r').shape), flush=True)
        else:
            np.save(out, f.astype(np.float32))
            print("[%d/%d] %s -> %s" % (i, len(names), n, f.shape), flush=True)
        del f
        import gc; gc.collect()


if __name__ == '__main__':
    main()
