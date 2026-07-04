#!/usr/bin/env python3
"""
Retrieval-encoder ablation. Fix K=3, prompt template, judge, etc. Vary only
the retrieval embedding.

  - clip   : open_clip ViT-B/32 OpenAI weights (the original RAG)
  - dinov2 : facebook/dinov2-base, mean-pooled patch tokens
  - rtfm   : RTFM snippet-score signature (resampled to length 32 via linear
             interpolation, then L2-normalized)

All three runs share one Qwen3-VL-32B load.
"""
from __future__ import annotations
import argparse, base64, gc, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

V1_BASE  = Path("/scratch/svc_td_ppml/qrx527/niaz_research")
V3_BASE  = Path("/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives")
RAG_BASE = Path("/scratch/svc_td_ppml/qrx527/niaz_research_rag_in_context_with_clip_embeddings")

TRAIN_MANIFEST   = V3_BASE / "data" / "out_v3" / "manifest.jsonl"
DEFAULT_RTFM_OUTPUTS = Path(os.environ.get("RTFM_OUTPUTS", V1_BASE / "rtfm_outputs"))
DEFAULT_ANNOTATIONS  = Path(os.environ.get("ANNOTATIONS",  V1_BASE / "annotations.json"))
DEFAULT_RESULTS_ROOT = Path(os.environ.get("RESULTS_DIR",  RAG_BASE / "tcsc_rag_encoder_ablation"))

BASE_MODEL_ID  = "Qwen/Qwen3-VL-32B-Instruct"
TOP_K          = 3
MAX_NEW_TOKENS = 300

SYSTEM_PROMPT = (
    "You are a surveillance video anomaly analyst. You will be shown a set of "
    "frames sampled from a surveillance video that has been flagged as anomalous "
    "by a weakly-supervised anomaly detection model (RTFM).\n\n"
    "The frames are ordered temporally. Each frame comes from a specific temporal "
    "snippet of the video, and you are given the anomaly score for that snippet "
    "(0 = normal, 1 = highly anomalous).\n\n"
    "The frames were specifically selected from the anomalous portions of the "
    "video — they represent the onset, peak, and resolution of the detected anomaly.\n\n"
    "Your task: Based on ALL the frames and their anomaly scores together, provide "
    "a single concise explanation (2-3 sentences) of what anomalous activity is "
    "happening. Focus on:\n"
    "- WHAT is happening (the specific anomalous activity)\n"
    "- WHO/WHAT is involved (people, vehicles, objects — describe appearance)\n"
    "- WHEN in the sequence it starts and ends\n"
    "- WHY it is anomalous (how it deviates from normal pedestrian behaviour)\n\n"
    'Respond with ONLY a JSON object in this exact format:\n{"explanation": "..."}'
)
JUDGE_PROMPT = (
    "You are an impartial judge evaluating the quality of an AI-generated "
    "explanation of an anomalous event in a surveillance video.\n\n"
    "Score the AI explanation on these 4 criteria (each 1-5):\n"
    "- correctness  : Does the AI identify the same anomaly as the human?\n"
    "- specificity  : Does the AI mention specific details (objects, people, actions)?\n"
    "- completeness : Does the AI capture all aspects the human mentioned?\n"
    "- fluency      : Is the explanation well-written and clear?\n\n"
    'Respond with ONLY a JSON object:\n'
    '{"correctness": 1-5, "specificity": 1-5, "completeness": 1-5, "fluency": 1-5, "justification": "..."}'
)
NORMAL_FP_GT = "There is nothing anomalous in this video. All pedestrians are walking normally."


# ── Encoders ───────────────────────────────────────────────────────────────

def load_clip():
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return m.cuda().eval(), pre

def embed_clip(model, preprocess, paths):
    if not paths: return np.zeros(512, dtype=np.float32)
    batch = [preprocess(Image.open(p).convert("RGB")) for p in paths]
    x = torch.stack(batch).cuda()
    with torch.no_grad():
        feats = model.encode_image(x).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mean = feats.mean(dim=0); mean = mean / mean.norm().clamp(min=1e-8)
    return mean.cpu().numpy().astype(np.float32)


def load_dinov2():
    from transformers import AutoImageProcessor, AutoModel
    print("Loading DINOv2 (facebook/dinov2-base)")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
    return model, processor

def embed_dinov2(model, processor, paths):
    if not paths: return np.zeros(768, dtype=np.float32)
    imgs = [Image.open(p).convert("RGB") for p in paths]
    x = processor(images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**x)
    # Use CLS token (out.last_hidden_state[:, 0]) for each image, then mean-pool
    feats = out.last_hidden_state[:, 0, :].float()                   # [N, 768]
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mean = feats.mean(dim=0); mean = mean / mean.norm().clamp(min=1e-8)
    return mean.cpu().numpy().astype(np.float32)


def embed_rtfm_signature(scores, target_len=32):
    """Linearly interpolate the snippet-score vector to a fixed length 32, then L2-normalize."""
    if scores is None or len(scores) == 0:
        return np.zeros(target_len, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    if len(s) == target_len:
        v = s
    else:
        xp = np.linspace(0, 1, len(s))
        xq = np.linspace(0, 1, target_len)
        v = np.interp(xq, xp, s)
    n = np.linalg.norm(v)
    if n < 1e-8: return v
    return (v / n).astype(np.float32)


# ── Qwen + retrieval + prompt (same logic as base RAG) ─────────────────────

def load_qwen():
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as Q
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as Q
    base = Q.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    base.eval()
    proc = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True,
                                          min_pixels=256*28*28, max_pixels=512*28*28)
    print(f"Qwen loaded, GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return base, proc


def retrieve(query, index, k=TOP_K):
    if query.sum() == 0: return index[:k]
    embs = np.stack([e["embedding"] for e in index])
    sims = embs @ query
    top  = np.argsort(-sims)[:k]
    return [{**index[i], "rank": r, "sim": float(sims[i])} for r, i in enumerate(top, 1)]


def build_request_with_examples(meta, video_dir, retrieved, max_test_frames=12, max_frames_per_example=4):
    test_frames = meta.get("extracted_frames", [])
    if len(test_frames) > max_test_frames:
        idx = sorted({int(round(i * (len(test_frames)-1) / (max_test_frames-1))) for i in range(max_test_frames)})
        test_frames = [test_frames[i] for i in idx]
    parts = [{"type": "text", "text": f"You will be shown {len(retrieved)} EXAMPLE videos with their correct explanations, then asked to explain a NEW video in the same style."}]
    for ex in retrieved:
        frames, scores = ex["frames"], ex["scores"]
        if len(frames) > max_frames_per_example:
            idx = sorted({int(round(i * (len(frames)-1) / (max_frames_per_example-1))) for i in range(max_frames_per_example)})
            frames = [frames[i] for i in idx]; scores = [scores[i] for i in idx]
        parts.append({"type": "text", "text": f"\n--- EXAMPLE {ex['rank']} (similarity={ex['sim']:.3f}) ---\nFrames:"})
        for fp in frames: parts.append({"type": "image", "image": f"file://{fp}"})
        parts.append({"type": "text", "text":
            f"Per-snippet anomaly scores: [{', '.join(f'{x:.4f}' for x in scores)}]\n"
            f'Correct explanation: "{ex["label"]}"'})
    parts.append({"type": "text", "text":
        f"\n=== NEW VIDEO (please explain) ===\n"
        f"This video has {meta.get('n_segments','?')} temporal snippets. RTFM gate score: {meta.get('gate_score', 0.0):.3f}.\nFrames (temporal order):"})
    tmp_paths, score_list = [], []
    for fr in test_frames:
        p = video_dir / fr["file"]
        if not p.is_file(): continue
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(p.read_bytes()); tmp.close(); tmp_paths.append(tmp.name)
        parts.append({"type": "image", "image": f"file://{tmp.name}"})
        score_list.append(fr["score"])
    parts.append({"type": "text", "text":
        f"Per-snippet anomaly scores for this new video: [{', '.join(f'{x:.4f}' for x in score_list)}]\n"
        f"Now describe the anomalous activity in this new video."})
    return parts, tmp_paths


def run_inference(model, processor, content_parts):
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content_parts}]
    text_in = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)[:2]
    inputs = processor(text=[text_in], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    text = raw
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
    try: return json.loads(text.strip()).get("explanation", raw)
    except: return raw


def call_judge(client, human, ai):
    msg = f'HUMAN ground-truth explanation:\n"{human}"\n\nAI-generated explanation:\n"{ai}"'
    try:
        r = client.chat.completions.create(model="gpt-4o",
            messages=[{"role":"system","content":JUDGE_PROMPT}, {"role":"user","content":msg}],
            max_tokens=300, response_format={"type":"json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {k: None for k in ["correctness","specificity","completeness","fluency"]} | {"justification": f"ERROR: {e}"}


# ── Index builders ─────────────────────────────────────────────────────────

def load_train_rows():
    rows = []
    with open(TRAIN_MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r.get("video_type") == "anomalous":
                rows.append(r)
    return rows


def build_index(encoder_name, train_rows, model_state, test_metas):
    """Returns (train_index, test_embs) for the chosen encoder."""
    train_index = []
    if encoder_name == "clip":
        model, preprocess = model_state
        for r in tqdm(train_rows, desc="train (CLIP)"):
            paths = [V3_BASE / "data" / "out_v3" / rel for rel in r["images"]]
            paths = [p for p in paths if p.is_file()]
            if not paths: continue
            emb = embed_clip(model, preprocess, paths)
            a = next(t["value"] for t in r["conversations"] if t["from"] == "assistant")
            try: label = json.loads(a).get("explanation", a)
            except: label = a
            train_index.append({"video_id": r["video_id"], "embedding": emb,
                                "frames": [str(p) for p in paths], "scores": r["scores"], "label": label})
        test_embs = {}
        for vname, (meta, vdir) in tqdm(test_metas.items(), desc="test (CLIP)"):
            paths = [vdir / fr["file"] for fr in meta.get("extracted_frames", []) if (vdir / fr["file"]).is_file()]
            if paths: test_embs[vname] = embed_clip(model, preprocess, paths)
    elif encoder_name == "dinov2":
        model, processor = model_state
        for r in tqdm(train_rows, desc="train (DINOv2)"):
            paths = [V3_BASE / "data" / "out_v3" / rel for rel in r["images"]]
            paths = [p for p in paths if p.is_file()]
            if not paths: continue
            emb = embed_dinov2(model, processor, paths)
            a = next(t["value"] for t in r["conversations"] if t["from"] == "assistant")
            try: label = json.loads(a).get("explanation", a)
            except: label = a
            train_index.append({"video_id": r["video_id"], "embedding": emb,
                                "frames": [str(p) for p in paths], "scores": r["scores"], "label": label})
        test_embs = {}
        for vname, (meta, vdir) in tqdm(test_metas.items(), desc="test (DINOv2)"):
            paths = [vdir / fr["file"] for fr in meta.get("extracted_frames", []) if (vdir / fr["file"]).is_file()]
            if paths: test_embs[vname] = embed_dinov2(model, processor, paths)
    elif encoder_name == "rtfm":
        # No vision model — just signature from snippet scores
        for r in tqdm(train_rows, desc="train (RTFM-sig)"):
            paths = [V3_BASE / "data" / "out_v3" / rel for rel in r["images"]]
            paths = [p for p in paths if p.is_file()]
            if not paths: continue
            emb = embed_rtfm_signature(r["scores"])
            a = next(t["value"] for t in r["conversations"] if t["from"] == "assistant")
            try: label = json.loads(a).get("explanation", a)
            except: label = a
            train_index.append({"video_id": r["video_id"], "embedding": emb,
                                "frames": [str(p) for p in paths], "scores": r["scores"], "label": label})
        test_embs = {}
        for vname, (meta, _) in test_metas.items():
            test_scores = [fr["score"] for fr in meta.get("extracted_frames", [])]
            test_embs[vname] = embed_rtfm_signature(test_scores)
    else:
        raise ValueError(encoder_name)
    return train_index, test_embs


def run_pass(encoder_name, train_index, test_embs, metas, annotations, qwen_model, qwen_proc, oc, results_root):
    out_root = results_root / encoder_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  ENCODER: {encoder_name}\n{'='*60}")
    retrieval_log = {}
    all_judge = []
    for vname, (meta, vdir) in tqdm(sorted(metas.items()), desc=encoder_name):
        if vname not in test_embs: continue
        ret = retrieve(test_embs[vname], train_index, k=TOP_K)
        retrieval_log[vname] = [{"video_id": r["video_id"], "sim": r["sim"], "rank": r["rank"]} for r in ret]
        content_parts, tmp_paths = build_request_with_examples(meta, vdir, ret)
        try:
            expl = run_inference(qwen_model, qwen_proc, content_parts)
        except Exception as e:
            expl = f"ERROR: {e}"
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except: pass
            gc.collect(); torch.cuda.empty_cache()
        if oc and not expl.startswith("ERROR"):
            cid = vname.split("_")[1] if "_" in vname else vname
            anom = len(cid) == 4 and vname in annotations
            h = annotations[vname]["explanation"] if anom else NORMAL_FP_GT
            vt = "anomalous" if anom else "normal_FP"
            sc = call_judge(oc, h, expl)
            all_judge.append({"video_id": vname, "model": BASE_MODEL_ID, "run_tag": f"rag_enc_{encoder_name}",
                              "retrieved_train_ids": [r["video_id"] for r in ret],
                              "video_type": vt, "human_explanation": h, "ai_explanation": expl, "scores": sc})
            time.sleep(0.4)
    (out_root / f"retrieval_{encoder_name}.json").write_text(json.dumps(retrieval_log, indent=2))
    (out_root / f"qwen_rag_enc_{encoder_name}_judge_summary.json").write_text(json.dumps(all_judge, indent=2))
    anom = [r for r in all_judge if r["video_type"] == "anomalous"]
    if anom:
        print(f"\n  {encoder_name.upper()} ANOMALOUS n={len(anom)}:")
        for m in ["correctness","specificity","completeness","fluency"]:
            vals = [r["scores"].get(m) for r in anom if r["scores"].get(m) is not None]
            print(f"    {m:<14s}  {np.mean(vals):.3f}")
        overall = np.mean([np.mean([r["scores"][m] for m in ["correctness","specificity","completeness","fluency"]]) for r in anom])
        print(f"    {'OVERALL':<14s}  {overall:.3f}")
    return all_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtfm-outputs", type=Path, default=DEFAULT_RTFM_OUTPUTS)
    ap.add_argument("--annotations",  type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--results-dir",  type=Path, default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--encoders",     nargs="+", default=["clip", "dinov2", "rtfm"])
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    annotations = {e["video_id"]: e for e in json.loads(args.annotations.read_text()) if "video_id" in e}
    metas = {}
    for vd in sorted(args.rtfm_outputs.iterdir()):
        if not vd.is_dir(): continue
        mf = vd / "metadata.json"
        if not mf.exists(): continue
        m = json.loads(mf.read_text()); metas[m["video_id"]] = (m, vd)
    print(f"Loaded {len(annotations)} annotations, {len(metas)} test videos")

    train_rows = load_train_rows()
    print(f"Train rows (anomalous): {len(train_rows)}")

    # Build all retrieval indices upfront (cheap CPU/small-model work)
    print("\n=== Building retrieval indices ===")
    indices = {}
    for enc in args.encoders:
        if enc == "clip":
            state = load_clip()
        elif enc == "dinov2":
            state = load_dinov2()
        elif enc == "rtfm":
            state = None
        else:
            raise ValueError(enc)
        train_idx, test_embs = build_index(enc, train_rows, state, metas)
        indices[enc] = (train_idx, test_embs)
        if state is not None:
            del state
        torch.cuda.empty_cache(); gc.collect()
    print(f"\nDone building all indices. Loading Qwen3-VL-32B for inference...")

    qwen_model, qwen_proc = load_qwen()

    oc = None
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"]); print("GPT-4o judge enabled")

    summary = []
    for enc in args.encoders:
        train_idx, test_embs = indices[enc]
        judge = run_pass(enc, train_idx, test_embs, metas, annotations, qwen_model, qwen_proc, oc, args.results_dir)
        anom = [r for r in judge if r["video_type"] == "anomalous"]
        if anom:
            s = {m: float(np.mean([r["scores"].get(m) for r in anom if r["scores"].get(m) is not None]))
                 for m in ["correctness","specificity","completeness","fluency"]}
            s["overall"] = float(np.mean(list(s.values())))
            summary.append({"encoder": enc, "n_anom": len(anom), **s})

    print(f"\n{'='*60}\n  ENCODER-ABLATION SUMMARY (anomalous test videos)\n{'='*60}")
    print(f"{'encoder':>10s} {'n':>4s} {'corr':>7s} {'spec':>7s} {'compl':>7s} {'flu':>7s} {'OVERALL':>9s}")
    for s in summary:
        print(f"{s['encoder']:>10s} {s['n_anom']:>4d} {s['correctness']:>7.3f} {s['specificity']:>7.3f} "
              f"{s['completeness']:>7.3f} {s['fluency']:>7.3f} {s['overall']:>9.3f}")
    (args.results_dir / "encoder_ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {args.results_dir / 'encoder_ablation_summary.json'}")


if __name__ == "__main__":
    main()
