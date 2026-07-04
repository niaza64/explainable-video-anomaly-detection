#!/usr/bin/env python3
# Requires OPENAI_API_KEY and HF_TOKEN in the environment (see .env.example);
# no credentials are hardcoded.
"""
Retrieval-Augmented In-Context inference with Qwen3-VL-32B (NO LoRA).

Pipeline per test video:
  1. CLIP-embed test frames -> mean -> query vector (512-d)
  2. Cosine retrieve top-K=3 anomalous TRAIN videos from index
  3. Build a multi-turn multimodal prompt:
       SYSTEM (same as v1)
       USER: "Here are 3 examples first. Then I will ask you about a new video."
             [example_1 frames + scores + enriched explanation]
             [example_2 ...]
             [example_3 ...]
             "Now describe this video:" [test frames + test scores]
  4. Generate with base Qwen3-VL-32B
  5. Judge with GPT-4o (same prompt as v1/v2/v3)

Train pool: 63 anomalous videos from v3 dataset (single-strategy frames
under .../data/out_v3/images/anom/<vid>/*.jpg) with v2 enriched labels
(.../data/out_v3/manifest.jsonl assistant fields).
"""
from __future__ import annotations
import argparse, base64, gc, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────────────────────
V1_BASE  = Path("/scratch/svc_td_ppml/qrx527/niaz_research")
V3_BASE  = Path("/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives")
RAG_BASE = Path("/scratch/svc_td_ppml/qrx527/niaz_research_rag_in_context_with_clip_embeddings")

TRAIN_IMG_ROOT   = V3_BASE / "data" / "out_v3" / "images" / "anom"
TRAIN_MANIFEST   = V3_BASE / "data" / "out_v3" / "manifest.jsonl"

DEFAULT_RTFM_OUTPUTS = Path(os.environ.get("RTFM_OUTPUTS", V1_BASE / "rtfm_outputs"))
DEFAULT_ANNOTATIONS  = Path(os.environ.get("ANNOTATIONS",  V1_BASE / "annotations.json"))
DEFAULT_RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR",  RAG_BASE / "tcsc_rag_results"))

BASE_MODEL_ID  = "Qwen/Qwen3-VL-32B-Instruct"
CLIP_MODEL_ID  = "ViT-B-32"            # open_clip
CLIP_PRETRAINED= "openai"
TOP_K          = 3
MAX_NEW_TOKENS = 300
RUN_TAG        = "rag_top3_clip"

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


# ── CLIP embedding ──────────────────────────────────────────────────────────

def load_clip():
    import open_clip
    print(f"Loading CLIP {CLIP_MODEL_ID}/{CLIP_PRETRAINED} on CUDA")
    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_ID, pretrained=CLIP_PRETRAINED)
    model = model.cuda().eval()
    return model, preprocess


def embed_frames(clip_model, preprocess, image_paths) -> np.ndarray:
    """Return one mean-pooled embedding (512-d, L2-normalized) for a list of frame paths."""
    if not image_paths:
        return np.zeros(512, dtype=np.float32)
    batch = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(preprocess(img))
        except Exception as e:
            print(f"  WARN: skip frame {p}: {e}")
    if not batch:
        return np.zeros(512, dtype=np.float32)
    x = torch.stack(batch).cuda()
    with torch.no_grad():
        feats = clip_model.encode_image(x).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mean = feats.mean(dim=0)
    mean = mean / mean.norm().clamp(min=1e-8)
    return mean.cpu().numpy().astype(np.float32)


# ── Train pool index ────────────────────────────────────────────────────────

def build_train_index(clip_model, preprocess) -> list[dict]:
    """Build the index of (embedding, frame_paths, enriched_label, scores) for the 63 anom train videos."""
    rows = []
    with open(TRAIN_MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            if r.get("video_type") != "anomalous":
                continue
            rows.append(r)
    print(f"Found {len(rows)} anomalous train rows in v3 manifest")

    index = []
    for r in tqdm(rows, desc="indexing train pool (CLIP)"):
        # images are paths like "images/anom/<vid>/<file>.jpg" relative to v3/data/out_v3
        abs_paths = [V3_BASE / "data" / "out_v3" / rel for rel in r["images"]]
        abs_paths = [p for p in abs_paths if p.is_file()]
        if not abs_paths:
            print(f"  WARN: {r['video_id']} no images found, skipping")
            continue
        emb = embed_frames(clip_model, preprocess, abs_paths)
        # extract enriched label from the assistant turn
        a = next(t["value"] for t in r["conversations"] if t["from"] == "assistant")
        try:
            label = json.loads(a).get("explanation", a)
        except Exception:
            label = a
        index.append({
            "video_id":  r["video_id"],
            "embedding": emb,
            "frames":    [str(p) for p in abs_paths],
            "scores":    r["scores"],
            "label":     label,
        })
    print(f"Train index built: {len(index)} entries")
    return index


# ── Retrieval ───────────────────────────────────────────────────────────────

def retrieve(query_emb: np.ndarray, index: list[dict], k: int = TOP_K) -> list[dict]:
    if query_emb.sum() == 0:
        return index[:k]
    embs = np.stack([e["embedding"] for e in index])
    sims = embs @ query_emb
    top  = np.argsort(-sims)[:k]
    out = []
    for rank, i in enumerate(top, 1):
        e = dict(index[i])
        e["rank"] = rank
        e["sim"]  = float(sims[i])
        out.append(e)
    return out


# ── Qwen inference ──────────────────────────────────────────────────────────

def load_qwen():
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
        print("Using Qwen3VLForConditionalGeneration")
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
    print(f"Loading {BASE_MODEL_ID}")
    model = QwenVLModel.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16,
                                         device_map="auto", trust_remote_code=True)
    model.eval()
    proc = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True,
                                          min_pixels=256*28*28, max_pixels=512*28*28)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, proc


def build_request_with_examples(meta, video_dir, retrieved, max_test_frames=12):
    """Build a single user-message body containing the K examples and the test video."""
    test_frames = meta.get("extracted_frames", [])
    if len(test_frames) > max_test_frames:
        # uniformly subsample
        idx = sorted({int(round(i * (len(test_frames)-1) / (max_test_frames-1))) for i in range(max_test_frames)})
        test_frames = [test_frames[i] for i in idx]

    parts = []
    # Intro
    parts.append({"type": "text", "text":
        f"You will be shown {len(retrieved)} EXAMPLE videos with their correct explanations, "
        f"then asked to explain a NEW video in the same style. Each example shows frames in temporal order "
        f"with per-snippet anomaly scores, followed by the gold explanation."})

    # In-context examples
    max_frames_per_example = 4
    for ex in retrieved:
        frames = ex["frames"]
        scores = ex["scores"]
        if len(frames) > max_frames_per_example:
            idx = sorted({int(round(i * (len(frames)-1) / (max_frames_per_example-1))) for i in range(max_frames_per_example)})
            frames = [frames[i] for i in idx]
            scores = [scores[i] for i in idx]
        parts.append({"type": "text",
                      "text": f"\n--- EXAMPLE {ex['rank']} (similarity={ex['sim']:.3f}) ---\nFrames:"})
        for fp in frames:
            try:
                parts.append({"type": "image", "image": f"file://{fp}"})
            except Exception as e:
                pass
        parts.append({"type": "text",
                      "text": f"Per-snippet anomaly scores: [{', '.join(f'{x:.4f}' for x in scores)}]\n"
                              f'Correct explanation: "{ex["label"]}"'})

    # Test query
    parts.append({"type": "text", "text":
        f"\n=== NEW VIDEO (please explain) ===\n"
        f"This video has {meta.get('n_segments','?')} temporal snippets. RTFM gate score: "
        f"{meta.get('gate_score', 0.0):.3f}.\nFrames (temporal order):"})

    tmp_paths = []
    score_list = []
    for fr in test_frames:
        p = video_dir / fr["file"]
        if not p.is_file():
            continue
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(p.read_bytes()); tmp.close(); tmp_paths.append(tmp.name)
        parts.append({"type": "image", "image": f"file://{tmp.name}"})
        score_list.append(fr["score"])
    parts.append({"type": "text",
                  "text": f"Per-snippet anomaly scores for this new video: [{', '.join(f'{x:.4f}' for x in score_list)}]\n"
                          f"Now describe the anomalous activity in this new video."})
    return parts, tmp_paths


def run_inference(model, processor, content_parts):
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content_parts}]
    text_in = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)[:2]
    inputs = processor(text=[text_in], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    text = raw
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip()).get("explanation", raw)
    except json.JSONDecodeError:
        return raw


def call_judge(client, human, ai):
    msg = f'HUMAN ground-truth explanation:\n"{human}"\n\nAI-generated explanation:\n"{ai}"'
    try:
        r = client.chat.completions.create(model="gpt-4o",
            messages=[{"role":"system","content":JUDGE_PROMPT},
                      {"role":"user","content":msg}],
            max_tokens=300, response_format={"type":"json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {k: None for k in ["correctness","specificity","completeness","fluency"]} | {"justification": f"ERROR: {e}"}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtfm-outputs", type=Path, default=DEFAULT_RTFM_OUTPUTS)
    ap.add_argument("--annotations",  type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--results-dir",  type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--skip-judge",   action="store_true")
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {args.results_dir}")

    # Annotations + test metadata
    annotations = {e["video_id"]: e for e in json.loads(args.annotations.read_text()) if "video_id" in e}
    print(f"Loaded {len(annotations)} annotations")
    metas = {}
    for vd in sorted(args.rtfm_outputs.iterdir()):
        if not vd.is_dir(): continue
        mf = vd / "metadata.json"
        if not mf.exists(): continue
        m = json.loads(mf.read_text()); metas[m["video_id"]] = (m, vd)
    print(f"Found {len(metas)} test videos")

    # Build train index (CLIP embeddings)
    clip_model, preprocess = load_clip()
    train_index = build_train_index(clip_model, preprocess)
    (args.results_dir / "train_index_meta.json").write_text(json.dumps(
        [{"video_id": e["video_id"], "label_first50": e["label"][:50]} for e in train_index], indent=2))

    # Precompute test embeddings and retrieval mapping (logged)
    print(f"Precomputing test embeddings + retrieval (K={TOP_K})")
    retrieval_log = {}
    test_embs = {}
    for vname, (meta, vdir) in tqdm(sorted(metas.items()), desc="test embeddings"):
        frames = meta.get("extracted_frames", [])
        paths = [vdir / fr["file"] for fr in frames if (vdir / fr["file"]).is_file()]
        if not paths:
            print(f"  WARN: {vname} no frames")
            continue
        test_embs[vname] = embed_frames(clip_model, preprocess, paths)
        ret = retrieve(test_embs[vname], train_index, k=TOP_K)
        retrieval_log[vname] = [{"video_id": r["video_id"], "sim": r["sim"], "rank": r["rank"]} for r in ret]
    (args.results_dir / "retrieval_top3.json").write_text(json.dumps(retrieval_log, indent=2))

    # Free CLIP memory before loading Qwen
    del clip_model
    torch.cuda.empty_cache(); gc.collect()

    # Load Qwen
    qwen_model, qwen_proc = load_qwen()

    # OpenAI judge
    oc = None
    if not args.skip_judge:
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"]); print("GPT-4o judge enabled")
        else:
            print("No OPENAI_API_KEY, judge skipped")

    # Inference loop
    all_expl, all_judge = [], []
    for vname, (meta, vdir) in tqdm(sorted(metas.items()), desc="RAG inference"):
        if vname not in test_embs:
            continue
        ret = retrieve(test_embs[vname], train_index, k=TOP_K)
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
        tqdm.write(f"  {vname} ← retrieved {[r['video_id'] for r in ret]}: {expl[:80]}...")

        out = args.results_dir / vname; out.mkdir(parents=True, exist_ok=True)
        er = {"video_id": vname, "model": BASE_MODEL_ID, "run_tag": RUN_TAG,
              "retrieved_train_ids": [r["video_id"] for r in ret],
              "retrieval_similarities": [r["sim"] for r in ret],
              "gate_score": meta.get("gate_score"), "n_segments": meta.get("n_segments"),
              "n_frames_sent_test": len([p for p in content_parts if p.get("type")=="image"]),
              "explanation": expl}
        all_expl.append(er)
        (out / f"explanation_{RUN_TAG}.json").write_text(json.dumps(er, indent=2))

        if oc and not expl.startswith("ERROR"):
            cid = vname.split("_")[1] if "_" in vname else vname
            anom = len(cid) == 4 and vname in annotations
            h = annotations[vname]["explanation"] if anom else NORMAL_FP_GT
            vt = "anomalous" if anom else "normal_FP"
            sc = call_judge(oc, h, expl)
            tqdm.write(f"    judge C={sc.get('correctness')} S={sc.get('specificity')} "
                       f"Co={sc.get('completeness')} F={sc.get('fluency')}")
            jr = {"video_id": vname, "model": BASE_MODEL_ID, "run_tag": RUN_TAG,
                  "retrieved_train_ids": [r["video_id"] for r in ret],
                  "video_type": vt, "human_explanation": h, "ai_explanation": expl,
                  "gate_score": meta.get("gate_score"), "scores": sc}
            all_judge.append(jr)
            (out / f"judge_{RUN_TAG}.json").write_text(json.dumps(jr, indent=2))
            time.sleep(0.4)

    (args.results_dir / f"qwen_{RUN_TAG}_explanations_summary.json").write_text(json.dumps(all_expl, indent=2))
    if all_judge:
        (args.results_dir / f"qwen_{RUN_TAG}_judge_summary.json").write_text(json.dumps(all_judge, indent=2))
        anom = [r for r in all_judge if r["video_type"] == "anomalous"]
        if anom:
            print(f"\n{'='*60}\n  Qwen3-VL-32B + RAG top-{TOP_K} CLIP  ANOMALOUS (n={len(anom)})\n{'='*60}")
            for m in ["correctness","specificity","completeness","fluency"]:
                vals = [r["scores"].get(m) for r in anom if r["scores"].get(m) is not None]
                arr  = np.array(vals); print(f"  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}")
        nfp = [r for r in all_judge if r["video_type"] == "normal_FP"]
        if nfp:
            print(f"\n{'='*60}\n  Qwen3-VL-32B + RAG top-{TOP_K} CLIP  NORMAL-FP (n={len(nfp)})\n{'='*60}")
            for m in ["correctness","specificity","completeness","fluency"]:
                vals = [r["scores"].get(m) for r in nfp if r["scores"].get(m) is not None]
                arr  = np.array(vals); print(f"  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
