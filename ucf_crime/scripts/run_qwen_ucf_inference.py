#!/usr/bin/env python3
"""Qwen3-VL-32B explanation inference on UCF-Crime, zero-shot or RAG.

Adapted from run_qwen_rag_inference.py (ShanghaiTech). Real differences:
  * --mode zeroshot|rag  - one script, so the two arms are identical apart from
    the retrieved in-context examples;
  * train pool + image root are arguments, not hardcoded v3 paths;
  * anomalous vs normal comes from the annotations record. The original inferred
    it from ShanghaiTech's 4-digit id convention (`len(cid)==4`), which is wrong
    for every UCF filename and would have mislabelled the whole run;
  * --gate-threshold selects which flagged videos to explain.
"""
import argparse, gc, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

BASE_MODEL_ID   = "Qwen/Qwen3-VL-32B-Instruct"
CLIP_MODEL_ID   = "ViT-B-32"
CLIP_PRETRAINED = "openai"
TOP_K           = 3
MAX_NEW_TOKENS  = 300

SYSTEM_PROMPT = (
    "You are a surveillance video anomaly analyst. You will be shown a set of frames sampled "
    "from a surveillance video that has been flagged as anomalous by a weakly-supervised "
    "anomaly detection model (RTFM).\n\n"
    "The frames are ordered temporally. Each frame comes from a specific temporal snippet of "
    "the video, and you are given the anomaly score for that snippet (0 = normal, 1 = highly "
    "anomalous).\n\n"
    "The frames were specifically selected from the anomalous portions of the video — they "
    "represent the onset, peak, and resolution of the detected anomaly.\n\n"
    "Your task: Based on ALL the frames and their anomaly scores together, provide a single "
    "concise explanation (2-3 sentences) of what anomalous activity is happening. Focus on:\n"
    "- WHAT is happening (the specific anomalous activity)\n"
    "- WHO/WHAT is involved (people, vehicles, objects — describe appearance)\n"
    "- WHEN in the sequence it starts and ends\n"
    "- WHY it is anomalous (how it deviates from ordinary, lawful activity in this setting)\n\n"
    'Respond with ONLY a JSON object in this exact format:\n{"explanation": "..."}'
)


# ---------------------------------------------------------------------------
# Prompt variants.
#
# Motivated by ASK-HINT (WACV 2026), which reports +30% AUC on UCF-Crime from
# fine-grained action-centric guiding questions over abstract prompting, with
# 6 grouped questions optimal (89.83%) and 12 degrading it (83.36%,
# hallucination). Our baseline prompt is abstract ("describe the anomalous
# activity"), and our measured failure mode is that the model writes fluent,
# specific descriptions of the WRONG event - exactly what a taxonomy or
# question scaffold should fix.
#
# The 13 UCF class names are public dataset metadata, not test labels - every
# UCF method assumes the label space. The per-video ground-truth class is
# never shown.
# ---------------------------------------------------------------------------

UCF_CLASSES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
               "Fighting", "RoadAccidents", "Robbery", "Shooting",
               "Shoplifting", "Stealing", "Vandalism"]

GUIDING_QUESTIONS = [
    "Do you see people confronting, attacking, fighting, or restraining each other?",
    "Is there evidence of weapons, gunfire, physical force, or law-enforcement activity?",
    "Is there any fire, smoke, or explosion?",
    "Is anyone taking, concealing, or carrying away items that do not appear to be theirs, "
    "or forcing entry into a building or vehicle?",
    "Is there a vehicle collision, a vehicle striking a person, or a sudden loss of control?",
    "Is anyone deliberately damaging, breaking, or destroying property?",
]

_BASE_TASK = (
    "Your task: Based on ALL the frames and their anomaly scores together, provide a single "
    "concise explanation (2-3 sentences) of what anomalous activity is happening. Focus on:\n"
    "- WHAT is happening (the specific anomalous activity)\n"
    "- WHO/WHAT is involved (people, vehicles, objects - describe appearance)\n"
    "- WHEN in the sequence it starts and ends\n"
    "- WHY it is anomalous (how it deviates from ordinary, lawful activity in this setting)\n\n"
)

_INTRO = (
    "You are a surveillance video anomaly analyst. You will be shown a set of frames sampled "
    "from a surveillance video that has been flagged as anomalous by a weakly-supervised "
    "anomaly detection model (RTFM).\n\n"
    "The frames are ordered temporally. Each frame comes from a specific temporal snippet of "
    "the video, and you are given the anomaly score for that snippet (0 = normal, 1 = highly "
    "anomalous).\n\n"
    "The frames were specifically selected from the anomalous portions of the video - they "
    "represent the onset, peak, and resolution of the detected anomaly.\n\n"
)

_JSON = 'Respond with ONLY a JSON object in this exact format:\n{"explanation": "..."}'
_JSON_CLS = ('Respond with ONLY a JSON object in this exact format:\n'
             '{"anomaly_type": "<one of the categories above>", "explanation": "..."}')


# ShanghaiTech taxonomy, derived from the 45 human annotations (cycling 30 mentions,
# pushing 9, skateboarding 8, running 6, vehicles 7, chasing 4, fighting 3, ...).
# ShanghaiTech ships no official class list, so this is the empirical equivalent of
# UCF's 13 published categories - same role, same fairness argument.
SH_CLASSES = ["Cycling", "Skateboarding", "Scooter riding", "Motor vehicle in a pedestrian area",
              "Running", "Chasing", "Fighting", "Pushing or shoving", "Throwing objects",
              "Jumping or climbing", "Loitering"]

SH_QUESTIONS = [
    "Is anyone riding a bicycle, scooter, skateboard or similar in a pedestrian-only area?",
    "Is a motor vehicle (car, van, truck) present on a pedestrian walkway?",
    "Is anyone running, chasing, or moving abnormally fast compared with other pedestrians?",
    "Is there physical conflict - pushing, shoving, fighting or grabbing?",
    "Is anyone throwing, dropping or scattering objects?",
    "Is anyone jumping over, climbing on, or interacting unusually with street furniture?",
]


def build_system_prompt(style):
    if style == "sh_taxonomy_questions":
        q = "\n".join(f"{i}. {x}" for i, x in enumerate(SH_QUESTIONS, 1))
        return (_INTRO +
                "This video contains exactly one of the following categories of anomalous "
                "activity:\n" + ", ".join(SH_CLASSES) + ".\n\n"
                "Before answering, consider each of these questions about the frames:\n" + q +
                "\n\nUse them to decide which category applies, then describe what you see.\n\n"
                + _BASE_TASK + _JSON_CLS)

    if style == "baseline":
        return _INTRO + _BASE_TASK + _JSON

    if style == "taxonomy":
        return (_INTRO +
                "This video contains exactly one of the following categories of anomalous "
                "activity:\n" + ", ".join(UCF_CLASSES) + ".\n\n"
                "First decide which category best matches what you see, then describe it.\n\n"
                + _BASE_TASK + _JSON_CLS)

    if style == "questions":
        q = "\n".join(f"{i}. {x}" for i, x in enumerate(GUIDING_QUESTIONS, 1))
        return (_INTRO +
                "Before answering, consider each of these questions about the frames:\n" + q +
                "\n\nUse whichever questions are relevant to decide what is actually "
                "happening.\n\n" + _BASE_TASK + _JSON)

    if style == "taxonomy_questions":
        q = "\n".join(f"{i}. {x}" for i, x in enumerate(GUIDING_QUESTIONS, 1))
        return (_INTRO +
                "This video contains exactly one of the following categories of anomalous "
                "activity:\n" + ", ".join(UCF_CLASSES) + ".\n\n"
                "Before answering, consider each of these questions about the frames:\n" + q +
                "\n\nUse them to decide which category applies, then describe what you see.\n\n"
                + _BASE_TASK + _JSON_CLS)

    if style == "cot":
        return (_INTRO +
                "Think step by step. First state what objects and people are visible and what "
                "they are doing across the frames. Then identify which single action is "
                "anomalous and why. Then give your final answer.\n\n" + _BASE_TASK +
                'Respond with ONLY a JSON object in this exact format:\n'
                '{"reasoning": "...", "explanation": "..."}')

    raise ValueError(style)


JUDGE_PROMPT = (
    "You are an impartial judge evaluating the quality of an AI-generated\n"
    "explanation of an anomalous event in a surveillance video.\n\n"
    "Score the AI explanation on these 4 criteria (each 1-5):\n"
    "- correctness  : Does the AI identify the same anomaly as the human?\n"
    "- specificity  : Does the AI mention specific details (objects, people, actions)?\n"
    "- completeness : Does the AI capture all aspects the human mentioned?\n"
    "- fluency      : Is the explanation well-written and clear?\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"correctness": 1-5, "specificity": 1-5, "completeness": 1-5, "fluency": 1-5, '
    '"justification": "..."}'
)


# ── CLIP ────────────────────────────────────────────────────────────────────
def load_clip():
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL_ID, pretrained=CLIP_PRETRAINED)
    return m.cuda().eval(), pre


@torch.no_grad()
def embed_frames(clip_model, preprocess, paths):
    from PIL import Image
    ims = []
    for p in paths:
        try:
            ims.append(preprocess(Image.open(p).convert("RGB")))
        except Exception:
            pass
    if not ims:
        return np.zeros(512, dtype=np.float32)
    x = torch.stack(ims).cuda()
    f = clip_model.encode_image(x).float()
    f = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    mu = f.mean(dim=0)
    mu = mu / mu.norm().clamp(min=1e-8)
    return mu.cpu().numpy().astype(np.float32)


def build_train_index(clip_model, preprocess, manifest, img_root):
    rows = [json.loads(l) for l in open(manifest) if l.strip()]
    rows = [r for r in rows if r.get("video_type") == "anomalous"]
    print(f"train pool rows: {len(rows)}")
    index = []
    for r in tqdm(rows, desc="indexing train pool (CLIP)"):
        paths = [Path(img_root) / rel for rel in r["images"]]
        paths = [p for p in paths if p.is_file()]
        if not paths:
            continue
        a = next(t["value"] for t in r["conversations"] if t["from"] == "assistant")
        try:
            label = json.loads(a).get("explanation", a)
        except Exception:
            label = a
        index.append({"video_id": r["video_id"], "embedding": embed_frames(clip_model, preprocess, paths),
                      "frames": [str(p) for p in paths], "scores": r["scores"], "label": label})
    print(f"train index: {len(index)}")
    return index


def retrieve(q, index, k=TOP_K):
    if q.sum() == 0:
        return index[:k]
    sims = np.stack([e["embedding"] for e in index]) @ q
    out = []
    for rank, i in enumerate(np.argsort(-sims)[:k], 1):
        e = dict(index[i]); e["rank"] = rank; e["sim"] = float(sims[i]); out.append(e)
    return out


# ── Qwen ────────────────────────────────────────────────────────────────────
def load_qwen(adapter=None):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
    model = QwenVLModel.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        print(f"attaching LoRA adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()      # fold LoRA in so generate() runs at full speed
    model.eval()
    return model, AutoProcessor.from_pretrained(BASE_MODEL_ID)


def build_request(meta, video_dir, retrieved, max_test_frames=12):
    # cap is configurable; UCF needs more frames than ShanghaiTech's short clips
    frames = meta.get("extracted_frames", [])
    frames = [f for f in frames if f.get("file")]
    if len(frames) > max_test_frames:
        idx = sorted({int(round(i * (len(frames)-1)/(max_test_frames-1))) for i in range(max_test_frames)})
        frames = [frames[i] for i in idx]

    parts, tmps = [], []
    if retrieved:
        parts.append({"type": "text", "text":
            f"You will be shown {len(retrieved)} EXAMPLE videos with their correct explanations, "
            f"then asked to explain a NEW video in the same style. Each example shows frames in "
            f"temporal order with per-snippet anomaly scores, followed by the gold explanation."})
        for ex in retrieved:
            fr, sc = ex["frames"], ex["scores"]
            if len(fr) > 4:
                idx = sorted({int(round(i*(len(fr)-1)/3)) for i in range(4)})
                fr = [fr[i] for i in idx]; sc = [sc[i] for i in idx]
            parts.append({"type": "text", "text": f"\n--- EXAMPLE {ex['rank']} (similarity={ex['sim']:.3f}) ---\nFrames:"})
            for p in fr:
                parts.append({"type": "image", "image": f"file://{p}"})
            parts.append({"type": "text", "text":
                f"Per-snippet anomaly scores: [{', '.join(f'{x:.4f}' for x in sc)}]\n"
                f'Correct explanation: "{ex["label"]}"'})
        parts.append({"type": "text", "text": "\n=== NEW VIDEO (please explain) ==="})

    parts.append({"type": "text", "text":
        f"This video has {meta.get('n_segments','?')} temporal snippets. RTFM gate score: "
        f"{meta.get('gate_score', 0.0):.3f}.\nFrames (temporal order):"})
    sl = []
    for fr in frames:
        p = Path(video_dir) / fr["file"]
        if not p.is_file():
            continue
        t = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        t.write(p.read_bytes()); t.close(); tmps.append(t.name)
        parts.append({"type": "image", "image": f"file://{t.name}"})
        sl.append(fr["score"])
    parts.append({"type": "text", "text":
        f"Per-snippet anomaly scores for this video: [{', '.join(f'{x:.4f}' for x in sl)}]\n"
        f"Now describe the anomalous activity in this video."})
    return parts, tmps


def run_inference(model, proc, parts, system_prompt=None):
    from qwen_vl_utils import process_vision_info
    msgs = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}, {"role": "user", "content": parts}]
    text_in = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)[:2]
    inputs = proc(text=[text_in], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    t = raw
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    try:
        return json.loads(t.strip()).get("explanation", raw)
    except json.JSONDecodeError:
        return raw


def call_judge(client, human, ai):
    msg = f'HUMAN ground-truth explanation:\n"{human}"\n\nAI-generated explanation:\n"{ai}"'
    try:
        r = client.chat.completions.create(model="gpt-4o",
            messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": msg}],
            max_tokens=300, temperature=0, seed=42, response_format={"type": "json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {k: None for k in ["correctness", "specificity", "completeness", "fluency"]} | \
               {"justification": f"ERROR: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zeroshot", "rag"], required=True)
    ap.add_argument("--gated-dir", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--train-img-root", default=None)
    ap.add_argument("--gate-threshold", type=float, default=0.6)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--lora-adapter", default=None, help="path to a trained LoRA adapter")
    ap.add_argument("--prompt-style", default="baseline",
                    choices=["baseline","taxonomy","questions","taxonomy_questions","cot","sh_taxonomy_questions"])
    a = ap.parse_args()
    tag = a.run_tag or (f"ucf_{a.mode}")
    SYS = build_system_prompt(a.prompt_style)
    print(f"prompt style: {a.prompt_style} ({len(SYS)} chars)", flush=True)
    out_root = Path(a.results_dir); out_root.mkdir(parents=True, exist_ok=True)

    ann = {e["video_id"]: e for e in json.load(open(a.annotations)) if "video_id" in e}
    print(f"annotations: {len(ann)}")

    metas = {}
    for vd in sorted(Path(a.gated_dir).iterdir()):
        mf = vd / "metadata.json"
        if not vd.is_dir() or not mf.exists():
            continue
        m = json.loads(mf.read_text())
        if (m.get("gate_score") or 0) > a.gate_threshold:
            metas[m["video_id"]] = (m, vd)
    print(f"videos passing gate>{a.gate_threshold}: {len(metas)}")
    if a.limit:
        metas = dict(sorted(metas.items())[:a.limit])
        print(f"limited to {len(metas)}")

    train_index, test_embs = [], {}
    if a.mode == "rag":
        clip_model, pre = load_clip()
        train_index = build_train_index(clip_model, pre, a.train_manifest, a.train_img_root)
        for v, (m, vd) in tqdm(sorted(metas.items()), desc="test embeddings"):
            paths = [vd / f["file"] for f in m.get("extracted_frames", []) if f.get("file") and (vd / f["file"]).is_file()]
            if paths:
                test_embs[v] = embed_frames(clip_model, pre, paths)
        (out_root / "retrieval_top3.json").write_text(json.dumps(
            {v: [{"video_id": r["video_id"], "sim": r["sim"], "rank": r["rank"]}
                 for r in retrieve(e, train_index)] for v, e in test_embs.items()}, indent=2))
        del clip_model; torch.cuda.empty_cache(); gc.collect()

    model, proc = load_qwen(a.lora_adapter)
    oc = None
    if not a.skip_judge and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"]); print("GPT-4o judge enabled")

    all_e, all_j = [], []
    for v, (m, vd) in tqdm(sorted(metas.items()), desc=f"{a.mode} inference"):
        ret = retrieve(test_embs[v], train_index) if (a.mode == "rag" and v in test_embs) else []
        parts, tmps = build_request(m, vd, ret, max_test_frames=a.max_frames)
        try:
            expl = run_inference(model, proc, parts, SYS)
        except Exception as e:
            expl = f"ERROR: {e}"
        finally:
            for p in tmps:
                try: os.unlink(p)
                except Exception: pass
            gc.collect(); torch.cuda.empty_cache()

        od = out_root / v; od.mkdir(parents=True, exist_ok=True)
        er = {"video_id": v, "model": BASE_MODEL_ID, "run_tag": tag, "mode": a.mode,
              "retrieved_train_ids": [r["video_id"] for r in ret],
              "gate_score": m.get("gate_score"), "explanation": expl}
        all_e.append(er); (od / f"explanation_{tag}.json").write_text(json.dumps(er, indent=2))
        tqdm.write(f"  {v}: {expl[:90]}")

        if oc and not expl.startswith("ERROR") and v in ann:
            rec = ann[v]
            vt = "anomalous" if rec.get("video_type") == "anomalous" else "normal_FP"
            sc = call_judge(oc, rec["explanation"], expl)
            jr = {"video_id": v, "run_tag": tag, "mode": a.mode, "video_type": vt,
                  "human_explanation": rec["explanation"], "ai_explanation": expl,
                  "gate_score": m.get("gate_score"), "scores": sc}
            all_j.append(jr); (od / f"judge_{tag}.json").write_text(json.dumps(jr, indent=2))
            time.sleep(0.3)

    (out_root / f"qwen_{tag}_explanations_summary.json").write_text(json.dumps(all_e, indent=2))
    if all_j:
        (out_root / f"qwen_{tag}_judge_summary.json").write_text(json.dumps(all_j, indent=2))
        for vt in ["anomalous", "normal_FP"]:
            rs = [r for r in all_j if r["video_type"] == vt]
            if not rs: continue
            print(f"\n===== {tag}  {vt}  (n={len(rs)}) =====")
            per = []
            for k in ["correctness", "specificity", "completeness", "fluency"]:
                vals = [r["scores"].get(k) for r in rs if r["scores"].get(k) is not None]
                if vals:
                    print(f"  {k:<14s} {np.mean(vals):.2f} ± {np.std(vals):.2f}")
                    per.append(np.mean(vals))
            if per: print(f"  {'OVERALL':<14s} {np.mean(per):.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
