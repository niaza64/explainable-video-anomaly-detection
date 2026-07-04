#!/usr/bin/env python3
"""
Qwen3-VL-32B + LoRA v3 inference on TCSC cluster.
Same inference pipeline as v1/v2, just pointing at the v3 adapter and v3
results directory. run_tag = finetune_v3.
"""
import argparse, base64, gc, json, os, tempfile, time
from pathlib import Path
import torch
from tqdm import tqdm

V1_BASE = Path("/scratch/svc_td_ppml/qrx527/niaz_research")
V3_BASE = Path("/scratch/svc_td_ppml/qrx527/niaz_research_v3_enriched_labels_single_strategy_with_normal_negatives")

DEFAULT_RTFM_OUTPUTS = Path(os.environ.get("RTFM_OUTPUTS", V1_BASE / "rtfm_outputs"))
DEFAULT_LORA_DIR     = Path(os.environ.get("LORA_DIR",     V3_BASE / "models" / "qwen3vl32_rtfm_lora_v3"))
DEFAULT_ANNOTATIONS  = Path(os.environ.get("ANNOTATIONS",  V1_BASE / "annotations.json"))
DEFAULT_RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR",  V3_BASE / "tcsc_v3_results"))
BASE_MODEL_ID        = "Qwen/Qwen3-VL-32B-Instruct"
MAX_NEW_TOKENS       = 300
RUN_TAG              = "finetune_v3"

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


def load_model(lora_dir):
    from peft import PeftModel
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
    print(f"Loading base: {BASE_MODEL_ID}")
    base = QwenVLModel.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16,
                                        device_map="auto", trust_remote_code=True)
    print(f"Applying LoRA v3 from: {lora_dir}")
    m = PeftModel.from_pretrained(base, str(lora_dir)); m.eval()
    p = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True,
                                       min_pixels=256*28*28, max_pixels=512*28*28)
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return m, p


def run_inference(model, processor, body):
    from qwen_vl_utils import process_vision_info
    tmps = []
    try:
        content = [{"type":"text","text":body["intro_text"]}]
        for fr in body["frames"]:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(base64.b64decode(fr["b64"])); tmp.close(); tmps.append(tmp.name)
            content.append({"type":"text","text":fr["label"]})
            content.append({"type":"image","image":f"file://{tmp.name}"})
        msgs = [{"role":"system","content":body["system_prompt"]},
                {"role":"user","content":content}]
        ti = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)[:2]
        inputs = processor(text=[ti], images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=body.get("max_new_tokens", MAX_NEW_TOKENS), do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        text = raw
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
        try:    return json.loads(text.strip()).get("explanation", raw)
        except: return raw
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        for p in tmps:
            try: os.unlink(p)
            except: pass
        gc.collect(); torch.cuda.empty_cache()


def build_request(meta, video_dir):
    frames = meta.get("extracted_frames", [])
    segs   = meta.get("anomalous_segments", [])
    def lbl(s):
        a = s.get("start_snippet", s.get("start","?")); b = s.get("end_snippet", s.get("end","?"))
        return f"snippets {a}-{b}"
    seg_str = ", ".join(lbl(s) for s in segs) if segs else "unknown"
    score_str = ", ".join(f"snippet {f['snippet_idx']}={f['score']:.3f}" for f in frames)
    intro = (
        f"This video has {meta['n_segments']} temporal snippets (~16 frames each). "
        f"RTFM flagged these segments as anomalous: [{seg_str}]. "
        f"Video-level anomaly gate score: {meta['gate_score']:.3f} (threshold=0.2). "
        f"Below are {len(frames)} frames from the anomalous segments. "
        f"Per-snippet anomaly scores: [{score_str}]."
    )
    items = []
    for fr in frames:
        p = video_dir / fr["file"]
        if not p.exists(): continue
        items.append({
            "label": f"Frame from snippet {fr['snippet_idx']} (frame #{fr['frame_num']}, anomaly score: {fr['score']:.3f}):",
            "b64": base64.b64encode(p.read_bytes()).decode("utf-8"),
        })
    return {"system_prompt": SYSTEM_PROMPT, "intro_text": intro, "frames": items, "max_new_tokens": MAX_NEW_TOKENS}


def call_judge(client, human, ai):
    msg = f'HUMAN ground-truth explanation:\n"{human}"\n\nAI-generated explanation:\n"{ai}"'
    try:
        r = client.chat.completions.create(model="gpt-4o",
            messages=[{"role":"system","content":JUDGE_PROMPT}, {"role":"user","content":msg}],
            max_tokens=300, response_format={"type":"json_object"})
        return json.loads(r.choices[0].message.content)
    except Exception as e:
        return {k: None for k in ["correctness","specificity","completeness","fluency"]} | {"justification": f"ERROR: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtfm-outputs", type=Path, default=DEFAULT_RTFM_OUTPUTS)
    ap.add_argument("--lora-dir",     type=Path, default=DEFAULT_LORA_DIR)
    ap.add_argument("--annotations",  type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--results-dir",  type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--skip-judge",   action="store_true")
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {args.results_dir}")

    annotations = {}
    if args.annotations.exists():
        annotations = {e["video_id"]: e for e in json.loads(args.annotations.read_text()) if "video_id" in e}
        print(f"Loaded {len(annotations)} annotations")

    metas = {}
    for vd in sorted(args.rtfm_outputs.iterdir()):
        if not vd.is_dir(): continue
        mf = vd / "metadata.json"
        if not mf.exists(): continue
        m = json.loads(mf.read_text()); metas[m["video_id"]] = (m, vd)
    print(f"Found {len(metas)} videos")

    model, proc = load_model(args.lora_dir)

    oc = None
    if not args.skip_judge:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            from openai import OpenAI
            oc = OpenAI(api_key=key); print("GPT-4o judge enabled")
        else:
            print("No OPENAI_API_KEY — skipping judge")

    all_expl, all_judge = [], []
    for vname, (meta, vdir) in tqdm(sorted(metas.items()), desc="Qwen v3 inference"):
        out = args.results_dir / vname
        out.mkdir(parents=True, exist_ok=True)
        body = build_request(meta, vdir)
        expl = run_inference(model, proc, body)
        tqdm.write(f"  {vname}: {expl[:80]}{'...' if len(expl)>80 else ''}")

        er = {"video_id": vname, "model": f"{BASE_MODEL_ID}+LoRA_v3", "run_tag": RUN_TAG,
              "gate_score": meta.get("gate_score"), "n_segments": meta.get("n_segments"),
              "anomalous_segments": meta.get("anomalous_segments", []),
              "n_frames_sent": len(body["frames"]), "explanation": expl}
        all_expl.append(er)
        (out / f"explanation_qwen_{RUN_TAG}.json").write_text(json.dumps(er, indent=2))

        if oc and not expl.startswith("ERROR"):
            cid = vname.split("_")[1] if "_" in vname else vname
            anom = len(cid) == 4 and vname in annotations
            h = annotations[vname]["explanation"] if anom else NORMAL_FP_GT
            vt = "anomalous" if anom else "normal_FP"
            sc = call_judge(oc, h, expl)
            tqdm.write(f"    judge C={sc.get('correctness')} S={sc.get('specificity')} "
                       f"Co={sc.get('completeness')} F={sc.get('fluency')}")
            jr = {"video_id": vname, "model": f"{BASE_MODEL_ID}+LoRA_v3", "run_tag": RUN_TAG,
                  "video_type": vt, "human_explanation": h, "ai_explanation": expl,
                  "gate_score": meta.get("gate_score"), "scores": sc}
            all_judge.append(jr)
            (out / f"judge_qwen_{RUN_TAG}.json").write_text(json.dumps(jr, indent=2))
            time.sleep(0.4)

    (args.results_dir / f"qwen_{RUN_TAG}_explanations_summary.json").write_text(json.dumps(all_expl, indent=2))
    if all_judge:
        (args.results_dir / f"qwen_{RUN_TAG}_judge_summary.json").write_text(json.dumps(all_judge, indent=2))
        import numpy as np
        anom = [r for r in all_judge if r["video_type"] == "anomalous"]
        nfp  = [r for r in all_judge if r["video_type"] == "normal_FP"]
        if anom:
            print(f"\n{'='*60}\n  Qwen3-VL-32B+LoRA_v3  ANOMALOUS (n={len(anom)})\n{'='*60}")
            for m in ["correctness","specificity","completeness","fluency"]:
                vals = [r["scores"].get(m) for r in anom if r["scores"].get(m) is not None]
                arr  = np.array(vals); print(f"  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}")
        if nfp:
            print(f"\n{'='*60}\n  Qwen3-VL-32B+LoRA_v3  NORMAL-FP (n={len(nfp)})\n{'='*60}")
            for m in ["correctness","specificity","completeness","fluency"]:
                vals = [r["scores"].get(m) for r in nfp if r["scores"].get(m) is not None]
                arr  = np.array(vals); print(f"  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
