#!/usr/bin/env python3
"""
Qwen3-VL-32B short-prompt A/B: zero-shot (LoRA disabled) vs LoRA fine-tuned.
Loads the base model + LoRA once; runs all videos in both modes; saves separate
summary JSONs so we get apples-to-apples comparison under the SAME short prompt.
"""

import argparse
import base64
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from tqdm import tqdm

BASE_DIR             = Path("/scratch/svc_td_ppml/qrx527/niaz_research")
DEFAULT_RTFM_OUTPUTS = Path(os.environ.get("RTFM_OUTPUTS", BASE_DIR / "rtfm_outputs"))
DEFAULT_LORA_DIR     = Path(os.environ.get("LORA_DIR",     BASE_DIR / "models" / "qwen3vl32_rtfm_lora"))
DEFAULT_ANNOTATIONS  = Path(os.environ.get("ANNOTATIONS",  BASE_DIR / "annotations.json"))
DEFAULT_RESULTS_DIR  = Path(os.environ.get("RESULTS_DIR",  BASE_DIR / "tcsc_shortprompt_results"))
BASE_MODEL_ID        = "Qwen/Qwen3-VL-32B-Instruct"
MAX_NEW_TOKENS       = 80   # enough for 15-25 words + JSON wrapper

SHORT_SYSTEM_PROMPT = (
    "You are a surveillance video anomaly analyst. Look at the frames and "
    "describe the anomalous activity in ONE sentence of 15 to 25 words. "
    "Mention the person's appearance (clothing colour) and what unusual "
    "thing they are doing in the pedestrian area. "
    "Do NOT mention snippet indices, frame numbers, anomaly scores, or timestamps. "
    'Respond with ONLY a JSON object: {"explanation": "..."}'
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


def load_model(lora_dir: Path):
    from peft import PeftModel
    from transformers import AutoProcessor
    print(f"Loading base model: {BASE_MODEL_ID}")
    try:
        from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
        print("Using Qwen3VLForConditionalGeneration")
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
        print("Falling back to Qwen2VLForConditionalGeneration")

    base = QwenVLModel.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Applying LoRA from: {lora_dir}")
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model.eval()
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, processor


def run_inference(model, processor, body: dict) -> str:
    from qwen_vl_utils import process_vision_info
    tmp_paths = []
    try:
        content = [{"type": "text", "text": body["intro_text"]}]
        for fr in body["frames"]:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(base64.b64decode(fr["b64"]))
            tmp.close()
            tmp_paths.append(tmp.name)
            content.append({"type": "text", "text": fr["label"]})
            content.append({"type": "image", "image": f"file://{tmp.name}"})
        messages = [
            {"role": "system", "content": body["system_prompt"]},
            {"role": "user",   "content": content},
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)[:2]
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=body.get("max_new_tokens", MAX_NEW_TOKENS),
                do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
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
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        for p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass
        gc.collect()
        torch.cuda.empty_cache()


def build_request(meta: dict, video_dir: Path, system_prompt: str) -> dict:
    """Short-prompt variant: keep intro minimal — no snippet/score language."""
    frames = meta.get("extracted_frames", [])
    intro = f"Below are {len(frames)} frames sampled from a surveillance video flagged as anomalous."
    frame_items = []
    for fr in frames:
        img_path = video_dir / fr["file"]
        if not img_path.exists():
            print(f"  WARN: frame not found: {img_path}")
            continue
        frame_items.append({
            "label": "Frame:",
            "b64": base64.b64encode(img_path.read_bytes()).decode("utf-8"),
        })
    return {
        "system_prompt":  system_prompt,
        "intro_text":     intro,
        "frames":         frame_items,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def call_judge(client, human_expl: str, ai_expl: str) -> dict:
    user_msg = f'HUMAN ground-truth explanation:\n"{human_expl}"\n\nAI-generated explanation:\n"{ai_expl}"'
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {k: None for k in ["correctness", "specificity", "completeness", "fluency"]} | {
            "justification": f"ERROR: {e}"
        }


def run_pass(model, processor, video_metas, annotations, results_root: Path,
             tag: str, model_label: str, openai_client) -> list:
    out_dir = results_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    all_explanations, all_judge = [], []
    print(f"\n{'='*60}\n  PASS: {tag} ({model_label})\n{'='*60}")
    for vname, (meta, vdir) in tqdm(sorted(video_metas.items()), desc=tag):
        body = build_request(meta, vdir, SHORT_SYSTEM_PROMPT)
        explanation = run_inference(model, processor, body)
        tqdm.write(f"  {vname}: {explanation[:90]}{'...' if len(explanation) > 90 else ''}")
        expl_result = {
            "video_id": vname, "model": model_label, "run_tag": tag,
            "gate_score": meta.get("gate_score"), "n_segments": meta.get("n_segments"),
            "n_frames_sent": len(body["frames"]), "explanation": explanation,
        }
        all_explanations.append(expl_result)

        if openai_client and not explanation.startswith("ERROR"):
            clip_id = vname.split("_")[1] if "_" in vname else vname
            is_anomalous = len(clip_id) == 4 and vname in annotations
            human_expl = annotations[vname]["explanation"] if is_anomalous else NORMAL_FP_GT
            video_type = "anomalous" if is_anomalous else "normal_FP"
            scores = call_judge(openai_client, human_expl, explanation)
            tqdm.write(
                f"    judge C={scores.get('correctness')} "
                f"S={scores.get('specificity')} "
                f"Co={scores.get('completeness')} "
                f"F={scores.get('fluency')}"
            )
            all_judge.append({
                "video_id": vname, "model": model_label, "run_tag": tag,
                "video_type": video_type, "human_explanation": human_expl,
                "ai_explanation": explanation, "gate_score": meta.get("gate_score"),
                "scores": scores,
            })
            time.sleep(0.4)

    (out_dir / f"explanations_summary_{tag}.json").write_text(json.dumps(all_explanations, indent=2))
    if all_judge:
        (out_dir / f"judge_summary_{tag}.json").write_text(json.dumps(all_judge, indent=2))
        import numpy as np
        anom = [r for r in all_judge if r["video_type"] == "anomalous"]
        if anom:
            print(f"\n--- {tag}  (n={len(anom)} anomalous) ---")
            for m in ["correctness", "specificity", "completeness", "fluency"]:
                vals = [r["scores"].get(m) for r in anom if r["scores"].get(m) is not None]
                arr = np.array(vals)
                print(f"  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}")
    return all_judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtfm-outputs", type=Path, default=DEFAULT_RTFM_OUTPUTS)
    ap.add_argument("--lora-dir",     type=Path, default=DEFAULT_LORA_DIR)
    ap.add_argument("--annotations",  type=Path, default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--results-dir",  type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--skip-judge",   action="store_true")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    annotations = {}
    if args.annotations.exists():
        annotations = {e["video_id"]: e for e in json.loads(args.annotations.read_text()) if "video_id" in e}
        print(f"Loaded {len(annotations)} annotations")

    video_metas = {}
    for vdir in sorted(args.rtfm_outputs.iterdir()):
        if not vdir.is_dir(): continue
        mf = vdir / "metadata.json"
        if not mf.exists(): continue
        meta = json.loads(mf.read_text())
        video_metas[meta["video_id"]] = (meta, vdir)
    print(f"Found {len(video_metas)} videos to process")

    model, processor = load_model(args.lora_dir)

    openai_client = None
    if not args.skip_judge:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            from openai import OpenAI
            openai_client = OpenAI(api_key=api_key)
            print("GPT-4o judge enabled")
        else:
            print("OPENAI_API_KEY not set — skipping judge step")

    # Pass 1: zero-shot (LoRA disabled)
    with model.disable_adapter():
        zs = run_pass(model, processor, video_metas, annotations,
                      args.results_dir, "zeroshot_shortprompt",
                      f"{BASE_MODEL_ID}", openai_client)

    # Pass 2: LoRA fine-tuned
    ft = run_pass(model, processor, video_metas, annotations,
                  args.results_dir, "finetune_shortprompt",
                  f"{BASE_MODEL_ID}+LoRA", openai_client)

    # Side-by-side
    if zs and ft:
        import numpy as np
        print(f"\n{'='*60}\n  HEAD-TO-HEAD: short prompt, n=42 anomalous\n{'='*60}")
        print(f"  {'metric':<14s} {'zero-shot':>12s} {'LoRA':>12s} {'delta':>10s}")
        for m in ["correctness", "specificity", "completeness", "fluency"]:
            zv = np.mean([r["scores"][m] for r in zs if r["video_type"]=="anomalous" and r["scores"].get(m) is not None])
            fv = np.mean([r["scores"][m] for r in ft if r["video_type"]=="anomalous" and r["scores"].get(m) is not None])
            print(f"  {m:<14s} {zv:>12.2f} {fv:>12.2f} {fv-zv:>+10.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
