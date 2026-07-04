#!/usr/bin/env python3
"""
Qwen3-VL-32B ZERO-SHOT (vanilla, no LoRA) inference on TCSC cluster.

Identical pipeline to run_qwen_inference.py but loads the BASE checkpoint
only — no PEFT adapter. Reads rtfm_outputs/ metadata + frames, generates
explanations, judges with GPT-4o, saves to the results dir.

Because it iterates over EVERY video dir in --rtfm-outputs, pointing it at
the same rtfm_outputs/ used by the LoRA and RAG runs reproduces the vanilla
baseline on the identical 80-video test set (42 anomalous + 38 normal-FP),
fixing the earlier 46-video mismatch.
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

# ── Paths (set via args or env) ───────────────────────────────────────────────
BASE_DIR               = Path("/scratch/svc_td_ppml/qrx527/niaz_research")
DEFAULT_RTFM_OUTPUTS   = Path(os.environ.get("RTFM_OUTPUTS",   BASE_DIR / "rtfm_outputs"))
DEFAULT_ANNOTATIONS    = Path(os.environ.get("ANNOTATIONS",    BASE_DIR / "annotations.json"))
DEFAULT_RESULTS_DIR    = Path(os.environ.get("RESULTS_DIR",    BASE_DIR / "tcsc_zeroshot_results"))
BASE_MODEL_ID          = "Qwen/Qwen3-VL-32B-Instruct"
MAX_NEW_TOKENS         = 300
RUN_TAG                = "zeroshot"

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


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load the BASE Qwen3-VL-32B checkpoint only — no LoRA adapter."""
    from transformers import AutoProcessor

    print(f"Loading base model (zero-shot, no LoRA): {BASE_MODEL_ID}")
    try:
        from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
        print("Using Qwen3VLForConditionalGeneration")
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
        print("Falling back to Qwen2VLForConditionalGeneration")

    model = QwenVLModel.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. GPU memory: {allocated:.1f} GB")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

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
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)[:2]
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=body.get("max_new_tokens", MAX_NEW_TOKENS),
                do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        text = raw
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        try:
            parsed = json.loads(text.strip())
            return parsed.get("explanation", raw)
        except json.JSONDecodeError:
            return raw

    except Exception as e:
        return f"ERROR: {e}"
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()


def build_request(meta: dict, video_dir: Path) -> dict:
    frames = meta.get("extracted_frames", [])
    segs   = meta.get("anomalous_segments", [])

    def seg_label(s):
        a = s.get("start_snippet", s.get("start", "?"))
        b = s.get("end_snippet",   s.get("end",   "?"))
        return f"snippets {a}-{b}"

    seg_str   = ", ".join(seg_label(s) for s in segs) if segs else "unknown"
    score_str = ", ".join(f"snippet {f['snippet_idx']}={f['score']:.3f}" for f in frames)
    intro = (
        f"This video has {meta['n_segments']} temporal snippets (~16 frames each). "
        f"RTFM flagged these segments as anomalous: [{seg_str}]. "
        f"Video-level anomaly gate score: {meta['gate_score']:.3f} (threshold=0.2). "
        f"Below are {len(frames)} frames from the anomalous segments. "
        f"Per-snippet anomaly scores: [{score_str}]."
    )
    frame_items = []
    for fr in frames:
        img_path = video_dir / fr["file"]
        if not img_path.exists():
            print(f"  WARN: frame not found: {img_path}")
            continue
        frame_items.append({
            "label": (
                f"Frame from snippet {fr['snippet_idx']} "
                f"(frame #{fr['frame_num']}, anomaly score: {fr['score']:.3f}):"
            ),
            "b64": base64.b64encode(img_path.read_bytes()).decode("utf-8"),
        })
    return {
        "system_prompt":  SYSTEM_PROMPT,
        "intro_text":     intro,
        "frames":         frame_items,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


# ── Judge ─────────────────────────────────────────────────────────────────────

def call_judge(client, human_expl: str, ai_expl: str) -> dict:
    user_msg = (
        f'HUMAN ground-truth explanation:\n"{human_expl}"\n\n'
        f'AI-generated explanation:\n"{ai_expl}"'
    )
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtfm-outputs",  type=Path, default=DEFAULT_RTFM_OUTPUTS)
    parser.add_argument("--annotations",   type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--results-dir",   type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--skip-judge",    action="store_true", help="Skip GPT-4o judge step")
    parser.add_argument("--video",         type=str,  default=None, help="Run on single video ID")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {args.results_dir}")

    # Load annotations
    annotations = {}
    if args.annotations.exists():
        annotations = {
            e["video_id"]: e
            for e in json.loads(args.annotations.read_text())
            if "video_id" in e
        }
        print(f"Loaded {len(annotations)} annotations")

    # Collect video metadata
    video_dirs = sorted(args.rtfm_outputs.iterdir()) if not args.video else [args.rtfm_outputs / args.video]
    video_metas = {}
    for vdir in video_dirs:
        if not vdir.is_dir():
            continue
        meta_file = vdir / "metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        video_metas[meta["video_id"]] = (meta, vdir)
    print(f"Found {len(video_metas)} videos to process")

    # Load model (base only, no LoRA)
    model, processor = load_model()

    # Optional judge client
    openai_client = None
    if not args.skip_judge:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            from openai import OpenAI
            openai_client = OpenAI(api_key=api_key)
            print("GPT-4o judge enabled")
        else:
            print("OPENAI_API_KEY not set — skipping judge step")

    # Run inference
    all_explanations = []
    all_judge_results = []

    for vname, (meta, vdir) in tqdm(sorted(video_metas.items()), desc="Qwen zero-shot inference"):
        out_dir = args.results_dir / vname
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build request and run inference
        body        = build_request(meta, vdir)
        explanation = run_inference(model, processor, body)
        tqdm.write(f"  {vname}: {explanation[:80]}{'...' if len(explanation) > 80 else ''}")

        expl_result = {
            "video_id":      vname,
            "model":         BASE_MODEL_ID,
            "run_tag":       RUN_TAG,
            "gate_score":    meta["gate_score"],
            "n_segments":    meta["n_segments"],
            "anomalous_segments": meta.get("anomalous_segments", []),
            "n_frames_sent": len(body["frames"]),
            "explanation":   explanation,
        }
        all_explanations.append(expl_result)
        (out_dir / f"explanation_qwen_{RUN_TAG}.json").write_text(
            json.dumps(expl_result, indent=2)
        )

        # Judge
        if openai_client and not explanation.startswith("ERROR"):
            clip_id = vname.split("_")[1] if "_" in vname else vname
            is_anomalous = len(clip_id) == 4 and vname in annotations
            human_expl  = annotations[vname]["explanation"] if is_anomalous else NORMAL_FP_GT
            video_type  = "anomalous" if is_anomalous else "normal_FP"

            scores = call_judge(openai_client, human_expl, explanation)
            tqdm.write(
                f"    judge C={scores.get('correctness')} "
                f"S={scores.get('specificity')} "
                f"Co={scores.get('completeness')} "
                f"F={scores.get('fluency')}"
            )
            judge_result = {
                "video_id":          vname,
                "model":             BASE_MODEL_ID,
                "run_tag":           RUN_TAG,
                "video_type":        video_type,
                "human_explanation": human_expl,
                "ai_explanation":    explanation,
                "gate_score":        meta["gate_score"],
                "scores":            scores,
            }
            all_judge_results.append(judge_result)
            (out_dir / f"judge_qwen_{RUN_TAG}.json").write_text(
                json.dumps(judge_result, indent=2)
            )
            time.sleep(0.5)

    # Save summaries
    summary_path = args.results_dir / f"qwen_{RUN_TAG}_explanations_summary.json"
    summary_path.write_text(json.dumps(all_explanations, indent=2))
    print(f"\nExplanations saved: {summary_path}")

    if all_judge_results:
        judge_summary = args.results_dir / f"qwen_{RUN_TAG}_judge_summary.json"
        judge_summary.write_text(json.dumps(all_judge_results, indent=2))
        print(f"Judge results saved: {judge_summary}")

        # Print metrics, split by video_type
        import numpy as np
        for vtype in ["anomalous", "normal_FP"]:
            subset = [r for r in all_judge_results if r["video_type"] == vtype]
            if not subset:
                continue
            print(f"\n{'='*50}")
            print(f"  Qwen3-VL-32B zero-shot  ({vtype}, n={len(subset)})")
            print(f"{'='*50}")
            for m in ["correctness", "specificity", "completeness", "fluency"]:
                vals = [r["scores"].get(m) for r in subset if r["scores"].get(m) is not None]
                arr  = np.array(vals)
                print(f"  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
