"""
local_qwen_caller.py
────────────────────
Reads pre-extracted frames + metadata from the OUTPUT_DIR written by
full_pipeline_colab.ipynb, posts each video to the Qwen-VL server running
on Colab (qwen_serve_colab.ipynb), then judges the explanations with GPT-4o
and compares them against the GPT-4o VLM baseline.

Usage
-----
1. Start qwen_serve_colab.ipynb on Colab and copy the ngrok URL.
2. Set COLAB_URL below (or pass via --url flag).
3. pip install openai requests tqdm
4. python local_qwen_caller.py
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests
from openai import OpenAI
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

# Paste the ngrok URL printed by qwen_serve_colab.ipynb Cell 4:
COLAB_URL = ''   # e.g. 'https://abcd-1234.ngrok-free.app'

# Where full_pipeline_colab.ipynb saved its outputs (local Google Drive mount
# or a synced local copy):
OUTPUT_DIR = Path('/Volumes/GoogleDrive/MyDrive/rtfm_pipeline_outputs')
# OUTPUT_DIR = Path('/Users/you/rtfm_pipeline_outputs')   # ← or local copy

ANNOTATIONS_PATH = Path('/Volumes/GoogleDrive/MyDrive/annotations.json')
# ANNOTATIONS_PATH = Path('/Users/you/annotations.json')

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

REQUEST_TIMEOUT = 120   # seconds per video (32B can be slow)
SLEEP_BETWEEN   = 0.5   # seconds between requests

# ── System prompt (identical to full_pipeline_colab.ipynb) ───────────────────

SYSTEM_PROMPT = """You are a surveillance video anomaly analyst. You will be shown a set of \
frames sampled from a surveillance video that has been flagged as anomalous \
by a weakly-supervised anomaly detection model (RTFM).

The frames are ordered temporally. Each frame comes from a specific temporal \
snippet of the video, and you are given the anomaly score for that snippet \
(0 = normal, 1 = highly anomalous).

The frames were specifically selected from the anomalous portions of the \
video — they represent the onset, peak, and resolution of the detected anomaly.

Your task: Based on ALL the frames and their anomaly scores together, provide \
a single concise explanation (2-3 sentences) of what anomalous activity is \
happening. Focus on:
- WHAT is happening (the specific anomalous activity)
- WHO/WHAT is involved (people, vehicles, objects — describe appearance)
- WHEN in the sequence it starts and ends
- WHY it is anomalous (how it deviates from normal pedestrian behaviour)

Respond with ONLY a JSON object in this exact format:
{"explanation": "..."}"""

JUDGE_PROMPT = """You are an impartial judge evaluating the quality of an AI-generated \
explanation of an anomalous event in a surveillance video.

Score the AI explanation on these 4 criteria (each 1-5):
- correctness: Does the AI identify the same anomaly as the human?
- specificity: Does the AI mention specific details (objects, people, actions)?
- completeness: Does the AI capture all aspects the human mentioned?
- fluency: Is the AI explanation well-written and clear?

Respond with ONLY a JSON object:
{"correctness": 1-5, "specificity": 1-5, "completeness": 1-5, "fluency": 1-5, "justification": "..."}"""

NORMAL_GT = 'There is nothing anomalous in this video. All pedestrians are walking normally.'

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_metadata(output_dir: Path) -> dict:
    metas = {}
    for meta_file in sorted(output_dir.glob('*/metadata.json')):
        with open(meta_file) as f:
            meta = json.load(f)
        vname = meta['video_id']
        valid = [fr for fr in meta['extracted_frames'] if fr.get('file')]
        if not valid:
            continue
        vid_dir = output_dir / vname
        for fr in valid:
            fr['_path'] = str(vid_dir / fr['file'])
        meta['extracted_frames'] = valid
        metas[vname] = meta
    return metas


def load_annotations(path: Path) -> dict:
    if not path.exists():
        print(f'WARNING: annotations.json not found at {path}')
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {e['video_id']: e for e in raw if 'video_id' in e}


def encode_frame(img_path: str) -> str:
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_request_body(meta: dict) -> dict:
    frames = meta['extracted_frames']
    segs   = meta['anomalous_segments']

    seg_str   = ', '.join(f"snippets {s['start']}-{s['end']}" for s in segs)
    score_str = ', '.join(
        f"snippet {f['snippet_idx']}={f['score']:.3f}" for f in frames
    )
    intro = (
        f"This video has {meta['n_segments']} temporal snippets (~16 frames each). "
        f"Anomalous segments: [{seg_str}]. "
        f"Video-level gate score: {meta['gate_score']:.3f}. "
        f"Below are {len(frames)} frames from the anomalous segments. "
        f"Per-snippet scores: [{score_str}]."
    )

    frame_items = []
    for fr in frames:
        img_path = fr.get('_path', '')
        if not Path(img_path).exists():
            print(f'  WARN: frame not found: {img_path}')
            continue
        frame_items.append({
            'label': (f"Frame from snippet {fr['snippet_idx']} "
                      f"(frame #{fr['frame_num']}, score: {fr['score']:.3f}):"),
            'b64': encode_frame(img_path),
        })

    return {
        'system_prompt':  SYSTEM_PROMPT,
        'intro_text':     intro,
        'frames':         frame_items,
        'max_new_tokens': 300,
    }


def call_colab(colab_url: str, body: dict, timeout: int) -> str:
    url = colab_url.rstrip('/') + '/explain'
    try:
        resp = requests.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get('explanation', '')
    except Exception as e:
        return f'ERROR: {e}'


def call_judge(client: OpenAI, human_expl: str, ai_expl: str) -> dict:
    user_msg = (
        f'HUMAN ground-truth explanation:\n"{human_expl}"\n\n'
        f'AI-generated explanation:\n"{ai_expl}"'
    )
    try:
        resp = client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': JUDGE_PROMPT},
                {'role': 'user',   'content': user_msg},
            ],
            max_tokens=300,
            response_format={'type': 'json_object'},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {k: None for k in ['correctness', 'specificity', 'completeness', 'fluency',
                                   'justification']} | {'justification': f'ERROR: {e}'}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',        default=COLAB_URL,
                        help='Colab ngrok URL, e.g. https://abcd.ngrok-free.app')
    parser.add_argument('--output_dir', default=str(OUTPUT_DIR))
    parser.add_argument('--skip_judge', action='store_true',
                        help='Skip GPT-4o judge step')
    args = parser.parse_args()

    colab_url = args.url.rstrip('/')
    if not colab_url:
        raise ValueError('Set COLAB_URL in the script or pass --url <ngrok_url>')

    out_dir  = Path(args.output_dir)
    ann_path = Path(args.output_dir).parent / 'annotations.json'
    if not ann_path.exists():
        ann_path = ANNOTATIONS_PATH

    # Health check
    try:
        r = requests.get(colab_url + '/health', timeout=10)
        print(f'Server health: {r.json()}')
    except Exception as e:
        raise RuntimeError(f'Cannot reach Colab server at {colab_url}: {e}')

    metas       = load_metadata(out_dir)
    annotations = load_annotations(ann_path)
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if not args.skip_judge else None

    print(f'\nLoaded {len(metas)} videos from {out_dir}')
    print(f'Annotations: {len(annotations)}')
    print(f'Running inference via {colab_url}\n')

    qwen_results = []
    judge_results = []

    for vname, meta in tqdm(sorted(metas.items()), desc='Qwen inference'):
        # ── Inference ────────────────────────────────────────────────────────
        body        = build_request_body(meta)
        explanation = call_colab(colab_url, body, REQUEST_TIMEOUT)
        tqdm.write(f'  {vname}  -> {explanation[:80]}...' if len(explanation) > 80
                   else f'  {vname}  -> {explanation}')

        expl_result = {
            'video_id':      vname,
            'model':         'Qwen3-VL-32B-Instruct',
            'gate_score':    meta['gate_score'],
            'n_frames_sent': len(body['frames']),
            'explanation':   explanation,
        }
        qwen_results.append(expl_result)

        vid_out = out_dir / vname
        vid_out.mkdir(parents=True, exist_ok=True)
        with open(vid_out / 'explanation_qwen.json', 'w') as f:
            json.dump(expl_result, f, indent=2)

        # ── Judge ─────────────────────────────────────────────────────────────
        if not args.skip_judge and openai_client and not explanation.startswith('ERROR'):
            clip_id = vname.split('_')[1] if '_' in vname else vname
            if len(clip_id) == 4 and vname in annotations:
                human_expl = annotations[vname]['explanation']
                video_type = 'anomalous'
            else:
                human_expl = NORMAL_GT
                video_type = 'normal_FP'

            scores = call_judge(openai_client, human_expl, explanation)
            tqdm.write(f'    judge: C={scores.get("correctness")} '
                       f'S={scores.get("specificity")} '
                       f'Co={scores.get("completeness")} '
                       f'F={scores.get("fluency")}')

            judge_result = {
                'video_id':          vname,
                'model':             'Qwen3-VL-32B-Instruct',
                'video_type':        video_type,
                'human_explanation': human_expl,
                'ai_explanation':    explanation,
                'gate_score':        meta['gate_score'],
                'scores':            scores,
            }
            judge_results.append(judge_result)

            with open(vid_out / 'judge_qwen.json', 'w') as f:
                json.dump(judge_result, f, indent=2)

            time.sleep(SLEEP_BETWEEN)

    # ── Save summaries ────────────────────────────────────────────────────────
    with open(out_dir / 'qwen_explanations_summary.json', 'w') as f:
        json.dump(qwen_results, f, indent=2)

    if judge_results:
        with open(out_dir / 'qwen_judge_summary.json', 'w') as f:
            json.dump(judge_results, f, indent=2)

    # ── Print summary stats ───────────────────────────────────────────────────
    import numpy as np
    METRICS = ['correctness', 'specificity', 'completeness', 'fluency']

    anom = [r for r in judge_results if r['video_type'] == 'anomalous']
    if anom:
        print(f'\n{"="*55}')
        print(f'  Qwen3-VL-32B  (anomalous n={len(anom)})')
        print(f'{"="*55}')
        for m in METRICS:
            vals = [r['scores'].get(m) for r in anom if r['scores'].get(m)]
            arr  = np.array(vals)
            print(f'  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}')

        # Load GPT-4o baseline for comparison
        gpt_judge_files = sorted(out_dir.glob('*/judge.json'))
        gpt_judge = []
        for jf in gpt_judge_files:
            with open(jf) as f:
                gpt_judge.append(json.load(f))
        gpt_anom = [r for r in gpt_judge if r.get('video_type') == 'anomalous']

        if gpt_anom:
            print(f'\n{"─"*55}')
            print(f'  GPT-4o baseline  (anomalous n={len(gpt_anom)})')
            print(f'{"─"*55}')
            for m in METRICS:
                vals = [r['scores'].get(m) for r in gpt_anom if r['scores'].get(m)]
                arr  = np.array(vals)
                print(f'  {m:<14s}  {arr.mean():.2f} ± {arr.std():.2f}')

    print(f'\nAll results saved to {out_dir}')
    print('  qwen_explanations_summary.json')
    if judge_results:
        print('  qwen_judge_summary.json')
        print('  <video>/explanation_qwen.json')
        print('  <video>/judge_qwen.json')


if __name__ == '__main__':
    main()
