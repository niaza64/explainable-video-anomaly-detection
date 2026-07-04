"""
build_judge_summary.py
──────────────────────
Converts full_results_without_qwen.json  →  judge_summary.json
(same schema the Colab pipeline writes)

Also prints the aggregate results table that was missing.

Usage:
    python pipeline/build_judge_summary.py
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR   = Path(__file__).parent / 'results'
INPUT_FILE    = RESULTS_DIR / 'full_results_without_qwen.json'
PIPELINE_JSON = Path(__file__).parent / 'rtfm_outputs' / 'pipeline_summary.json'

METRICS   = ['correctness', 'specificity', 'completeness', 'fluency']
VARIANTS  = ['with_heatmap', 'no_heatmap', 'video_only']

# ── Load ──────────────────────────────────────────────────────────────────────
with open(INPUT_FILE) as f:
    data = json.load(f)

per_video = data['table2_per_video']

# Gate scores from pipeline_summary.json (optional — fills gate_score field)
gate_scores = {}
if PIPELINE_JSON.exists():
    with open(PIPELINE_JSON) as f:
        ps = json.load(f)
    for v in ps.get('videos', []):
        gate_scores[v['video_id']] = v.get('gate_score', None)

# ── Build judge_summary.json entries for each variant ────────────────────────
for variant in VARIANTS:
    records = []
    for row in per_video:
        vname = row['video_id']
        clip_id = vname.split('_')[1] if '_' in vname else vname
        video_type = 'anomalous' if len(clip_id) == 4 else 'normal_FP'

        records.append({
            'video_id':          vname,
            'video_type':        video_type,
            'human_explanation': row.get('human_explanation', ''),
            'ai_explanation':    row.get(f'{variant}_explanation', ''),
            'gate_score':        gate_scores.get(vname),
            'scores': {
                'correctness':  row.get(f'{variant}_correctness'),
                'specificity':  row.get(f'{variant}_specificity'),
                'completeness': row.get(f'{variant}_completeness'),
                'fluency':      row.get(f'{variant}_fluency'),
                'justification': row.get(f'{variant}_justification', ''),
            }
        })

    out_path = RESULTS_DIR / f'judge_summary_{variant}.json'
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2)

# ── Also write the canonical judge_summary.json (with_heatmap variant) ───────
canonical = []
for row in per_video:
    vname   = row['video_id']
    clip_id = vname.split('_')[1] if '_' in vname else vname
    video_type = 'anomalous' if len(clip_id) == 4 else 'normal_FP'
    canonical.append({
        'video_id':          vname,
        'video_type':        video_type,
        'human_explanation': row.get('human_explanation', ''),
        'ai_explanation':    row.get('with_heatmap_explanation', ''),
        'gate_score':        gate_scores.get(vname),
        'scores': {
            'correctness':  row.get('with_heatmap_correctness'),
            'specificity':  row.get('with_heatmap_specificity'),
            'completeness': row.get('with_heatmap_completeness'),
            'fluency':      row.get('with_heatmap_fluency'),
            'justification': row.get('with_heatmap_justification', ''),
        }
    })

with open(RESULTS_DIR / 'judge_summary.json', 'w') as f:
    json.dump(canonical, f, indent=2)

# ── Print aggregate table ─────────────────────────────────────────────────────
print(f'\n{"="*70}')
print(f'  AGGREGATE JUDGE SCORES  —  ShanghaiTech anomalous videos')
print(f'  Source: {INPUT_FILE.name}   n={len([r for r in per_video if len(r["video_id"].split("_")[1])==4])} anomalous')
print(f'{"="*70}')
print(f'  {"Variant":<16s}  {"Corr":>6s}  {"Spec":>6s}  {"Comp":>6s}  {"Flu":>6s}  {"Overall":>8s}')
print(f'  {"─"*60}')

agg = data.get('table1_aggregate_scores', {})
for v in VARIANTS:
    scores = agg.get(v, {})
    vals   = [scores.get(m, {}).get('mean', float('nan')) for m in METRICS]
    overall = np.nanmean(vals)
    print(f'  {v:<16s}  {vals[0]:>6.2f}  {vals[1]:>6.2f}  {vals[2]:>6.2f}  {vals[3]:>6.2f}  {overall:>8.2f}')

print(f'{"="*70}')

# ── Print per-video table (anomalous only) ────────────────────────────────────
anomalous = [r for r in per_video if len(r['video_id'].split('_')[1]) == 4]
print(f'\n  Per-video — with_heatmap variant  (n={len(anomalous)})')
print(f'  {"Video":<14s}  {"C":>3s}  {"S":>3s}  {"Co":>3s}  {"F":>3s}  Human GT (truncated)')
print(f'  {"─"*75}')
for r in sorted(anomalous, key=lambda x: (x.get('with_heatmap_correctness') or 0), reverse=True):
    vname = r['video_id']
    c  = r.get('with_heatmap_correctness', '?')
    s  = r.get('with_heatmap_specificity', '?')
    co = r.get('with_heatmap_completeness', '?')
    fl = r.get('with_heatmap_fluency', '?')
    gt = r.get('human_explanation', '')[:45]
    print(f'  {vname:<14s}  {str(c):>3s}  {str(s):>3s}  {str(co):>3s}  {str(fl):>3s}  {gt}')

print(f'\n  Saved:')
for v in VARIANTS:
    print(f'    pipeline/results/judge_summary_{v}.json')
print(f'    pipeline/results/judge_summary.json  (canonical = with_heatmap)')
