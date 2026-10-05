#!/usr/bin/env python3
"""Re-judge ALREADY-GENERATED explanations against a different gold set.

Isolates the effect of reference quality: the model outputs are byte-identical,
only the reference changes. Same deterministic judge settings as the Holmes
baseline comparison (gpt-4o, temperature=0, seed=42).
"""
import json, os, time, argparse
import numpy as np
from openai import OpenAI

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
M = ["correctness", "specificity", "completeness", "fluency"]

ap = argparse.ArgumentParser()
ap.add_argument("--judge-summary", required=True)
ap.add_argument("--new-gold", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

oc = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
gold = {r["video_id"]: r for r in json.load(open(a.new_gold))}
old = json.load(open(a.judge_summary))

out = []
for i, r in enumerate(old, 1):
    if r["video_type"] != "anomalous":
        out.append(r); continue
    g = gold.get(r["video_id"])
    if not g or not g.get("explanation"):
        continue
    msg = (f'HUMAN ground-truth explanation:\n"{g["explanation"]}"\n\n'
           f'AI-generated explanation:\n"{r["ai_explanation"]}"')
    try:
        resp = oc.chat.completions.create(model="gpt-4o",
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": msg}],
            max_tokens=300, temperature=0, seed=42,
            response_format={"type": "json_object"})
        sc = json.loads(resp.choices[0].message.content)
    except Exception as e:
        sc = {k: None for k in M} | {"justification": f"ERROR: {e}"}
    out.append(dict(r, scores=sc, human_explanation=g["explanation"],
                    gold_source=g.get("source"), gold_version="v2"))
    if i % 25 == 0: print(f"  {i}/{len(old)}", flush=True)
    time.sleep(0.25)

json.dump(out, open(a.out, "w"), indent=1)
for vt in ["anomalous", "normal_FP"]:
    rs = [r for r in out if r["video_type"] == vt]
    if not rs: continue
    print(f"\n===== {os.path.basename(a.out)}  {vt}  n={len(rs)} =====")
    per = []
    for k in M:
        v = [r["scores"].get(k) for r in rs if r["scores"].get(k) is not None]
        if v: print(f"  {k:<14s} {np.mean(v):.2f} ± {np.std(v):.2f}"); per.append(np.mean(v))
    if per: print(f"  {'OVERALL':<14s} {np.mean(per):.2f}")
