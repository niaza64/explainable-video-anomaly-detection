#!/usr/bin/env python3
"""
Explanation-quality evaluation for video anomaly detection on ShanghaiTech.

Given a file of model-generated explanations, score each against the human
reference with GPT-4o as an automatic judge on four 1-5 criteria
(correctness, specificity, completeness, fluency), and report per-split means
and the overall score. A paired-bootstrap helper is included for comparing
two systems.

Usage:
    export OPENAI_API_KEY="sk-..."
    python evaluate.py --pred predictions.json [--out scores.json]

predictions.json format:
    {"01_0015": "generated explanation ...", "01_074": "...", ...}
(video_id -> explanation string). Any video_id present in references.json is
scored; anomalous videos use their human reference, the localiser's
false-positive normals use the fixed "no anomaly" reference.
"""
import argparse, json, os, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGE_PROMPT = (HERE / "JUDGE_PROMPT.txt").read_text().strip()
CRITERIA = ["correctness", "specificity", "completeness", "fluency"]


def load_references(path):
    d = json.loads(Path(path).read_text())
    fp_ref = d["normal_fp_reference"]
    refs = {r["video_id"]: (r["reference"], "anomalous") for r in d["anomalous"]}
    return refs, fp_ref


def judge(client, human, ai):
    msg = f'HUMAN ground-truth explanation:\n"{human}"\n\nAI-generated explanation:\n"{ai}"'
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": msg}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=300,
    )
    return json.loads(resp.choices[0].message.content)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarise(rows):
    for split in ["anomalous", "normal_FP"]:
        sub = [r for r in rows if r["video_type"] == split]
        if not sub:
            continue
        print(f"\n[{split}]  n={len(sub)}")
        per = {}
        for m in CRITERIA:
            per[m] = mean([r["scores"][m] for r in sub if r["scores"].get(m) is not None])
            print(f"  {m:<12s} {per[m]:.2f}")
        print(f"  {'overall':<12s} {mean(list(per.values())):.2f}")


def paired_bootstrap(scores_a, scores_b, key="overall", n=10000, seed=42):
    """95% CI on the mean overall-score delta (a - b) over common video_ids."""
    random.seed(seed)
    ids = sorted(set(scores_a) & set(scores_b))
    d = [scores_a[i] - scores_b[i] for i in ids]
    if not d:
        return None
    res = sorted(sum(d[random.randrange(len(d))] for _ in d) / len(d) for _ in range(n))
    return {"delta": mean(d), "ci": [res[int(0.025 * n)], res[int(0.975 * n)]], "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.json (video_id -> explanation)")
    ap.add_argument("--references", default=str(HERE / "references.json"))
    ap.add_argument("--out", default=None, help="write per-video scores here")
    args = ap.parse_args()

    refs, fp_ref = load_references(args.references)
    preds = json.loads(Path(args.pred).read_text())

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    rows = []
    for vid, expl in preds.items():
        if vid in refs:
            human, vtype = refs[vid]
        else:
            # video_ids with a 3-digit clip index are normal false positives
            human, vtype = fp_ref, "normal_FP"
        scores = judge(client, human, expl)
        rows.append({"video_id": vid, "video_type": vtype,
                     "human_reference": human, "ai_explanation": expl, "scores": scores})
        print(f"{vid} [{vtype}]  C={scores.get('correctness')} "
              f"S={scores.get('specificity')} Co={scores.get('completeness')} F={scores.get('fluency')}")

    summarise(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        print(f"\nper-video scores written to {args.out}")


if __name__ == "__main__":
    main()
