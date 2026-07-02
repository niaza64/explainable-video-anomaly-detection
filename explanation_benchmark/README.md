# ShanghaiTech Explainable-VAD: Human References and Evaluation Protocol

Human-written reference explanations and a GPT-4o-as-judge protocol for
evaluating **explanation quality** in video anomaly detection (VAD) on the
weakly-supervised **ShanghaiTech Campus** benchmark.

Existing explainable-VAD benchmarks (e.g. UCF-Crime, XD-Violence) target
frame-level *detection* AUC. This resource instead lets you measure how well a
model *explains* an anomaly in natural language on ShanghaiTech, where only
weak video-level labels exist.

## What is included

| File | Contents |
|------|----------|
| `references.json` | 44 human one-sentence anomaly explanations (one per anomalous test video), plus the fixed reference used for the localiser's false-positive normal videos. |
| `JUDGE_PROMPT.txt` | The exact GPT-4o judge prompt (four 1-5 criteria). |
| `evaluate.py` | Scores a set of model explanations against the references and reports per-split means, the overall score, and a paired-bootstrap helper. |

## The protocol

Each generated explanation is scored by **GPT-4o as an impartial judge** on
four criteria, each on a 1-5 scale:

- **Correctness** - does it identify the same anomaly as the human?
- **Specificity** - does it mention concrete detail (objects, people, actions)?
- **Completeness** - does it cover all aspects the human mentioned?
- **Fluency** - is it well written?

**Overall** is the mean of the four criteria. For significance we use a
**paired bootstrap** (10,000 resamples, seed 42, 95% percentile CI on the mean
delta over common videos); a difference is significant iff its CI excludes zero.

Videos are split into **anomalous** (scored against the human reference) and
**normal-FP** (normal videos the localiser mis-flagged, scored against the
fixed "no anomaly" reference).

## Usage

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
python evaluate.py --pred predictions.json --out scores.json
```

`predictions.json` maps `video_id -> explanation`:

```json
{
  "01_0015": "A person is skateboarding across a pedestrian plaza ...",
  "01_074":  "The scene shows people walking normally; no anomaly is visible."
}
```

## Reproducibility notes

- Judge model: `gpt-4o`, `temperature=0`, `response_format=json_object`.
- GPT-4o-as-judge is **not perfectly deterministic run-to-run**, so compare all
  systems **within a single judging pass** rather than against numbers judged at
  a different time.

## Citation

> _Paper under review; citation to be added._

## License

Released under the MIT License (see `LICENSE`). The reference sentences are
derived from human annotation of the public ShanghaiTech Campus dataset; please
also cite the original dataset.
