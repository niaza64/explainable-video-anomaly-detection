# VAD-Explain: Vision-Language Models for Explainable Video Surveillance

A two-stage, **weakly-supervised** pipeline for explainable video anomaly detection
(xVAD) on the ShanghaiTech Campus benchmark. A weakly-supervised detector (**RTFM**)
first flags anomalous videos and localises the suspicious temporal region; a frozen
vision-language model (**Qwen3-VL-32B**) then generates a natural-language explanation
of the anomaly.

The central question this repo answers: **under weak supervision with few labelled
anomalies, what is the best way to adapt a VLM for explanation — fine-tuning or
retrieval?** We compare three strategies and find that **training-free
retrieval-augmented prompting (RAG) beats both zero-shot prompting and LoRA
fine-tuning**, while requiring no additional training.

```
I3D features → RTFM anomaly scores → temporal localisation → frame sampling → VLM explanation
```

---

## The three explanation approaches

All three share the same RTFM front-end, the same sampled frames, and the same
evaluation protocol — so results isolate the adaptation strategy alone.

| Approach | Idea | Where the code lives | Main entry points |
|---|---|---|---|
| **Zero-shot** | Prompt a frozen Qwen3-VL with the sampled frames + RTFM scores | [`pipeline/`](pipeline/), [`cluster/`](cluster/) | `cluster/run_qwen_zeroshot_inference.py` (`slurm_zeroshot.sh`); `pipeline/generate_explanations_rtfm.py` |
| **LoRA fine-tuning** | Adapt Qwen3-VL with LoRA on the 63 training anomalies (v1 = short labels, v2 = GPT-4o-enriched labels, v3 = + normal-video negatives) | [`qwen_finetune_colab/`](qwen_finetune_colab/) (v1), [`qwen_finetune_v2/`](qwen_finetune_v2/) (v2), [`qwen_finetune_v3_enriched_labels_single_strategy_with_normal_video_negatives/`](qwen_finetune_v3_enriched_labels_single_strategy_with_normal_video_negatives/) (v3) | `qwen_finetune_colab/colab_qwen3vl_32b_lora_fine_tune.ipynb`; `build_sft_data*.py` |
| **RAG** (proposed) | Retrieve visually-similar training videos via CLIP and add them as in-context exemplars to a frozen Qwen3-VL | [`qwen_rag_retrieval_augmented_in_context_with_clip_embeddings_top3/`](qwen_rag_retrieval_augmented_in_context_with_clip_embeddings_top3/) | `cluster/run_qwen_rag_inference.py` (`slurm_rag.sh`) |

> These folder names are referenced by path in several scripts (e.g.
> `REPO / "qwen_finetune_colab" / "out"`), so they are intentionally kept verbose
> rather than renamed. Use this table as the map.

GPT-4o is used **only as the automatic judge** (see Evaluation), not as a compared
explainer.

---

## Repository structure

```
rtfm/                    Stage-1 weakly-supervised localiser (RTFM). I3D feature
                         extraction, training, 10-crop test, segment scoring.
                         (Adapted fork of tianyu0207/RTFM — see Attribution.)

pipeline/                Zero-shot explanation generation + local orchestration
                         (detect → localise → sample → explain → judge).
cluster/                 SLURM launch scripts + inference drivers for the cluster
                         (zero-shot, short-prompt variants).

qwen_finetune_colab/     LoRA v1 (original short human labels).
qwen_finetune_v2/        LoRA v2 (GPT-4o-enriched labels) + label-enrichment tools.
qwen_finetune_v3_.../    LoRA v3 (enriched labels + normal-video negatives).

qwen_rag_..._top3/       Proposed RAG method: CLIP retrieval + in-context prompting,
                         plus K-ablation, encoder-ablation, and the bootstrap
                         significance / judge-correlation analysis scripts.

explanation_benchmark/   Released human reference explanations (references.json),
                         the GPT-4o judge prompt, and the evaluation script.

scripts/                 Shared data-prep helpers (train-anomaly annotations, etc.).

experiments/             Archived, exploratory runs (earlier AED-MAE / Avenue /
aed-mae/                 SAM heatmap work). NOT part of the reported pipeline;
rtfm_train_viz/          kept for transparency. May contain machine-specific paths.
```

---

## Setup

```bash
git clone https://github.com/niaza64/explainable-video-anomaly-detection.git
cd explainable-video-anomaly-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Secrets are read from the environment. Copy the template and fill in your own keys:

```bash
cp .env.example .env
# OPENAI_API_KEY — used by the GPT-4o judge
# HF_TOKEN       — used to download Qwen3-VL / gated weights
```

**No secret keys are committed to this repository.** Every script reads credentials
from `OPENAI_API_KEY` / `HF_TOKEN` at runtime.

---

## Data & weights (download separately)

These are excluded from git (size / licensing) and must be obtained before running:

| Artifact | Source | Expected location |
|---|---|---|
| ShanghaiTech Campus videos/frames | [ShanghaiTech dataset](https://svip-lab.github.io/dataset/campus_dataset.html) | `data/SHANGHAI/` |
| Pre-extracted I3D features (10-crop) | RTFM release ([tianyu0207/RTFM](https://github.com/tianyu0207/RTFM)) | as referenced in `rtfm/list/*.list` |
| I3D backbone (feature extraction) | [pytorch-resnet3d](https://github.com/facebookresearch/VMZ) / RTFM instructions | `rtfm/pytorch-resnet3d/` |
| RTFM checkpoint | Train via `rtfm/train_rtfm.py` or use released weights | `rtfm/rtfm_checkpoints/` |
| Qwen3-VL-32B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) (auto-download) | HF cache |

---

## Reproducing the results

**1. Localise (RTFM).** Extract/obtain I3D features, then score videos and produce
per-video segment scores:

```bash
python rtfm/test_10crop.py          # video-level + snippet scores
# gate at tau_g = 0.2; localise segments at tau_s = 0.3; sample up to 8 frames
```

**2a. Zero-shot explanation.**

```bash
python cluster/run_qwen_zeroshot_inference.py     # or sbatch cluster/slurm_zeroshot.sh
```

**2b. LoRA fine-tuning.** Build the SFT data, then fine-tune (rank 16, α=32, lr 2e-5,
2 epochs) — see `qwen_finetune_colab/` (v1), `qwen_finetune_v2/` (v2):

```bash
python qwen_finetune_v2/build_sft_data_v2.py
# then run the LoRA notebook / cluster job in the corresponding folder
```

**2c. RAG (proposed).**

```bash
python qwen_rag_..._top3/cluster/run_qwen_rag_inference.py   # or sbatch slurm_rag.sh
```

**3. Evaluate.** Score generated explanations with the GPT-4o judge against the human
references, and run significance tests:

```bash
python explanation_benchmark/evaluate.py
python qwen_rag_..._top3/scripts/bootstrap_significance.py    # paired bootstrap, 95% CI
```

---

## Results (ShanghaiTech, 40 correctly-localised anomalous test videos)

GPT-4o judge, 1–5 Likert; **Overall** is the mean of the four criteria.

| Method | Corr. | Spec. | Compl. | Flu. | Overall |
|---|---|---|---|---|---|
| Qwen3-VL-32B zero-shot | 2.80 | 3.55 | 2.75 | 4.50 | 3.40 |
| LoRA v1 (short labels) | 2.65 | 3.40 | 2.40 | 4.53 | 3.24 |
| LoRA v2 (enriched labels) | 2.92 | 3.33 | 2.73 | 4.50 | 3.37 |
| **RAG (CLIP top-3, vanilla Qwen)** | **3.62** | **3.98** | **3.52** | **4.62** | **3.94** |

RAG significantly beats zero-shot (Δ = +0.54, 95% CI [+0.34, +0.76]) and both LoRA
variants; LoRA is statistically indistinguishable from zero-shot.

---

## The explanation benchmark (a contribution of this work)

ShanghaiTech ships frame-level masks but **no textual anomaly descriptions**. We
manually annotate a one-sentence human reference explanation for each anomalous test
video and release them, together with the GPT-4o judge prompt and evaluation script,
in [`explanation_benchmark/`](explanation_benchmark/) to support reproducible xVAD
evaluation.

---

## Attribution

- **RTFM** — `rtfm/` is an adapted fork of
  [tianyu0207/RTFM](https://github.com/tianyu0207/RTFM) (Tian et al., ICCV 2021).
  Please cite the original work and respect its license.
- **Qwen3-VL** — [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct).
- **CLIP / OpenCLIP** — retrieval encoder (`open_clip` ViT-B/32).

## License

> **TODO:** add a top-level `LICENSE`. Note that `rtfm/` derives from third-party code
> under its own license; keep that license intact and choose a compatible license for
> the rest of the repository.
