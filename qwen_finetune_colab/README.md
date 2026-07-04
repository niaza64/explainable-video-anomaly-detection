# Qwen SFT data

- **`SFT_FRAME_POOLING.md`** — definitions and the five strategies.
- **`pool.py`** — snippet bounds, human window **I**, strategy functions.
- **`build_sft_data.py`** — RTFM scoring, frame export, `manifest.jsonl` + `summary.json`.

**System prompt** is copied verbatim from `pipeline/local_qwen_caller.py` (`SYSTEM_PROMPT`, lines 48–68); edit both places if you change the wording. The human turn is a short score list + instruction.

From the repo root (needs train `.mp4`s, `*_i3d.npy` features, `rtfm/ckpt/rtfm_best.pkl`):

```bash
python qwen_finetune_colab/build_sft_data.py --out-dir qwen_finetune_colab/out
```

Optional **train/val manifests by `video_id`** (same JSONL schema as `manifest.jsonl`):

```bash
python qwen_finetune_colab/build_sft_data.py --out-dir qwen_finetune_colab/out --val-frac 0.15 --split-seed 42
```

Writes `manifest_train.jsonl`, `manifest_eval.jsonl`, and `manifest_split.json` next to `manifest.jsonl`. Or run `colab_qwen3vl_32b_lora.ipynb`, which writes the same files when it builds the LLaMA-Factory JSON.

By default this runs **all five** strategies (`every_snippet_mid`, `every_snippet_first`, `every_snippet_mid_frame_band`, `human_span_smart`, `top3_snippets_mid_frame_band`). Pass e.g. `--strategies every_snippet_mid` only when you want a subset.

**Pooling-only refresh** (JPEGs already built; e.g. annotations or flags changed):

```bash
python qwen_finetune_colab/build_sft_data.py --out-dir qwen_finetune_colab/out --no-images
```

Still needs MP4s (frame count), features, and RTFM for scores; only skips writing image files.

Other flags: `--delta`, `--snippet-budget`, `--min-gap`, paths for annotations, videos, features, checkpoint.

**Default data root:** `data/SHANGHAI/SHANGHAI_TRAIN/videos_anomalous_train_with_human_annotations/` — same folder for `Anomalous_train_annotations.json` and `*.mp4`. Use `--videos-dir` / `--annotations` if yours is elsewhere.

Each row’s second turn uses **`from`: `assistant`** with the **human-written** reference explanation (SFT target). Older ShareGPT tooling sometimes expects `gpt` for that role; rename in a one-liner if your trainer requires it.

Validate on held-out **`video_id`s**, not random manifest rows (several strategies per video).
