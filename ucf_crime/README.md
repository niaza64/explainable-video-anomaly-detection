# UCF-Crime extension

Code and released annotations for the UCF-Crime study in *VAD-Explain*.
The ShanghaiTech pipeline lives in the repository root; this folder contains only
what is specific to UCF-Crime.

## Released resource

`annotations/ucf_gold_references.json` — reference explanations for all 290
UCF-Crime test videos (140 anomalous + 150 normal). Built from the human-written
UCA annotations (Yuan et al., CVPR 2024) by selecting sentences overlapping the
official anomaly window. **No LLM is involved**, so the references stay independent
of the GPT-4o judge. `..._v2.json` ranks candidate sentences by class-cue presence
and concentration instead of raw overlap.

## Pipeline

```
1. extract_i3d_ucf.py         mp4 -> I3D 10-crop features (T,10,2048)
2. validate_all.py            verify shapes and T == ceil(frames/16)
3. precompute_seg32.py        pre-pool train features to RTFM's 32 segments
4. (train RTFM)               ../rtfm/train_rtfm.py --dataset ucf --gt list/gt-ucf.npy
5. build_ucf_gated_outputs.py RTFM scores -> selected frames + metadata.json
6. build_gold_explanations_from_uca.py   UCA -> reference explanations
7. build_train_manifest_from_uca.py      LoRA / RAG training pool
8. run_qwen_ucf_inference.py  zero-shot | RAG | LoRA, + GPT-4o judge
```

Analysis: `gate_eval_ucf.py` (video-level gate), `localisation_sweep.py`
(frame-selection strategies), `shanghaitech_sampler_check.py` (cross-dataset
sampler comparison), `rejudge_against_new_gold.py` (re-score stored explanations
against different references).

## Notes that matter for reproduction

- **Snippet count is `ceil(n_frames/16)`**, not floor. Verified against the
  official features (155 frames -> 10 snippets, 243 -> 16, 1409 -> 89).
  The ShanghaiTech script uses floor.
- Features are `(T, 10, 2048)` float32 for **both** splits; `rtfm/dataset.py`
  transposes internally.
- Training reads the **pre-pooled** `(10, 32, 2048)` features. Pooling is
  deterministic and verified bit-identical to the on-the-fly path; without it a
  run needs roughly 38 TB of reads.
- The published RTFM UCF features are incomplete (109 of 290 test files), so all
  features here are extracted locally with the same extractor used for
  ShanghaiTech. The two feature sets are **not** interchangeable (correlation 0.01).
- `run_qwen_ucf_inference.py --prompt-style` selects the prompting strategy;
  `taxonomy_questions` is the grounded variant used in the paper's prompt ablation.

## Not included

Extracted frames, I3D features, RTFM checkpoints and LoRA adapters are omitted
for size. Videos come from the official UCF-Crime distribution.

## Citation

UCA references: Yuan et al., *Towards Surveillance Video-and-Language
Understanding*, CVPR 2024.
