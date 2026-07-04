# UCF-Crime Evaluation — Paused (future work)

This folder was a scaffold for a UCF-Crime second-dataset evaluation.
**Status: paused.** The work has been deprioritized for the ACCV 2026
submission in favour of deeper single-dataset analysis on ShanghaiTech.
Nothing in this folder affects the ShanghaiTech / v1 / v2 / v3 / RAG
results. It is kept here as a starting point for a future extension.

## Why it stalled

| Issue | Detail |
|---|---|
| Partial Google Drive download | The RTFM-authors' UCF I3D test-feature zip contained only 109 / 290 expected videos, covering only 3 of 13 anomaly classes (Normal_Videos, RoadAccidents, Robbery). Other classes (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Shooting, Shoplifting, Stealing, Vandalism) were missing. |
| Train features rate-limited | The train-features Google Drive link returned ``access is particularly large or is shared with many people'' and is temporarily not downloadable. May unlock in 24 h. |
| Video files not piecemeal | UCF-Crime ships as one 120 GB Dropbox folder with no per-file API. HuggingFace mirrors are either (a) low-resolution 64x64 PNGs unusable for VLM input or (b) a 50-video single-class subset. There is no clean way to acquire only the 109 videos we have features for. |

## What we built that is still useful

- `STATUS.md` (this file)
- `data/ucf-i3d-test.list` --- the 290-video test-set list from RTFM
- `data/ucf-i3d.list` --- the 1610-video full list
- `data/gt-ucf.npy` --- 1.1M frame-level binary anomaly labels
- `data/make_gt_ucf.py` --- script that produced gt-ucf.npy
- `scripts/inspect_ucf_features.py` --- inspector for any features we later acquire

## What we acquired but did not use

- 109 UCF test I3D feature `.npy` files (sit on cluster at `/scratch/svc_td_ppml/qrx527/niaz_research_ucf_crime_separate_workspace/data/i3d_test/UCF_test_feature/`)
- Each feature: shape `(N_snippets, 10, 2048)` --- ResNet-50 I3D, 10-crop augmented

## How to resume

1. Acquire the full UCF-Crime video set (120 GB; manual Dropbox download or
   submit a CRCV academic access request).
2. Retry the train-features Google Drive link (`1i2P9Nn62i0cVil_WS24HKzbzmyUxA9vX`)
   once the rate-limit clears.
3. Either train RTFM on the train features (4-8h on H200) or fetch the
   RTFM-authors' pretrained UCF checkpoint.
4. Extract test frames from videos using ffmpeg + the RTFM-predicted
   anomaly windows (mirror the ShanghaiTech `rtfm_outputs/` layout).
5. Generate per-video GPT-4o ``gold'' explanations from the class label +
   sampled frames (option B from the original status doc).
6. Run the existing RAG inference script (`qwen_rag_retrieval_augmented_in_context_with_clip_embeddings_top3/cluster/run_qwen_rag_inference.py`) pointed at the UCF data; build a UCF-only train pool of ~20 videos per class for the in-context exemplars.
7. Judge with GPT-4o on the UCF subset, add a second row to
   `results_comparison.tex`.

Estimated effort if all data magically appears: **~3-4 days** (1-2 days
data prep, 1 day eval, 1 day write-up).
