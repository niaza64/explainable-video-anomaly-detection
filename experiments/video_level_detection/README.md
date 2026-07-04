# Video-Level Detection

This module computes a **video-level anomaly verdict** by:

1. Running selected videos through AED-MAE to get per-frame anomaly scores.
2. Building a video score as `mean(top-k frame scores)`.
3. Predicting anomaly if `video_score >= threshold`.
4. Searching the best `k` and threshold on a balanced subset.

## Run

From project root:

```bash
python video_level_detection/run_video_level_detection.py
```

Default behavior:
- 10 anomaly + 10 normal videos
- `k` candidates: `1,3,5,8,10,15,20,30,50`
- threshold grid steps: `200`

## Useful options

```bash
python video_level_detection/run_video_level_detection.py \
  --num-anomaly 10 \
  --num-normal 10 \
  --seed 42 \
  --k-values 1,3,5,8,10,15,20 \
  --threshold-steps 300
```

Force re-run model inference (ignore cache):

```bash
python video_level_detection/run_video_level_detection.py --no-cache
```

## Outputs

- `video_level_detection/scores_cache/*_smoothed_scores.npy`
  - Cached per-frame smoothed scores generated from model inference.
- `video_level_detection/results/video_level_tuning_report.json`
  - Best `k`, best threshold, metrics, and per-video predictions.

## Evaluate on all test videos

After tuning, run full evaluation with the discovered best params:

```bash
python video_level_detection/evaluate_all_videos.py
```

This loads `k` and threshold from:
- `video_level_detection/results/video_level_tuning_report.json`

Or override manually:

```bash
python video_level_detection/evaluate_all_videos.py --k 10 --threshold 0.72
```

Output:
- `video_level_detection/results/video_level_full_eval.json`
  - Full confusion matrix, metrics, and per-video predictions for all test videos.

## LLM behavior (3 variants)

Run explanation generation across all test videos with three inputs:
- `frames_only`
- `frames_plus_score`
- `frames_score_heatmap`

```bash
python video_level_detection/run_llm_behavior_all_videos.py --variants all
```

Then judge each variant against human explanations (normal videos use default GT: "No anomaly in the video."):

```bash
python video_level_detection/judge_llm_behavior_all_videos.py --variants all
```

Outputs:
- `video_level_detection/results/llm_variants/per_video/*_{variant}.json`
- `video_level_detection/results/llm_variants/llm_{variant}_all_test_summary.json`
- `video_level_detection/results/llm_variants/judged_{variant}.json`
- `video_level_detection/results/llm_variants/judged_metrics_{variant}.json`
- `video_level_detection/results/llm_variants/judged_metrics_all_variants.json`
