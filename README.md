# Explainable Video Anomaly Detection

A modular pipeline for explainable video anomaly detection that decomposes the task into **video-level classification**, **temporal localization**, and **natural language explanation** — each component independently trainable under minimal supervision.

## Project Structure

```
├── data/                   # Datasets
│   ├── SHANGHAI/           # ShanghaiTech Campus dataset
│   │   ├── SHANGHAI_TRAIN/
│   │   ├── SHANGHAI_Test/
│   │   └── anomalous_videos/
│   └── Avenue_dataset_related_Work/
│       ├── Avenue Dataset/
│       ├── avenue/
│       └── avenue_cls_head/
│
├── aed-mae/                # Temporal anomaly localizer (unsupervised)
│   ├── model/              # MAE-CvT architecture
│   ├── configs/            # Training configs
│   ├── util/               # Utilities
│   ├── sam/                # SAM weights (spatial heatmaps)
│   ├── inference.py        # Score extraction
│   └── main.py             # Training entry point
│
├── rtfm/                   # Video-level classifier (weakly supervised)
│   ├── model.py            # RTFM architecture
│   ├── train.py            # Training script
│   ├── test_10crop.py      # Evaluation
│   └── list/               # Train/test splits
│
├── pipeline/               # Explanation generation & evaluation
│   ├── run_pipeline.py             # Full pipeline: detect → localize → explain
│   ├── run_full_evaluation.py      # Batch evaluation over all videos
│   ├── generate_explanations.py    # VLM explanation generation (V1/V2)
│   ├── generate_explanations_video_only.py  # V3: VLM-only baseline
│   ├── judge_explanations.py       # GPT-4o-as-judge evaluation
│   ├── outputs/                    # Generated explanations per video
│   └── results/                    # Aggregated metrics & tables
│
├── experiments/            # Archived earlier experiments
│   ├── video_level_detection/
│   ├── results_v2_segment_threshold_detection/
│   ├── new_plane/
│   └── my_attempt/
│
├── methodology.tex         # Paper methodology section
└── requirements.txt
```

## Pipeline Components

| Stage | Component | Supervision | Role |
|-------|-----------|-------------|------|
| 1. Classification | **RTFM** | Video-level labels (binary) | Anomalous vs. normal video |
| 2. Localization | **AED-MAE** | None (unsupervised) | Temporal segment with peak anomaly |
| 3. Explanation | **GPT-4o / Qwen2-VL** | 44 annotated descriptions | Natural language explanation |

## Key Findings

- **VLMs are temporally blind**: GPT-4o alone detects only 43.2% of anomalies. With AED-MAE guidance, 100%. Detection is the bottleneck, not generation.
- **Heatmaps add negligible value**: Pixel-level spatial attention (V1) vs. score-only input (V2) shows Δ ≤ 0.09 across all metrics. Temporal localization is what matters.
- **Minimal supervision**: The full pipeline requires only video-level binary labels + 44 descriptions, compared to 8,000+ annotated QA pairs for end-to-end approaches (HAWK, Holmes-VAU).
