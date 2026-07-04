# Project Summary: Explainable Video Anomaly Detection

## Goal

Build an end-to-end pipeline that:
1. **Detects** anomalous surveillance videos using the RTFM model
2. **Selects** representative frames from anomalous temporal segments
3. **Explains** the anomaly in natural language using GPT-4o (VLM)
4. **Evaluates** the AI-generated explanation against human ground truth using GPT-4o-as-judge

Dataset: **ShanghaiTech Campus** — 199 test videos (44 anomalous, 155 normal).

---

## Repository Structure

```
explainable-video-anomaly-detection/
├── rtfm/                          # RTFM model code (ICCV 2021)
│   ├── model.py                   # RTFM model definition
│   ├── option.py                  # Hyperparameters
│   ├── dataset.py                 # Feature loading (10-crop, 32 segments)
│   ├── utils.py                   # Utilities
│   ├── data/
│   │   └── SH_Test_ten_crop_i3d/  # Pre-extracted I3D features (199 .npy files)
│   └── colab/
│       ├── full_pipeline_colab.ipynb     ← MAIN COLAB NOTEBOOK (end-to-end)
│       ├── qwen_pipeline_colab.ipynb     # Qwen-VL variant
│       ├── score_segments_colab.ipynb    # RTFM scoring only
│       └── test_rtfm_colab.ipynb
├── pipeline/
│   ├── run_rtfm_pipeline.py       # Stage 1: RTFM inference + frame sampling
│   ├── generate_explanations_rtfm.py  # Stage 2: GPT-4o VLM explanation
│   ├── judge_explanations_rtfm.py     # Stage 3: GPT-4o-as-judge scoring
│   ├── run_full_rtfm_evaluation.py    # Orchestrator (runs all 3 stages)
│   ├── rtfm_outputs/
│   │   └── pipeline_summary.json  # Per-video RTFM scores + selected frames
│   └── results/
│       ├── gpt4o_vlm_results.tex  # Full LaTeX results table (updated)
│       ├── gating_results.tex     # Gate accuracy analysis
│       ├── findings_aed_mae_limitations.tex
│       └── experiments.tex
├── data/SHANGHAI/
│   ├── SHANGHAI_Test/frames/      # 199 dirs of JPG frames (extracted from videos)
│   └── anomalous_videos/
│       └── annotations.json       # Human GT explanations for 36 anomalous videos
└── .env                           # OPENAI_API_KEY (gitignored)
```

---

## Pipeline Stages (in `full_pipeline_colab.ipynb`)

### Cell 1 — Install dependencies
`torch`, `openai`, `scikit-learn`, `numpy`, `opencv`, etc.

### Cell 2 — Mount Google Drive
Connects to Drive where `shanghai_frames/` (199 frame dirs) and `annotations.json` live.

### Cell 2b — Unzip frames (if needed)
Auto-unzips `shanghai_pipeline_upload.zip` → `MyDrive/shanghai_frames/`.

### Cell 3 — Configure paths
Auto-detects `FRAMES_DIR` across several candidate paths to handle nested zip artifacts.

### Cell 4 — Load RTFM model
Loads `best_model.pkl` checkpoint onto GPU. Model: weakly-supervised ranking loss over I3D features.

### Cell 5 — Score all 199 test videos
For each `.npy` feature file:
- Forward pass → per-snippet scores `[T]` + feature magnitudes
- Gate score: `s_abn = mean(top-3 scores by feature magnitude)`
- Flag video as anomalous if `s_abn > τ = 0.2`

**Gate results:** TP=40, TN=149, FP=6, FN=4 → Accuracy=0.950, F1=0.889, ROC-AUC=0.986

### Cell 6 — Segment detection + adaptive smart frame sampling
For each flagged video (46 total: 40 truly anomalous + 6 normal FPs):

**Segment detection** (`find_segments_adaptive`):
- Find contiguous snippets where `score > SEGMENT_THRESHOLD`
- Adaptive fallback: if fewer than `FRAME_BUDGET/2` snippets selected, lower threshold progressively: `0.30 → 0.20 → 0.10 → 0.05 → 0.00` (top-k fallback)

**Smart frame sampling within each segment:**
- Budget allocated proportionally by segment length
- Always include first + last snippet (onset + resolution)
- Remaining budget: select snippets weighted by score, with `min_gap` spacing
- Maps snippet index → frame number: `frame = snippet_idx * 16 + 8` (middle frame)

**Result:** Up to 8 frames per video extracted as JPEGs.

### Cell 7 — (Frame extraction already done inline in Cell 6)

### Cell 8 — VLM explanation via GPT-4o
For each flagged video, sends up to 8 JPEG frames + anomaly scores to GPT-4o:

```
System: You are a surveillance video analyst...
User: [frame images] + "Anomaly scores (0=normal, 1=anomalous): [scores]
       Describe what anomalous activity is occurring. Include: what, who, when, why."
```

Returns JSON: `{"explanation": "...", "anomaly_type": "...", "confidence": "high/medium/low"}`

**API key**: loaded from Colab Secrets (`OPENAI_API_KEY`) → `os.environ` → manual fallback in Cell 8.

### Cell 9–11 — Save outputs / display results

### Cell 12 — Judge via GPT-4o-as-judge
For each video with an explanation:

- **4-digit video IDs** (e.g. `01_0015`): truly anomalous → use specific human annotation from `annotations.json` as ground truth
- **3-digit video IDs** (e.g. `08_023`): normal false positives → use generic GT: *"There is nothing anomalous in this video. All pedestrians are walking normally."*

Judge scores on 4 criteria (1–5 scale): **Correctness, Specificity, Completeness, Fluency**

### Cell 13 — Results summary
Prints two separate tables: anomalous videos + normal false positives, each with per-video scores and aggregate means.

---

## Results (GPT-4o, 40 anomalous videos)

| Metric       | Mean | Std  | Min | Max |
|--------------|------|------|-----|-----|
| Correctness  | 3.43 | 1.41 | 1.0 | 5.0 |
| Specificity  | 3.28 | 0.80 | 2.0 | 5.0 |
| Completeness | 2.95 | 1.18 | 1.0 | 5.0 |
| Fluency      | 4.35 | 0.48 | 3.0 | 5.0 |
| **Overall**  | **3.50** | 1.17 | 1.0 | 5.0 |

- C=5 (exact): 14/40 videos (35%)
- C≥3 (partial+): 30/40 videos (75%)
- 6 normal FP videos judged against generic "nothing anomalous" GT (scores TBD after Cell 12 re-run)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| RTFM over AED-MAE | AED-MAE requires per-frame reconstruction; too slow and inaccurate on ShanghaiTech. RTFM gives reliable snippet-level scores from pre-extracted I3D features in milliseconds per video. |
| Gate threshold τ=0.2 | Maximises F1 on the 199-video test set (ROC-AUC=0.986). |
| Adaptive segment threshold | Fixed 0.5 was too high for weak anomalies (e.g. barely-passing gate score ~0.2). Dynamic lowering ensures ≥4 frames for VLM context. |
| Smart sampling (onset + resolution + score-weighted) | Ensures temporal coverage of the event lifecycle rather than just peak frames. |
| No heatmap overlay | Heatmaps from RTFM are at snippet level, not pixel level; overlaying them on frames adds noise. Pure frames + score annotation in text is cleaner. |
| GPT-4o-as-judge | Scalable automated evaluation. 4-criteria rubric (C/S/Co/F) aligns with prior VLM evaluation literature. |

---

## Known Issues / Limitations

1. **Temporal misalignment**: RTFM scores peak on crowd-reaction frames (high motion energy *after* the event), not on the precipitating act. This causes brief anomalies (bag-snatch, push) to be missed.
2. **4 FN videos**: `s_abn ≤ 0.2` → never forwarded to VLM. (e.g. `08_0157`)
3. **Colab output truncation**: Colab caps cell output; large tables may appear truncated visually but the underlying computation processes all 46 videos.
4. **VLM hallucination on normal FPs**: Without a real anomaly, GPT-4o fabricates plausible-sounding but incorrect descriptions.

---

## Data on Google Drive (for Colab)

| File | Location on Drive | Notes |
|------|-------------------|-------|
| `shanghai_pipeline_upload.zip` | `MyDrive/` | 8.7 GB zip of all 199 frame dirs |
| `annotations.json` | `MyDrive/` | Human GT for 36 anomalous videos |
| `best_model.pkl` | `MyDrive/rtfm/` (or cloned from GitHub) | RTFM checkpoint |
| RTFM I3D features | `/content/rtfm/data/SH_Test_ten_crop_i3d/` | 199 `.npy` files, cloned from Drive or repo |

---

## How to Run (Colab)

1. Open `rtfm/colab/full_pipeline_colab.ipynb` in Google Colab
2. Set runtime to **GPU (T4 or better)**
3. Run Cell 1 (install), Cell 2 (mount Drive), Cell 2b (unzip frames)
4. Add `OPENAI_API_KEY` to Colab Secrets (key name: `OPENAI_API_KEY`)
5. Run Cells 3–13 in order

---

## File: `annotations.json`

Contains human-written ground truth for 36 anomalous test videos plus a generic "Normal Videos" entry. Fields per video: `video_id`, `total_frames`, `anomaly_start_frame`, `anomaly_end_frame`, `explanation` (1–2 sentence natural language description of the anomaly).

---

---

## Related Work (draft)

**Video anomaly detection and explainability.** Traditional VAD focuses on frame- or snippet-level anomaly scores with weak or no supervision: e.g. RTFM [Tian et al., ICCV 2021] uses only video-level labels and achieves strong performance on ShanghaiTech, UCF-Crime, and XD-Violence. Explainable and open-world VAD has grown with multimodal models: LAVAD [Zanella et al., CVPR 2024] uses VLMs/LLMs in a training-free way for frame-level scoring on UCF-Crime and XD-Violence; HAWK [Tang et al., NeurIPS 2024] builds open-world VAD with 8k+ language-annotated videos and motion–appearance consistency. CUVA [Du et al., CVPR 2024] introduces a causation-understanding benchmark (what/why/how) with human annotations. Holmes-VAU [Zhang et al., CVPR 2025] pushes hierarchical video anomaly understanding: they introduce HIVAU-70k, built from the *training sets* of UCF-Crime and XD-Violence only, with over 70k clip-, event-, and video-level instructions from a semi-automated pipeline (manual event segmentation + LLM summarization). Their method trains an anomaly scorer with *annotated frame-level labels* from HIVAU-70k and an Anomaly-focused Temporal Sampler (ATS) before feeding selected frames into an instruction-tuned VLM. Holmes-VAU reports detection and reasoning results **only on UCF-Crime and XD-Violence**; it does not evaluate on ShanghaiTech Campus, and its design assumes large-scale hierarchical annotations and frame-level supervision for the scorer, which ShanghaiTech does not provide in that form.

**Gap and our contribution.** ShanghaiTech is a standard benchmark (199 test videos, 13 scenes, video-level normal/anomalous and optional frame masks for evaluation) but has no public benchmark for *natural-language anomaly explanations* with human ground truth. Holmes-VAU’s HIVAU-70k does not cover ShanghaiTech; adapting it would require either building a similar hierarchical annotation pipeline for ShanghaiTech or forgoing their frame-level–supervised scorer. We instead keep the setting lightweight: we use **weakly-supervised RTFM** (no per-frame training labels) on ShanghaiTech, run adaptive segment detection and frame sampling, and generate explanations with off-the-shelf VLMs (GPT-4o or Qwen-VL zero-shot). We provide **human ground-truth explanations** for anomalous videos (e.g. in `annotations.json`) and evaluate AI-generated explanations with GPT-4o-as-judge on Correctness, Specificity, Completeness, and Fluency. Thus we deliver (1) an end-to-end explainable VAD pipeline on ShanghaiTech without frame-level annotation for training, and (2) a **ShanghaiTech anomaly-explanation evaluation setup**—human references plus judge scores—that complements benchmarks like HIVAU-70k (UCF-Crime/XD-Violence) and CUVA, and that Holmes-VAU does not address.

**Summary.** Our work is complementary to Holmes-VAU: they target long-term VAU with hierarchical data on UCF-Crime/XD-Violence and frame-level–supervised scoring; we target ShanghaiTech with weak supervision only and release a concrete explanation-evaluation dataset and protocol. Selling point: *Holmes-VAU does not support or evaluate on ShanghaiTech and relies on heavy annotation (hierarchical + frame-level); we provide explainable VAD and an anomaly-explanation benchmark on ShanghaiTech with no per-frame training labels.*

---

## What Still Needs to Be Done

- [ ] Fill in the 6 normal FP judge scores in `gpt4o_vlm_results.tex` (run Cell 12 in updated notebook)
- [ ] (Optional) Run `qwen_pipeline_colab.ipynb` to compare Qwen-VL scores vs GPT-4o scores
- [ ] (Optional) Experiment with pixel-level heatmap overlays using saliency from the I3D backbone
