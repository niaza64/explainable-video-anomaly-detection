# RTFM train visualization

**`plot_train_rtfm_scores.ipynb`**

### High quality (default)

- **`rtfm_train_segment_scores.pdf`** — **one page per video**, large figure size (`PAGE_W_IN` × `PAGE_H_IN` inches, default 14×5). Vector PDF: zoom/print friendly.
- **`pages_png/`** — optional **300 DPI** PNG per video (`SAVE_PNG_PAGES`, `PNG_DPI`).

Split into several PDFs if needed: set **`MAX_VIDEOS_PER_PDF`** (e.g. `20`); otherwise `0` = single PDF.

### Quick preview

- **`overview_grid.png`** — small multi-panel grid (not publication quality).
- **`snippet_score_summary.csv`**, **`rtfm_score_vs_segment_mean_std.png`** (when all clips share the same $T$).

Outputs go to `rtfm_train_viz/outputs/segment_curves/` (gitignored).

Requires: `matplotlib`, `torch`, `numpy`, `opencv-python`, `tqdm`; `pandas` optional.
