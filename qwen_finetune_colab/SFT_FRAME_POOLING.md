# SFT frame pooling (spec)

How to build training rows from **RTFM snippet scores** + **human anomaly interval** only. There are **no RTFM “segments”** (no threshold, no merging of high-score runs). Strategies **1–4** use **every** snippet in **I**. Strategy **5** uses only the **top three** snippets by RTFM score inside **I**.

**Implementation:** `pool.py` (geometry + strategies), `build_sft_data.py` (CLI).

---

## Shared definitions

**Video:** `F` = total frame count, `T` = number of RTFM snippets (e.g. 32).

**Snippet `i` (`0 … T−1`) covers a frame interval** (same geometry as `pipeline/run_rtfm_pipeline.py`):

- `frames_per_snippet = F / T`
- `L(i) = ⌊ i · F / T ⌋` — first frame index of snippet `i`
- `R(i) = min(F − 1, ⌊ (i+1) · F / T ⌋ − 1)` — last frame index of snippet `i`  
  (if `L(i) > R(i)` in edge cases, use `R(i) = L(i)`.)

**Human anomaly interval:** `[a₀, a₁]` = `anomaly_start_frame`, `anomaly_end_frame` (inclusive).

**Snippets in the human window:** every index `i` such that snippet `[L(i), R(i)]` **intersects** `[a₀, a₁]`. Call this sorted list **I** = `(i₁, …, iₖ)`. If **I** is empty, skip the video or fall back to a single snippet whose interval is closest to `[a₀, a₁]` (implementation choice).

**RTFM score:** one scalar `s[i]` per snippet (logit / mean as in your RTFM forward). For a **frame** taken from snippet `i`, attach **that snippet’s** score `s[i]` (even when the pixel comes from a frame other than the snippet mid).

**Training row:** ordered frames + matching scores + prompt template; **same** human `explanation` for every strategy row from that video.

**Dedup:** when building a row, if the same frame index appears twice, keep one copy in **time order** (or drop duplicate indices).

**Why not “start + end frame per snippet”?** For consecutive snippets `i` and `i+1`, `R(i)` and `L(i+1)` are **adjacent** in time. Taking **both** `L(i)` and `R(i)` for every `i` makes the row full of **seam duplicates** (`R(i)` almost the same instant as the next snippet’s start). Taking **only** `L(i)` per snippet (strategy **2**) is fine: successive `L(i)` are **not** adjacent — they are ~`F/T` frames apart.

---

## Five strategies

| # | Name | Rule |
|---|------|------|
| **1** | `every_snippet_mid` | For each `i ∈ I`, take **one** frame: the **middle** of snippet `i`, i.e. `⌊ (L(i) + R(i)) / 2 ⌋` (same idea as `snippet_to_frame_num`’s midpoint). |
| **2** | `every_snippet_first` | For each `i ∈ I`, take **one** frame: the **first** frame of the snippet, **`L(i)`**. Temporal order = increasing `i`. |
| **3** | `every_snippet_mid_frame_band` | For each `i ∈ I`, let `m = ⌊ (L(i) + R(i)) / 2 ⌋`. Take every frame `m + d` with integer `d ∈ [−δ, δ]` **clamped** to `[L(i), R(i)]` (default **`δ = 2`** → up to five frames per snippet). |
| **4** | `human_span_smart` | Work only with snippets **I**. Fix a global **frame budget** `B` and **minimum snippet-index gap** `G`. Always include the **first** and **last** snippet in **I** (smallest and largest index in **I**). Split the remaining budget across **I** proportionally to snippet “length” `R(i)−L(i)+1` if you want, or use a single pool: from interior snippets in **I**, add indices in **descending** `s[i]` order while no chosen pair has index distance `< G`. (Same *idea* as the old “smart” pool, but the allowed set is **I**, not thresholded segments.) |
| **5** | `top3_snippets_mid_frame_band` | Sort snippets in **I** by **`s[i]` descending** (ties: smaller snippet index `i` first). Take the **first `K = min(3, |I|)`** distinct snippets. For **each**, apply the **same** mid±δ **frame** rule as strategy **3** (same default **`δ = 2`**). **Order in the row:** all frames from the **highest‑score** snippet (sorted by frame index), then the **second**, then the **third** — different from strategy **3**, which runs over **every** snippet in **I**. |

---

## Notes

- **Scale:** strategies **1** and **2** → **k** images each; **3** → up to **5k**; **5** → up to **15** (3 snippets × 5 frames) before dedup — strategy **3** grows fast when **|I|** is large.
- **Validation:** several rows share the same caption — evaluate with **held-out `video_id`s**, not i.i.d. rows.
