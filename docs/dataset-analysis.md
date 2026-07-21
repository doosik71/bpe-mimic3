# Dataset Analysis Plan

This document lists the statistical analyses that can be run against the
data currently sitting in `data/mimic3_index.csv`, `data/dataset/_progress.csv`,
and `data/dataset/{train,val,test}/*.npz`, now that a full dataset build
exists. It is a **plan only** — none of these analyses have been executed
yet; that is deferred to a follow-up pass (most naturally as the
`dataset-statistic` tool referenced in [README.md](../README.md),
[development-plan.md](development-plan.md), and
[data-cleaning.md](data-cleaning.md), which has not been implemented yet).

Current on-disk scale (checked while writing this plan): `mimic3_index.csv`
has 217,350 segment rows; `_progress.csv` lists 3,126 subjects; the final
split has 1,541 / 514 / 514 kept subjects in train/val/test.

Each analysis below is tagged with the data source it needs:

- **[index]** — `data/mimic3_index.csv` (pre-QC, segment-level: one row per
  WFDB segment that exposes both `PLETH` and an arterial pressure channel;
  columns `subject_id, record_name, record_dir, segment_name, segment_index,
  sample_offset, n_samples, fs, bp_signal, sig_names`).
- **[progress]** — `data/dataset/_progress.csv` (per-subject QC outcome:
  `subject_id, status (kept|excluded), n_windows_total, n_windows_kept`).
- **[npz]** — `data/dataset/{train,val,test}/*.npz` (post-QC, window-level:
  `x` PPG windows, `y` = [SBP, DBP], `calib_x`/`calib_y`, `fs` per kept
  subject).

## 1. Raw Corpus Coverage (index-level, pre-QC)

1. **Segment count per subject** — distribution of how many qualifying
   (PLETH + ABP/ART) segments each subject contributes; identifies subjects
   with unusually many/few candidate segments.
2. **Segment duration distribution** — `n_samples / fs` per segment, to see
   how much raw signal is available before windowing/QC even starts.
3. **BP signal naming split** — count of segments using `ABP` vs. `ART` as
   the arterial channel name (`bp_signal` column), confirming both are
   handled and checking whether one dominates.
4. **Available-channel co-occurrence** — parse `sig_names` to tabulate which
   other channels (`II`, `III`, `V`, `RESP`, etc.) most often accompany a
   PLETH+BP pair, useful context for any future multi-modal extension.
5. **Records vs. segments per subject** — using `record_name`, how many
   distinct ICU stays/records a subject contributes vs. how many segments
   per record; checks whether yield is dominated by a few long records or
   spread across many short ones.
6. **Total candidate signal duration** — sum of `n_samples / fs` across the
   whole index, as an upper bound on how much usable signal existed before
   any QC filtering (for comparison against the final kept-window duration
   in §3).

## 2. QC Retention Analysis (progress-level)

7. **Overall kept/excluded subject rate** — fraction of the 3,126 attempted
   subjects that ended up `kept` vs. `excluded`, i.e. the top-line
   patient-level attrition number.
8. **Window-level retention rate** — `sum(n_windows_kept) / sum(n_windows_total)`
   overall and per subject; this is the "~90% of raw segments discarded"
   claim from the README, now checkable against real numbers.
9. **Retention rate distribution across kept subjects** — histogram of
   `n_windows_kept / n_windows_total` per subject; identifies whether
   retention is uniformly moderate or bimodal (mostly-clean vs.
   mostly-rejected subjects).
10. **Exclusion reason breakdown** — of excluded subjects, how many failed
    on `n_windows_kept < min_valid_windows` (too few surviving windows) vs.
    `reject_fraction > max_reject_fraction` (too noisy overall); currently
    both collapse to `status=excluded` in `_progress.csv`, so this requires
    either recomputing from `n_windows_total`/`n_windows_kept` against the
    known thresholds (375, 95%) or extending the ledger with a reason
    column.
11. **`n_windows_total` distribution, kept vs. excluded** — do excluded
    subjects tend to have short recordings (few candidate windows to begin
    with) or long-but-noisy ones (many windows, still failing)?
12. **Retention-rate vs. threshold sensitivity** — the analysis the
    `dataset-statistic` tool was originally proposed for: recompute
    kept/excluded counts under alternative values of
    `min_valid_windows`, `max_reject_fraction`, `ppg_periodicity_threshold`,
    `abp_periodicity_threshold`, and `min_ppg_std`
    ([data-cleaning.md](data-cleaning.md) parameter table) to see how
    sensitive the final dataset size is to each threshold. Requires
    re-running the per-window filters (not just the ledger), so it depends
    on `bpe/preprocess/quality.py` rather than `_progress.csv` alone.

## 3. Final Dataset Composition (npz-level, post-QC)

13. **Subject and window counts per split** — confirm the 60/20/20
    patient-level split target is actually met (1,541/514/514 subjects
    counted above) and check the resulting window-count split, which need
    not match 60/20/20 exactly since windows-per-patient varies.
14. **Windows per subject distribution** — histogram/summary stats
    (min/median/mean/max) of `x.shape[0]` across all kept subjects; flags
    whether a handful of subjects dominate total window count (relevant to
    per-patient vs. per-window sampling weight during training).
15. **SBP distribution** — histogram, mean/std/percentiles of `y[:, 0]`
    pooled across all windows, and separately per split, to confirm splits
    are comparably distributed (no split-induced label shift).
16. **DBP distribution** — same as above for `y[:, 1]`.
17. **Pulse pressure distribution** — `SBP - DBP` per window; a derived
    clinical quantity worth checking for plausibility (should stay positive
    and within a physiologically sane range given the `[75,165]`/`[40,85]`
    bounds already enforced).
18. **SBP/DBP joint distribution** — scatter or 2D density of SBP vs. DBP,
    to see the correlation structure and whether the enforced range filters
    leave an oddly shaped joint distribution (e.g. clipped edges).
19. **Per-subject BP variability** — within-subject std of SBP and DBP
    across a subject's own windows; distinguishes subjects with stable BP
    throughout their stay from subjects with large swings (relevant since
    the calibration-based model predicts `ΔBP` from a per-subject
    baseline).
20. **Calibration-window BP vs. rest-of-subject BP** — compare
    `calib_y` against the mean/spread of that same subject's full `y` pool;
    checks whether the calibration reference (the surviving window closest
    to the subject's own median SBP/DBP, previously the chronologically-first
    window -- see [data-cleaning.md](data-cleaning.md) §4) is representative
    or an outlier relative to the subject's later windows.
21. **PPG amplitude statistics** — per-window std/peak-to-peak of `x`,
    pooled and per subject; a sanity check that the `min_ppg_std` gate
    (0.005) is doing its job and that no near-flatline windows slipped
    through (the failure mode documented in
    [data-cleaning.md](data-cleaning.md)'s case study).
22. **PPG waveform shape consistency** — mean and variance of the
    normalized PPG waveform aligned to systolic peak (superimposed pulse
    shape) across a sample of windows; a qualitative check that windows
    genuinely contain physiological pulses rather than passing QC on
    coincidental periodicity.
23. **Sample rate consistency check** — confirm every npz's `fs` scalar is
    exactly 125.0 with no stray resampled outliers.
24. **Total retained signal duration** — sum of `x.shape[0] * 8 s` (window
    count × window length) across all kept windows, compared against the
    total candidate duration from §1.6, giving the final, honest attrition
    percentage the README previously estimated as "on the order of 90%" (a
    full-scale run has since measured 55.2% window-level / 17.8%
    subject-level attrition and fixed a 2x overlap-counting bug in this
    comparison -- see [dataset-statistic.md](dataset-statistic.md) §2.6/§5).
25. **Age-like drift within subject (chronological check)** — using window
    order within a subject (already sorted chronologically per
    [data-cleaning.md](data-cleaning.md) §1), check whether BP trends
    (drifts up/down) over the course of a stay — informative for whether
    calibration-based correction should account for elapsed time, not just
    a fixed reference window.
26. **Train/val/test distributional parity** — compare SBP/DBP
    mean/std/percentiles and windows-per-subject distribution across the
    three splits directly (formalizes the check implied by §13/§15/§16);
    flags any split imbalance before it shows up as an unexplained
    train/test gap during model evaluation.

## Notes on Feasibility

- §1's analyses need only `pandas` over `data/mimic3_index.csv` (already a
  flat CSV, no WFDB access required).
- §2's analyses need only `data/dataset/_progress.csv`, except #12, which
  additionally needs to re-run `bpe/preprocess/quality.py` functions against
  raw signals (not just read the ledger), making it the most expensive item
  in this plan.
- §3's analyses need to load `x`/`y`/`calib_x`/`calib_y` from every kept
  subject's `.npz` (2,569 files across all splits); straightforward with
  `numpy.load`, but iterating all of them is the second most expensive item
  here after #12.
- None of these require touching `data/mimic3` (the raw WFDB symlink) —
  everything needed already lives in the derived `data/mimic3_index.csv` and
  `data/dataset/` outputs, consistent with the project rule that
  `data/mimic3` is read-only and only touched by the indexing step.
