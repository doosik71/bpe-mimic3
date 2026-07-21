# Dataset Statistic Tool: Design, Usage, and Results

`scripts/dataset-statistic.py` (`bin/dataset-statistic`) is the analysis tool
planned in [dataset-analysis.md](dataset-analysis.md) and referenced (as a
not-yet-built tool) throughout [README.md](../README.md),
[development-plan.md](development-plan.md), and
[data-cleaning.md](data-cleaning.md). This document is its design record,
user guide, and the results of the first full run against the real dataset
(2026-07-21).

## 1. Purpose

Give a single, repeatable report over everything the preprocessing pipeline
produces, at three points in the pipeline:

1. **Pre-QC segment index** (`data/mimic3_index.csv`) -- how much raw
   material exists before any filtering.
2. **QC retention ledger** (`data/dataset/_progress.csv`) -- how many
   subjects/windows survived, and how unevenly.
3. **Final npz splits** (`data/dataset/{train,val,test}/*.npz`) -- the
   PyTorch-ready dataset itself: label distributions, per-subject
   concentration, calibration-window representativeness, and QC-gate sanity
   checks.

It was built to answer the specific open questions
[development-plan.md](development-plan.md) §7 and
[data-cleaning.md](data-cleaning.md) flagged as unvalidated (the periodicity
thresholds and `min_valid_windows`, both seeded from the source paper's
30 s-window values and never checked against real 8 s-window yield).

## 2. Detailed Design

### 2.1 Inputs

| Source | Level | Cost |
| --- | --- | --- |
| `data/mimic3_index.csv` | one row per qualifying WFDB segment | cheap -- one `pandas.read_csv` over a flat file, no WFDB access. Optional: skipped (not an error) if the file isn't found. |
| `data/dataset/_progress.csv` | one row per attempted subject | cheap -- reuses `bpe.preprocess.pipeline.read_progress`, the same reader `construct-dataset` itself uses for resumability. |
| `data/dataset/{train,val,test}/*.npz` | one file per kept subject | the expensive part -- see 2.3. |

### 2.2 Module Structure

Everything lives in the one script file (no new `bpe/` module), mirroring
the reference sample this was built from and matching the size/scope of
similar single-purpose analysis scripts already in `scripts/`:

| Function | Responsibility |
| --- | --- |
| `_summary_stats` / `_concentration_stats` | Generic reusable stat blocks (mean/std/percentiles; top-10%-concentration + top-5 list) shared by every section below. |
| `load_index_stats` / `plot_index_overview` | Section 1: segments/duration per subject, records per subject, BP channel naming, co-occurring channels. |
| `load_progress_stats` / `plot_retention_overview` | Section 2: kept/excluded counts, per-subject retention-rate distribution, attempted-window counts for kept vs. excluded. |
| `_load_one_subject` / `load_split` / `merge_splits` / `compute_split_summary` | Section 3 data loading: one npz in, one small per-subject result dict out (see 2.3), then pooled into per-split and dataset-wide ("all") summaries. |
| `plot_bp_distribution`, `plot_windows_per_subject`, `plot_bp_sd_per_subject`, `plot_calibration_offset`, `plot_ppg_amplitude` | Section 3 plots, one concern each. |
| `main` | Wires the three sections together, prints a summary table, writes `statistic.json` and every `*.png`. |

### 2.3 Design Decisions

- **`x` is never held past one file.** A subject's window array can be
  `(N, 1000)` with `N` in the hundreds of thousands (the largest subject in
  the full run has 1,041,821 attempted windows). `_load_one_subject` reduces
  `x` to a per-window `std` immediately inside the `with np.load(...)` block
  and returns only that -- the raw windows are never concatenated across
  subjects, so memory stays bounded regardless of dataset size. This is the
  single most expensive step (full I/O read of every window), so it's
  optional via `--skip-ppg-amplitude`.
- **Parallel npz loading.** `load_split` uses a `ThreadPoolExecutor`
  (`--workers`, default 8) the same way `build-mimic3-index.py` and
  `construct-dataset.py` already do, with a `tqdm` progress bar
  (`--no-progress` to disable). `np.load` releases the GIL during the actual
  file read, so threads (not processes) are enough here.
- **The index and progress ledger are optional inputs, not requirements.**
  A user can run this against a dataset that hasn't been fully indexed, or
  before `_progress.csv` exists; each section degrades independently (prints
  a one-line notice and sets that key to `null` in `statistic.json`) rather
  than failing the whole run.
- **`--limit-subjects` caps only the npz-loading section**, for quick
  iteration while developing/debugging the script itself; the index and
  progress sections always reflect the full ledger since they're cheap
  regardless of scale.

### 2.4 Output Artifacts

Written under `--output-dir` (default: same as `--dataset-dir`, i.e.
`data/dataset` itself):

| File | Content |
| --- | --- |
| `statistic.json` | Every numeric result below, nested under `index` / `progress` / `splits.{train,val,test,all}` / `overall_yield_pct`. |
| `index_overview.png` | Segments/duration per subject, BP channel naming (skipped if the index csv is missing). |
| `retention_overview.png` | Kept/excluded counts, per-subject retention rate, attempted-window counts for kept vs. excluded. |
| `bp_distribution.png` | SBP / DBP / pulse-pressure density per split. |
| `windows_per_subject.png` | Windows-per-subject concentration per split. |
| `bp_sd_per_subject.png` | Within-subject SBP/DBP variability per split. |
| `calibration_offset.png` | Calibration-window BP vs. subject's own mean BP. |
| `ppg_amplitude.png` | Per-window PPG std vs. the `min_ppg_std` QC gate (skipped with `--skip-ppg-amplitude`). |

### 2.5 Deferred Analyses

Three items from [dataset-analysis.md](dataset-analysis.md) are deliberately
**not** implemented here:

- **#10, exclusion-reason breakdown** (too few windows vs. too noisy) --
  `_progress.csv` always logs excluded subjects with `n_windows_kept=0`; the
  true `n_valid` used inside `should_exclude_patient` is discarded before
  being written, so the two reasons can't be told apart from the ledger
  alone. Fixing this would require a `_progress.csv` schema change in
  `bpe/preprocess/pipeline.py`, not just a new statistic.
- **#12, retention-vs-threshold sensitivity** -- needs re-running the
  per-window QC filters (`bpe/preprocess/quality.py`) against raw signals
  under alternative threshold values, not just reading stored outputs. A
  meaningfully heavier tool than the rest of this script.
- **#22, superimposed pulse-shape consistency** -- a visual/qualitative
  check, better suited to `bin/dataset-browser`'s waveform view than a
  batch statistic.

### 2.6 Known Limitation Found During the Full Run

Running this tool against the complete dataset (§4 below) surfaced a real
unit-mismatch bug in one derived metric, `overall_yield_pct`:

- `total_candidate_duration_hr` (from the index) is **true elapsed time**:
  `sum(n_samples / fs)` over raw, non-overlapping segments.
- `total_retained_duration_hr` (from the npz splits) is
  `window_count * window_sec`, computed from windows that were extracted at
  an 8 s length with a **4 s stride** (50% overlap, per
  [data-cleaning.md](data-cleaning.md) §2). For a fully-retained span of real
  duration `D`, that span yields `D / 4` windows, each counted as 8 s -- i.e.
  `~2D`. So `total_retained_duration_hr` runs at roughly **2x** the real
  elapsed time it represents.

`overall_yield_pct = retained_hr / candidate_hr` therefore compares two
differently-scaled quantities and reads misleadingly high (89.45% in §4.4,
implying only ~10% attrition) while the already-consistent
`window_retention_rate_overall` (`windows_kept / windows_attempted`, both
counted the same stride-windowed way, so the 2x factor cancels) reports the
real figure: **44.8%** window-level retention. The math checks out exactly:
`windows_total_attempted * 8s` = 430,501 h, vs. the index's
215,448.6 h candidate duration -- a 2.00x ratio, confirming the overlap
factor is the entire explanation.

**`overall_yield_pct` should not be trusted as-is** -- treat
`window_retention_rate_overall` (§4.3) as the correct yield figure instead.
Left in place for this write-up rather than silently patched, per this
project's assumption-transparency convention; see the recommendation in §5
for the fix under consideration.

## 3. User Guide

### 3.1 Running the Tool

```bash
uv run python scripts/dataset-statistic.py
# or, equivalently:
bin/dataset-statistic          # Linux/macOS
bin\dataset-statistic.bat      # Windows
```

Requires `data/dataset/_progress.csv` and/or `data/dataset/{train,val,test}`
to exist already (run `bin/construct-dataset` first). `data/mimic3_index.csv`
is read too if present, but its absence only skips §1 of the report.

### 3.2 CLI Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset-dir PATH` | `data/dataset` | Root holding `_progress.csv` and `train/val/test/`. |
| `--index-csv PATH` | `data/mimic3_index.csv` | Pre-QC segment index; skipped (not an error) if missing. |
| `--output-dir PATH` | same as `--dataset-dir` | Where `statistic.json` and every `*.png` are written. |
| `--limit-subjects N` | none | Only load the first N subjects per split -- for a quick trial run while iterating on the script itself. |
| `--skip-ppg-amplitude` | off | Skip loading `x` entirely (the most expensive step) -- drops `ppg_amplitude.png` and the `ppg_window_std` / `ppg_windows_below_min_std_pct` fields from `statistic.json`. |
| `--no-plots` | off | Only write `statistic.json`; skip all PNGs. |
| `--workers N` | 8 | Thread-pool size for concurrent npz loading. |
| `--no-progress` | off | Disable the tqdm progress bars. |

### 3.3 Common Invocations

```bash
# Full report (what produced §4 below)
uv run python scripts/dataset-statistic.py

# Quick smoke test while developing the script -- 5 subjects/split, fast
uv run python scripts/dataset-statistic.py --limit-subjects 5 --output-dir /tmp/ds-test

# Numbers only, no plots, no PPG-amplitude I/O -- fastest possible run
uv run python scripts/dataset-statistic.py --skip-ppg-amplitude --no-plots

# Point at a different output location so re-running doesn't clobber
# an existing report while iterating on thresholds
uv run python scripts/dataset-statistic.py --output-dir data/dataset/report_v2
```

Re-running with the same `--output-dir` overwrites the previous
`statistic.json` and PNGs in place; there is no versioning, so copy them out
first if you want to diff against a later run (e.g. after re-tuning a QC
threshold and re-running `construct-dataset --force`).

### 3.4 Runtime Cost

`--skip-ppg-amplitude` off (the default) means every window's `x` array is
read from disk once. On the full dataset (2,569 kept subjects, ~86.7M
windows) this is I/O-bound: most files load in well under a second, but a
handful of very large subjects (over 1M attempted windows) can take over a
minute each depending on storage throughput, especially if `data/dataset`
is a symlink onto a network/USB drive rather than local SSD. There is no
progress checkpointing across a run -- an interrupted run must be restarted
from scratch (unlike `construct-dataset`'s resumable `_progress.csv`
design).

### 3.5 Interpreting `statistic.json`

Top-level keys:

- `index` -- `null` if `data/mimic3_index.csv` wasn't found, else the §2.4
  `index_overview.png` numbers.
- `progress` -- `null` if `_progress.csv` wasn't found, else the §2.4
  `retention_overview.png` numbers, plus `note_exclusion_reason_breakdown`
  explaining why exclusion causes can't be split out (§2.5 #10).
- `splits.train` / `.val` / `.test` / `.all` -- each holds `sbp`, `dbp`,
  `pulse_pressure`, `windows_per_subject`, `sbp_sd_per_subject` /
  `dbp_sd_per_subject`, `calib_sbp_offset_from_subject_mean` /
  `calib_dbp_offset_from_subject_mean`, `sbp_drift_corr_per_subject` /
  `dbp_drift_corr_per_subject`, `distinct_fs_values`, and (unless
  `--skip-ppg-amplitude`) `ppg_window_std` / `ppg_windows_below_min_std_pct`.
  `.all` pools every split together for a dataset-wide view.
- `overall_yield_pct` -- see the §2.6 caveat before using this number.

## 4. Dataset Statistical Analysis Results

Run on 2026-07-21 against the full dataset with default settings (no
`--limit-subjects`, PPG amplitude check included). Outputs live in
`data/dataset/` (`statistic.json` and the seven `*.png` files listed in
§2.4); this section narrates what they show. `data/` is git-ignored, so
these files exist only on the machine that ran the tool -- re-run
`bin/dataset-statistic` to regenerate them elsewhere.

### 4.1 Run Configuration

| | |
| --- | --- |
| dataset dir | `data/dataset` |
| index csv | `data/mimic3_index.csv` |
| PPG amplitude check | enabled |
| subjects limit | none (full dataset) |

### 4.2 Pre-QC Segment Index (`index_overview.png`)

| Metric | Value |
| --- | --- |
| Qualifying segments | 217,350 |
| Subjects | 3,126 |
| Distinct records | 5,314 |
| Total candidate duration | 215,448.6 h (~24.6 years) |
| Segments per subject | mean 69.5, median 18, max 3,978 |
| Records per subject | mean 1.70, median 1, max 14 |
| Segments per record | mean 40.9, median 11, max 3,978 |
| `ABP` vs. `ART` naming | 206,168 (94.9%) vs. 11,182 (5.1%) |

**Segment duration is extremely right-skewed**: mean 3,568.5 s (~59.5 min)
but **median only 3.0 s** (p25 = 1.0 s, p75 = 751 s, max = 312,243 s ≈
86.7 h). Half of all "qualifying" segments (PLETH + BP both present) are
essentially fragments a few seconds long, almost certainly the many short
data segments a multi-segment WFDB record splits into around sensor
dropouts/gaps; the total candidate duration is dominated by a comparatively
small number of long, continuous segments. This matters for interpreting
"segments per subject" too -- a subject with hundreds of segments may still
have very little real signal if most of those segments are few-second
fragments.

**Co-occurring channels** (of the 217,350 qualifying segments, how many
also carry each channel): `II` in 204,119 (93.9%), `RESP` in 186,994 (86.0%),
`V` in 178,036 (81.9%), `AVR` in 145,328 (66.9%), `CVP` in 71,048 (32.7%).
ECG (`II`) and respiration are present in the large majority of qualifying
segments -- useful context if a future extension wants to bring in
additional channels.

### 4.3 QC Retention (`retention_overview.png`)

| Metric | Value |
| --- | --- |
| Subjects attempted | 3,126 |
| Subjects kept | 2,569 (82.2%) |
| Subjects excluded | 557 (17.8%) |
| Windows attempted (all subjects) | 193,725,620 |
| Windows kept | 86,726,310 |
| **Window-level retention rate** | **44.8%** |
| Per-subject retention rate | mean 48.2%, median 49.4%, p25 30.7%, p75 66.4% |
| Windows attempted, kept subjects | mean 70,206, median 37,669, max 1,041,821 |
| Windows attempted, excluded subjects | mean 23,995, median 1,534, p25 42, max 571,129 |

This is a materially better yield than the project's own working
assumption: the README and [data-cleaning.md](data-cleaning.md) both cite
"on the order of 90%" window attrition (echoing the source paper's ~95%),
but the observed figures are **55.2% window-level attrition** and only
**17.8% subject-level attrition** -- notably milder. See §5 for the
implication.

The per-subject retention-rate histogram (middle panel) is broad and
roughly unimodal, centered around 45-55%, without a sharp cliff at the 5%
`max_reject_fraction` floor (dotted line) -- most kept subjects clear the
bar comfortably rather than barely scraping past it.

The right panel (attempted windows, kept vs. excluded, log scale) shows
excluded subjects split into two rough populations: a large cluster with
very few attempted windows (consistent with short recordings that never had
a chance -- p25 of only 42 attempted windows), and a smaller but non-trivial
tail extending up to 571,129 attempted windows -- subjects with substantial
raw data who were still excluded, implying rejection by noise/periodicity
(`max_reject_fraction`) rather than by sheer shortness. `_progress.csv`
cannot currently distinguish these two populations per-subject (§2.5 #10),
but the aggregate shape here is suggestive evidence both failure modes are
real and neither dominates completely.

### 4.4 Final Dataset Composition

| Split | Subjects | Windows | Retained duration* |
| --- | --- | --- | --- |
| train | 1,541 | 52,582,131 | 116,849.2 h |
| val | 514 | 17,719,901 | 39,377.6 h |
| test | 514 | 16,424,278 | 36,498.4 h |
| **all** | **2,569** | **86,726,310** | **192,725.1 h** |

\* window-count x 8 s; per §2.6, this runs ~2x real elapsed time due to 50%
window overlap -- do not compare directly against §4.2's candidate duration.

**Patient-level split ratios** are close to the configured 60/20/20 by
subject count (60.0% / 20.0% / 20.0%), and reasonably close by window count
too (60.6% / 20.4% / 18.9%) -- the split isn't meaningfully skewed by a few
window-heavy subjects landing disproportionately in one split.

**SBP / DBP / pulse pressure** (`bp_distribution.png`), pooled and by split:

| | SBP (mmHg) | DBP (mmHg) | Pulse pressure (mmHg) |
| --- | --- | --- | --- |
| train | 113.0 ± 19.0 | 60.3 ± 10.3 | 52.8 ± 17.4 |
| val | 112.4 ± 18.4 | 60.2 ± 10.2 | 52.2 ± 16.9 |
| test | 113.6 ± 18.9 | 61.1 ± 10.4 | 52.5 ± 17.5 |
| all | 113.0 ± 18.9 | 60.4 ± 10.3 | 52.6 ± 17.4 |

Means differ by at most ~1.2 mmHg (SBP) and ~0.9 mmHg (DBP) across splits --
no meaningful label-distribution shift between train/val/test. Shapes
(`bp_distribution.png`) are unimodal and consistent with the enforced
`[75,165]`/`[40,85]` mmHg range filters. **Pulse pressure has a slightly
negative minimum in every split** (train -3.92, val -2.40, test -4.85 mmHg)
-- a small number of windows have labeled DBP marginally exceeding SBP.
Neither `physiological_range_ok` nor any other current filter checks
`SBP > DBP` directly (each is range-checked independently), so this
edge case slips through; worth a look if it turns out to matter for
training (see §5).

**Windows-per-subject concentration** (`windows_per_subject.png`): highly
skewed in every split -- train max/median ratio 40.5x (top 10% of subjects
hold 47.5% of train windows), val 19.8x (45.5%), test 19.1x (42.4%). The
single largest subject (`p083182`, train) alone contributes 634,084 windows
-- 1.2% of the entire train split and over 40x the train median (15,671).
Per-window sampling
during training will heavily overweight a handful of subjects unless
explicitly corrected (e.g. per-subject weighting), consistent with the
concern [dataset-analysis.md](dataset-analysis.md) #14 flagged.

**Within-subject BP variability** (`bp_sd_per_subject.png`): SBP SD mean
12.0 mmHg (median 11.7); DBP SD mean 6.7 mmHg (median 6.6). Only **0.9%** of
subjects have SBP SD below the 5 mmHg "easy to calibrate" reference line,
and only **18.8%** for DBP. In other words, the large majority of subjects
carry genuine, non-trivial intra-subject BP variation -- a per-subject mean
(or a naive calibration offset) would leave substantial residual error for
almost every subject, which is the case the calibration-based (Siamese)
model is meant to address.

**Calibration-window representativeness** (`calibration_offset.png`): the
calibration window's BP vs. that same subject's own mean BP is broad and
only mildly biased -- SBP offset mean -1.8 mmHg (median -2.7, std 16.0,
p25/p75 -15.3/+10.7), DBP offset mean +1.8 mmHg (median +1.6, std 11.0,
p25/p75 -5.7/+9.1). The chronologically-first surviving window (used as the
calibration reference) is *on average* close to the subject's own mean, but
individual subjects can differ by over 30 mmHg in either direction (min/max
around ±35 SBP mmHg) -- for those subjects, the calibration reference is a
poor stand-in for "typical" BP, and the Siamese model has to correct a large
baseline offset rather than a small one.

**Chronological BP drift** (`sbp_drift_corr_per_subject` /
`dbp_drift_corr_per_subject` in `statistic.json`, no dedicated plot): mean
correlation between window order and BP is small on average (SBP +0.030,
DBP +0.020) -- no strong *systematic* dataset-wide drift -- but the
per-subject spread is wide (p25/p75 around -0.23/+0.29, min/max beyond
±0.90). A non-trivial fraction of individual subjects show a strong
directional BP trend over their stay in either direction, which the
calibration-based model (anchored to a single early reference window) does
not explicitly account for.

**PPG amplitude / `min_ppg_std` gate check** (`ppg_amplitude.png`): **0.0%**
of windows fall below the 0.005 `min_ppg_std` threshold in every split --
the amplitude gate added after the flatline case study in
[data-cleaning.md](data-cleaning.md) is confirmed working at full scale;
no near-flatline windows slipped through. The distribution itself is
bimodal (a broad hump from ~0.01-0.15, then a sharper peak around 0.15-0.2,
plus a smaller peak near 0.5-0.6) -- consistent with real pulsatile signal
at varying ADC gain settings across subjects (per-window z-score
normalization in `bpe/dataset.py` already accounts for this at training
time).

**Sample-rate consistency**: `distinct_fs_values` is exactly `[125.0]` in
every split -- no resampling artifacts, as expected since MIMIC-III's native
rate already matches the target rate.

## 5. Key Findings and Recommendations

1. **Attrition is milder than assumed.** The README/data-cleaning docs'
   "~90-95% window attrition" expectation (carried over from the source
   paper) does not hold at full scale: observed is 55.2% window-level and
   17.8% subject-level attrition. Worth updating that framing once this is
   corroborated, and reason to be somewhat more optimistic about dataset
   size than originally planned for.
2. **`overall_yield_pct` is currently miscomputed** (§2.6) -- it compares a
   ~2x-inflated windowed-duration figure against true elapsed time. Either
   drop it from `statistic.json`/the printed summary, or fix it (e.g. by
   dividing `total_retained_duration_hr` by the known overlap factor, or by
   computing both sides consistently in "windows attempted" units). Flagging
   here rather than silently patching, since the fix choice affects the
   metric's exact definition.
3. **Consider a `DBP < SBP` guard.** A small number of windows in every
   split have pulse pressure at or slightly below zero. `physiological_range_ok`
   currently checks each of SBP/DBP against independent bounds but never
   their relative order; whether this is worth an explicit QC filter
   depends on how much it affects a trained model (likely negligible given
   how few windows are affected, but worth a threshold check with
   `dataset-statistic --skip-ppg-amplitude --no-plots` after any QC change).
4. **Windows-per-subject concentration is severe** (up to 40x max/median,
   top 10% of subjects holding ~46% of windows). If per-window random
   sampling is used during training as-is, a handful of subjects will
   dominate gradient updates; a per-subject-weighted sampler is worth
   considering if training curves show this causing instability.
5. **The calibration reference window is not always representative.**
   Most subjects' calibration BP sits within ~15 mmHg of their own mean, but
   the tails reach ±35 mmHg. Combined with the drift-correlation finding
   (some subjects trend strongly over time), a future iteration could
   explore alternative calibration-window selection (e.g. a window nearer
   the subject's median BP, rather than strictly the first surviving one)
   as an ablation against the current Siamese model design.
