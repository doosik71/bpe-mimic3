# Data Cleaning Method

This document describes, precisely, what `bpe/preprocess/` actually does to
turn raw MIMIC-III waveform segments into the windows stored in
`data/dataset/`. It is the implementation-level companion to
[method-spectrogram-cnn.md](method-spectrogram-cnn.md) (the source methodology) and
[development-plan.md](development-plan.md) §4 (the design rationale); this
document tracks the code as built, including gaps discovered after the fact.

All of this runs in `bpe/preprocess/pipeline.py:process_patient`, called once
per subject by `convert_dataset` (see [development-plan.md](development-plan.md)
§4 step 10 for the resumable two-phase construction process this sits inside).

## Pipeline, in the order it's applied

### 1. Chronological ordering across records

A subject's segments (possibly spanning multiple ICU stays/records) are
sorted by admission timestamp (parsed from the MIMIC-III record name, e.g.
`p000109-2142-01-14-18-53`) plus in-record sample offset, so "the patient's
first window" means first in real time, not first in whatever order the
index happened to list records.

### 2. Per-segment: read, resample, window

For each segment: read the `PLETH` and arterial-pressure (`ABP`/`ART`)
channels via `wfdb`, resample both to the target rate (a no-op in practice,
since MIMIC-III's native 125 Hz already matches -- see `resample_signal`),
then slice into 8 s windows with 4 s stride (`window_signal`). PPG and ABP
windows are index-aligned (same window `i` covers the same time span in
both channels).

### 3. Per-window filters

Applied to every window, in this order; a window is dropped (not counted
toward `n_valid`) the moment it fails any check:

| #   | Check                             | Function                                  | Rationale                                                                                                                     |
| --- | --------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 3.1 | No NaN in either channel          | inline (`np.isnan(...).any()`)            | WFDB encodes sensor gaps/invalid samples as NaN.                                                                              |
| 3.2 | PPG has enough amplitude          | `has_sufficient_amplitude` (`quality.py`) | See "Flatline PPG" below — a disconnected/malfunctioning sensor can output a near-constant reading.                           |
| 3.3 | SBP/DBP can be labeled            | `compute_sbp_dbp` (`labels.py`)           | Peak/trough detection on the ABP window; returns `None` (drop) if fewer than 3 systolic peaks or diastolic troughs are found. |
| 3.4 | SBP/DBP physiologically plausible | `physiological_range_ok` (`quality.py`)   | SBP ∈ `[75, 165]` mmHg, DBP ∈ `[40, 85]` mmHg (docs/method-spectrogram-cnn.md's bounds, unchanged by window length).          |
| 3.5 | PPG is periodic                   | `is_periodic` (`quality.py`)              | Autocorrelation-based; a real pulse signal stays correlated across many lags, noise decays quickly.                           |
| 3.6 | ABP is periodic                   | `is_periodic` (`quality.py`)              | Same check, applied to the arterial pressure window.                                                                          |

A window that survives all six becomes one row of the patient's
`(x, y)` pool.

### 4. Per-patient filters

Applied once, after every segment's windows have been filtered:

- **Patient-level exclusion** (`should_exclude_patient`, `patient.py`): drop
  the entire patient if fewer than `min_valid_windows` (default 375)
  windows survived step 3, or if more than `max_reject_fraction` (default
  95%) of all attempted windows were rejected.
- **Outlier removal** (`outlier_keep_mask`, `patient.py`): using the
  patient's chronologically-first surviving window as a reference, drop any
  later window whose SBP or DBP deviates more than `max_bp_deviation`
  (default 40 mmHg) from it.
- **Calibration window** (`calibration_index`, `patient.py`): the
  chronologically-first surviving window (index 0 after outlier removal,
  since a window can't deviate from itself) becomes the patient's
  calibration pair, stored in the output npz as `calib_x`/`calib_y`. It is
  *also* kept in the regular `(x, y)` pool — see
  [development-plan.md](development-plan.md) §1's calibration-window-reuse
  decision.

### 5. Output

One npz per surviving patient: `x` (N, 1000) PPG windows, `y` (N, 2)
`[SBP, DBP]`, `calib_x` (1000,), `calib_y` (2,), `fs` (scalar, 125.0). See
[development-plan.md](development-plan.md) §4 step 11 for the full schema
and where these files land (`data/dataset/{train,val,test}/` after
`finalize_split`, flat under `data/dataset/` before it).

## Parameters and their status

| Parameter                                                | Default                    | Status                                                                                                                                                                                                                            |
| -------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sbp_range`, `dbp_range`                                 | `[75,165]`, `[40,85]` mmHg | From method-spectrogram-cnn.md, not window-length-dependent.                                                                                                                                                                      |
| `max_bp_deviation`                                       | 40 mmHg                    | From method-spectrogram-cnn.md, not window-length-dependent.                                                                                                                                                                      |
| `min_ppg_std`                                            | 0.005                      | **Empirically derived** (see case study below) from the observed bimodal std distribution across the first ~100 converted subjects. Worth re-checking at full-dataset scale.                                                      |
| `ppg_periodicity_threshold`, `abp_periodicity_threshold` | 0.05, 0.05                 | **Unvalidated placeholders**, carried over from the 30 s-window paper without being re-tuned for 8 s windows. `periodicity_score` integrates over one lag per sample, so the 1000-sample change shifts its typical magnitude too. |
| `min_valid_windows`                                      | 375                        | **Estimated**, scaled from the paper's 30 s-window value (100) to 8 s windows by duration. Not yet validated against real yield.                                                                                                  |

The two "unvalidated" rows are exactly what `dataset-statistic` (see
[scripts/dataset-statistic.py](../scripts/dataset-statistic.py)) is meant to
help tune, by reporting retention rate and window counts at the currently
configured thresholds. Note it reports *current* retention, not retention
*as a function of* threshold (docs/dataset-analysis.md #12) -- that would
require re-running the per-window QC filters against raw signals under
alternative threshold values, which the tool doesn't do yet.

## Case study: flatline PPG passing the periodicity filter

While inspecting `p001049` in `dataset-browser`, window 0's PPG waveform was
visibly not a real pulse signal — it looked like a flat line. Checking it
directly:

```text
mean=0.5005  std=0.000120  ptp=0.000321
periodicity_score = 0.1381        (threshold was 0.05 -- passed)
```

The signal is a near-constant reading (consistent with a disconnected or
malfunctioning PPG sensor) with only tiny ADC quantization jitter. That
jitter happened to repeat with a short period, so it registered as
"periodic". This is not a one-off: across this one subject, 2,661 of
51,971 windows (~5%) had the same signature, and the same failure mode was
present across the whole partially-converted dataset at the time (~90,000
of ~4.7M windows, i.e. roughly 2%, had `std < 0.001`).

**Root cause**: `periodicity_score` computes autocorrelation on a
DC-removed signal and normalizes it so the lag-0 value is 1 (see
`normalized_autocorrelation`). That normalization divides by the signal's
own variance, making the score scale-invariant by design — appropriate for
comparing periodicity shape, but it means an amplitude-collapsed signal is
graded purely on the shape of whatever residual noise it has, not on
whether it looks like a plausible physiological signal at all. Nothing else
in the step-3 filter chain checks amplitude either: `compute_sbp_dbp` and
`physiological_range_ok` only look at the *ABP* channel, so a dead PPG
sensor next to a perfectly normal arterial line sails through unnoticed.

**Fix**: added `has_sufficient_amplitude` (step 3.2 above) as an explicit,
scale-*sensitive* absolute-amplitude gate, independent of the periodicity
check. The threshold (0.005) was chosen from the data itself: plotting the
per-window std across all subjects converted so far showed a large cluster
sitting at `std ≈ 0.00012` (the dead-sensor population) separated by a
clean gap from real signal, which starts around `std ≈ 0.007` at the 2nd
percentile and climbs to ≈0.15 at the median. 0.005 sits in that gap.

**Consequence**: this changes what counts as a valid window, so it only
takes effect for subjects converted (or re-converted) after the fix.
Subjects already in `data/dataset/_progress.csv` from before this change
were converted without it and should be regenerated with
`construct-dataset --force` (see [development-plan.md](development-plan.md)
§4 step 10) to benefit from it.
