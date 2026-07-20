# Development Plan — BP Estimation from PPG using MIMIC-III

This document defines the implementation plan for `bpe-mimic3`. It translates
the methodology in [method.md](method.md) (Schlesinger et al., 2020) into a
concrete pipeline built on the **MIMIC-III Waveform Database Matched Subset**
available locally at `data/mimic3`.

## 1. Decisions Already Made

These were confirmed with the project owner and are treated as fixed
constraints for the rest of this plan:

| Topic                   | Decision                                                                                                                                                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Calibration-based model | Reproduce the **Siamese network** from method.md as-is (twin CNN feature extractors, feature-vector subtraction, ΔBP regression head). No bias-correction/OSU alternative for now.                                             |
| Calibration-free model  | Implement **one** CNN first — the AlexNet-inspired architecture from method.md. The model layer is designed as a registry from the start so more architectures can be added later without breaking the training/eval pipeline. |
| Segment length          | **8 s**, not the paper's 30 s. Rationale: 30 s is too long for a real-time BP estimate. This changes downstream STFT sub-window sizing and the artifact-filter thresholds (see §4).                                            |
| Target sample rate      | **100 Hz**, not the native 125 Hz of MIMIC-III waveforms. All PPG/ABP signals are resampled 125→100 Hz before windowing.                                                                                                       |
| Dataset scope           | Process the **entire** `mimic3wdb-matched/1.0` subset from the start (not a capped pilot). Expect roughly 10 % of raw segments to survive QC, similar in spirit to the ~5 % retention reported in method.md.                   |
| Data flow               | `data/mimic3` (read-only) → cleaning/labeling pipeline → `data/dataset` (PyTorch-ready). All training/evaluation code reads only from `data/dataset`; nothing ever writes to `data/mimic3`.                                    |
| Repository shape        | Build the full script/CLI structure up front, at parity with the previous `bpe-vitaldb` project (see §3), rather than growing it incrementally.                                                                                |
| Calibration-window reuse | A patient's calibration window is also used as a normal training sample for the calibration-free model (no held-out anchor point).                                                                                            |
| Spectrogram sub-window   | Fixed at **1 s** (Hamming window, 95 % overlap) for the 8 s segment, not derived proportionally from the paper's 6 s/30 s ratio.                                                                                              |
| Multi-segment records    | Processed **segment-by-segment only** — windows never span a segment boundary, since segments can have different active signal sets and inter-segment gaps aren't guaranteed to be zero.                                     |

## 2. Source Data Characteristics (verified locally)

`data/mimic3` is a symlink to `mimic3wdb-matched/1.0`:

- **10,282 subjects** (`p00`…`p09` shards), **22,317 waveform records** total
  (`RECORDS`, `RECORDS-waveforms`).
- All records are natively **125 Hz** (WFDB format: `.hea` + `.dat`).
- Most records are **multi-segment**: a master header (e.g.
  `p000020-2183-04-28-17-47.hea`) references a `*_layout.hea` (declares the
  full signal set for the stay) plus a sequence of numbered segment records
  (`3544749_0001.hea` … `_000N.hea`), each of which may expose only a
  **subset** of the layout's signals (e.g. one segment has `II, AVF, ABP, PAP`
  with no `PLETH`, while a sibling patient's segment has `PLETH` but no
  `ABP`). Signal availability must be checked **per segment**, not just at
  the layout/record level.
- Not every record has PPG. Empirical sampling of the first 300 subjects
  under `p00` found PLETH present in ~30 % of records, but PLETH **and** ABP
  together in only ~7 %. The full-dataset equivalent is expected to yield on
  the order of several hundred to ~1,000 usable subjects before the
  window-level QC in method.md is even applied — consistent with the
  question 3 answer that heavy attrition is expected and accepted.
- No MIMIC-III **clinical** tables (admissions, demographics) are present
  under `data/mimic3` — only the waveform matched subset. This is actually
  consistent with method.md, which never uses demographic features (that is
  the whole point of the Siamese calibration design).
- `data/mimic3` must remain **read-only**: no script may write, rename, or
  delete anything under it.

## 3. Target Repository Layout

Mirrors `bpe-vitaldb`'s structure, adapted for MIMIC-III/WFDB and the 100 Hz /
8 s / dual-mode design:

```text
bpe-mimic3/
├── bin/                              # Windows .bat + POSIX sh launchers
│   ├── build-mimic3-index[.bat]      # scan data/mimic3 → index CSV
│   ├── construct-dataset[.bat]       # build data/dataset (100 Hz, 8 s, QC)
│   ├── mimic3-browser[.bat]          # GUI raw WFDB waveform browser
│   ├── dataset-browser[.bat]         # GUI: waveform + spectrogram + PSD over data/dataset
│   ├── dataset-statistic[.bat]       # split/QC-retention statistics
│   ├── check-cuda.bat
│   ├── print-model[.bat] / print-all-model[.bat]
│   ├── train-model[.bat] / train-all-model[.bat]
│   ├── eval-model[.bat] / eval-all-model[.bat]           # calibration-free
│   ├── eval-calib-model[.bat] / eval-all-calib-model[.bat] # Siamese
│   ├── generate-train-status[.bat] / generate-all-train-status[.bat]
│   ├── collect-result[.bat] / summarize-result[.bat] / generate-overview[.bat]
│   └── share-data[.bat] / download-shared-data[.bat] # share data/dataset (not data/mimic3)
├── scripts/                          # one .py per bin/ entry above
├── bpe/                              # package
│   ├── io/                           # WFDB record/segment reading helpers
│   ├── preprocess/                  # resample, window, QC, peak-based labeling
│   ├── features/                    # spectrogram (STFT) + PSD computation
│   ├── models/                      # calibration-free CNN, Siamese, registry
│   ├── dataset.py                    # PyTorch Dataset/DataLoader (+ calib pairing)
│   ├── trainer.py
│   └── metrics.py                    # MAE/RMSE/ME/SD, BHS grade, AAMI pass/fail
├── docs/
│   ├── method.md                     # source methodology (existing)
│   ├── development-plan.md           # this document
│   └── evaluation-result*.md         # written once models are evaluated
├── data/                             # git-ignored
│   ├── mimic3/                       # read-only symlink, DO NOT MODIFY
│   ├── mimic3_index.csv              # output of build-mimic3-index
│   ├── dataset/                      # output of construct-dataset (train/val/test)
│   └── models/                       # training checkpoints & metrics
├── AGENTS.md
├── pyproject.toml
└── README.md
```

## 4. Preprocessing Pipeline (method.md, adapted to 8 s / 100 Hz)

Applied per candidate record, then aggregated per patient:

1. **Indexing** (`build-mimic3-index`): walk `RECORDS-waveforms`, open each
   record's segments with `wfdb`, and record which segments expose **both**
   `PLETH` and an arterial pressure channel (`ABP` or `ART`) at the WFDB
   level, along with segment length and start time. Output: a CSV under
   `data/` (never under `data/mimic3`) that all later steps read instead of
   re-scanning the raw files. This is the pruning step that keeps the
   full-dataset pass tractable.
2. **Resampling**: PPG and ABP channels are resampled 125 Hz → 100 Hz
   (`scipy.signal.resample_poly`, `up=4, down=5`, an exact ratio since
   125 = 25·5 and 100 = 25·4 — no fractional-rate approximation needed).
3. **Windowing**: signals are cut into **8 s** windows (800 samples at
   100 Hz). Stride/overlap defaults to the 50 % convention used previously
   (4 s stride) but is a tunable CLI parameter.
4. **Per-window SBP/DBP labeling**: from the ABP window, detect peaks/troughs
   and take the mean of the max/min peaks as SBP/DBP, exactly as in
   method.md §1.
5. **Physiological range filter**: reject windows with SBP outside
   `[75, 165]` mmHg or DBP outside `[40, 85]` mmHg (unchanged from method.md —
   these bounds are about plausible BP values, not window length).
6. **Autocorrelation-based quality filter**: reject windows where the
   (DC-removed, normalized) autocorrelation of PPG or ABP falls below a
   periodicity threshold. The threshold is **re-tuned empirically** for 8 s
   windows (a shorter window naturally yields fewer autocorrelation lags to
   integrate over) rather than reusing the paper's 30 s-calibrated constant;
   `dataset-statistic` reports retention rate vs. threshold to support this
   tuning.
7. **Patient-level exclusion**: drop a patient if fewer than *N* valid
   windows remain, or more than 95 % of their windows were rejected. The
   paper used `N = 100` for 30 s windows (~50 min of usable signal); scaled
   to 8 s windows at equivalent usable duration this becomes roughly
   `N ≈ 375`, but this is a configurable default to validate against actual
   yield, not a hard requirement.
8. **Outlier removal**: for each patient, drop windows whose SBP/DBP
   deviates more than ±40 mmHg from their first valid window's BP (unchanged
   from method.md — this is a physiological-plausibility bound, not
   window-length-dependent).
9. **Calibration reference**: each patient's chronologically-first surviving
   window is retained as the **calibration pair** (`calib_x` = PPG segment,
   `calib_y` = [SBP, DBP]) used by the Siamese model in §5. This window is
   also kept in the regular `(x, y)` pool, so the calibration-free model
   trains on it like any other window.
10. **Patient-level split**: 60/20/20 train/val/test, split by patient (not
    by window) to prevent leakage, matching method.md and `bpe-vitaldb`.
11. **Output format**: one `.npz` per patient under
    `data/dataset/{train,val,test}/{subject_id}.npz` containing:

    ```text
    x         float32  (N, 800)   PPG windows (8 s @ 100 Hz)
    y         float32  (N, 2)     [SBP, DBP] mmHg per window
    calib_x   float32  (800,)     calibration-window PPG
    calib_y   float32  (2,)       calibration-window [SBP, DBP]
    fs        float32  scalar     sample rate the windows were built at (target_fs)
    ```

    The calibration-free CNN trains directly on `(x, y)`. The Siamese model
    additionally consumes `(calib_x, calib_y)` per patient to compute
    `y - calib_y` as its regression target.

## 5. Model Architectures

### 5.1 Calibration-free CNN (method.md §2)

- Input: spectrogram of an 8 s PPG window (STFT, **1 s** Hamming sub-window,
  95 % overlap).
- 5 conv layers + 3 FC layers, AlexNet-inspired, max-pooling after conv
  1/2/5, batch norm after every conv layer, dropout before the first two FC
  layers, ReLU throughout, final FC → linear regression head (SBP, DBP).
- Loss: L1 (MAE). Optimizer: Adam, batch size 32 (paper default, tunable).

### 5.2 Siamese calibration-based network (method.md §3)

- Two weight-sharing copies of the CNN in §5.1, each ending in a feature
  vector instead of a direct BP regression.
- One branch takes the current window's spectrogram, the other takes the
  patient's calibration-window spectrogram.
- Feature vectors are **subtracted** (signed, not an absolute-value/Euclidean
  distance) so the model can express the direction of BP change.
- The difference vector → ReLU → linear regression head → predicted
  `ΔBP = current_BP − calib_BP`. Final BP = `calib_y + predicted_delta`.
- Same optimizer/hyperparameters as §5.1.

Both live under `bpe/models/` behind a small registry (name → constructor),
so `train-model.py --model <name>` works the same way it did in
`bpe-vitaldb`, and adding more calibration-free architectures later is a
pure addition, not a refactor.

## 6. Phased Execution Order

1. **Environment**: add real dependencies to `pyproject.toml`
   (`wfdb`, `numpy`, `scipy`, `torch`, `pandas`, `matplotlib`, `tqdm`) and
   `uv sync`.
2. **Indexing**: `build-mimic3-index` over the full matched subset →
   `data/mimic3_index.csv`. This is the first real validation that the
   PLETH+ABP co-occurrence assumption holds at full scale.
3. **Preprocessing pipeline** (`bpe/preprocess/`): implement resampling,
   windowing, labeling, and each QC filter as independently testable units,
   since the thresholds in §4 need empirical tuning.
4. **Dataset construction** (`construct-dataset`): run the full pipeline
   end-to-end over the indexed records → `data/dataset/{train,val,test}`.
   Validate with `dataset-statistic` (retention rate, SBP/DBP distributions,
   per-split patient/window counts) before trusting the output for training.
5. **Dataset inspection tooling**: a single `dataset-browser` GUI (split /
   subject / window list on the left, stacked waveform + spectrogram + PSD
   plots on the right) — needed to sanity-check real segments and spectra
   before committing to spectrogram hyperparameters (sub-window length,
   overlap).
6. **Model implementation**: calibration-free CNN, then the Siamese
   wrapper reusing it as the twin backbone.
7. **Training pipeline**: `train-model` for the calibration-free CNN first
   (simpler data flow), then extend the dataset loader / trainer to support
   calibration pairs for the Siamese model.
8. **Evaluation pipeline**: `eval-model` (calibration-free: MAE, RMSE, ME,
   SD, BHS grade, AAMI pass/fail) and `eval-calib-model` (Siamese, evaluated
   per patient using their stored calibration pair).
9. **Reporting tooling**: `collect-result`, `summarize-result`,
   `generate-overview`, `generate-train-status` — ported from
   `bpe-vitaldb` with minimal changes, since result-file shapes are the
   same.
10. **Write-up**: once real numbers exist, add
    `docs/evaluation-result.md` and update README's results section
    (mirroring the `bpe-vitaldb` README's "Experiment Results" section).

Each phase's script should be runnable and independently verifiable
(`uv run python scripts/<name>.py --help` at minimum) before moving to the
next; per AGENTS.md, no phase should be marked done without an actual run
against real data as evidence.

## 7. Open Questions / Assumptions to Revisit

These are called out explicitly (per AGENTS.md) as assumptions made to keep
this plan concrete, not as settled decisions:

- **QC threshold retuning**: the autocorrelation threshold and per-patient
  minimum-window count in §4 are seeded from the paper's 30 s-window values
  but scaled/guessed for 8 s windows. They need empirical validation via
  `dataset-statistic` once real data flows through the pipeline, and may
  need another pass of tuning.
- **ABP vs ART channel naming**: some records label the arterial waveform
  `ABP`, others `ART` (seen in local sampling). The indexer must treat both
  as the ground-truth arterial pressure channel.
- **NaN handling before resampling**: `construct-dataset` resamples each
  segment once, whole, before windowing. If a segment has NaN gaps (sensor
  dropouts), `resample_poly`'s FIR filter can smear a NaN across nearby
  output samples, so a window can be contaminated slightly beyond the raw
  gap's width. The per-window NaN check still catches and drops these
  windows, so this is a yield cost, not a correctness bug, but if
  `dataset-statistic` shows it costing an outsized fraction of otherwise-good
  data, consider masking/interpolating gaps before resampling instead.
