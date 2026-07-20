# bpe-mimic3 (Blood Pressure Estimation from PPG using MIMIC-III)

A PyTorch project that estimates arterial blood pressure (SBP / DBP) from
photoplethysmography (PPG) waveforms, trained on the **MIMIC-III Waveform
Database Matched Subset**. It reproduces the calibration-free and
calibration-based (Siamese network) CNN techniques from Schlesinger,
Vigderhouse, Eytan & Moshe (2020), *"Blood Pressure Estimation From PPG
Signals Using Convolutional Neural Networks And Siamese Network"* — see
[docs/method.md](docs/method.md) for the full methodology summary this
project is built from, and [docs/development-plan.md](docs/development-plan.md)
for the concrete implementation plan and current status.

> **Project status**: methodology and pipeline design are complete
> (this README + the development plan); the data pipeline, models, and
> training/evaluation code have not been implemented yet. See
> [docs/development-plan.md](docs/development-plan.md) for the phased build
> order.

## Project Goal

Take a short PPG waveform segment and predict continuous blood pressure
(SBP / DBP) without requiring an invasive arterial line at inference time —
in both an unpersonalized (**calibration-free**) mode and a personalized
(**calibration-based**) mode that needs a single PPG/BP reference reading
per patient.

```text
PPG waveform (100 Hz, 8 s)  ──►  [ CNN ]                            ──►  SBP / DBP (mmHg) (calibration-free)
PPG waveform (100 Hz, 8 s)  ──►  [ Siamese CNN vs. calib. PPG/BP ]  ──►  SBP / DBP (mmHg) (calibration-based)
```

This differs from the source paper (and from an earlier, related project
that used VitalDB) in three deliberate ways:

| Aspect         | Source paper (method.md) | This project                                                                                                                                         |
| -------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sample rate    | 125 Hz                   | **100 Hz** — the target deployment PPG source samples at 100 Hz, so all signals are resampled 125→100 Hz during dataset construction.                |
| Segment length | 30 s                     | **8 s** — 30 s is too long for a real-time BP estimate; artifact-filter thresholds are re-derived for the shorter window (see the development plan). |
| Dataset        | MIMIC-II, 125 Hz         | **MIMIC-III Waveform Database Matched Subset**, processed in full and downsampled to 100 Hz.                                                         |

## Dataset — MIMIC-III Waveform Database (Matched Subset)

The raw dataset lives at `data/mimic3` (a **read-only** symlink into a local
copy of `mimic3wdb-matched/1.0`). It must never be modified — every
preprocessing script only reads from it and writes derived output elsewhere
(`data/dataset`).

| Item                | Detail                                                     |
| ------------------- | ---------------------------------------------------------- |
| Subjects            | 10,282 (shards `p00`…`p09`)                                |
| Waveform records    | 22,317 (many subjects have multiple ICU stays/records)     |
| Format              | WFDB (`.hea` + `.dat`), often **multi-segment** per record |
| Native sample rate  | 125 Hz for all waveform channels                           |
| PPG channel         | `PLETH`                                                    |
| Arterial BP channel | `ABP` or `ART` (naming varies by record)                   |
| Clinical metadata   | None bundled locally — only waveform data is present       |

Not every record contains PPG, and not every PPG-containing record also has
an arterial line. Local sampling of the first 300 subjects found PLETH in
~30 % of records, but PLETH **and** an arterial pressure channel together in
only ~7 %. Signal availability must be checked **per segment** (a record's
layout header can declare a signal that a given segment doesn't actually
carry) — this is handled by an indexing pass before any heavy processing;
see [docs/development-plan.md](docs/development-plan.md) §2 and §4.

## Methodology Summary

Full detail in [docs/method.md](docs/method.md); summarized here as it
applies to this project.

### Preprocessing

1. Index the matched subset for records that expose both PPG and an
   arterial pressure channel in the same segment.
2. Resample PPG and ABP 125 Hz → 100 Hz.
3. Slice into 8 s windows and label each window's SBP/DBP from the ABP
   window's peak/trough statistics.
4. Reject physiologically implausible windows (SBP outside `[75, 165]`
   mmHg, DBP outside `[40, 85]` mmHg).
5. Reject low-periodicity (noisy) windows via autocorrelation thresholding
   on both PPG and ABP, and reject flatline/disconnected-sensor PPG windows
   via a minimum-amplitude check the periodicity test alone can miss (see
   [docs/data-cleaning.md](docs/data-cleaning.md)).
6. Drop patients with too few surviving windows or too high a rejection
   rate.
7. Drop per-patient outlier windows (BP more than ±40 mmHg from that
   patient's first valid window).
8. Keep each patient's first valid window as their **calibration pair**
   (PPG segment + true SBP/DBP) for the calibration-based model.
9. Split by **patient** (not by window) into train/60% val/20% test/20% to
   prevent leakage.

Expect heavy attrition — on the order of 90 % of raw segments are expected
to be discarded by this pipeline, similar in spirit to the ~95 % attrition
reported in the source paper. The processed, PyTorch-ready result is written
to `data/dataset/{train,val,test}/{subject_id}.npz`; all model training and
evaluation reads only from `data/dataset`, never from `data/mimic3`.

### Models

- **Calibration-free**: an AlexNet-inspired 1D/spectrogram CNN (5 conv + 3
  FC layers, batch norm, dropout) that predicts SBP/DBP directly from a
  PPG window's spectrogram. Trained with L1 loss / Adam.
- **Calibration-based (Siamese)**: two weight-sharing copies of the same
  CNN backbone — one processes the current PPG window, the other the
  patient's stored calibration window. Their feature vectors are
  **subtracted** (signed, preserving direction of change) and regressed to
  `ΔBP = current_BP − calibration_BP`. Final estimate is
  `calibration_BP + predicted ΔBP`.

Both modes share the same CNN backbone and are registered in a small model
registry so additional calibration-free architectures can be added later
without changing the training/evaluation pipeline.

## Repository Layout

The full script/CLI structure below is the target layout (see
[docs/development-plan.md](docs/development-plan.md) §3 for the phased
build order); items are added as each phase is implemented.

```text
bpe-mimic3/
├── bin/                              # Windows .bat + POSIX sh launchers
│   ├── build-mimic3-index[.bat]      # scan data/mimic3 → index CSV
│   ├── construct-dataset[.bat]       # build data/dataset (100 Hz, 8 s, QC)
│   ├── mimic3-browser[.bat]          # GUI raw WFDB waveform browser
│   ├── dataset-browser[.bat]         # GUI: waveform + spectrogram + PSD; also browses in-progress (unsplit) data
│   ├── dataset-statistic[.bat]       # split/QC-retention statistics
│   ├── check-cuda.bat
│   ├── print-model[.bat] / print-all-model[.bat]
│   ├── train-model[.bat] / train-all-model[.bat]
│   ├── eval-model[.bat] / eval-all-model[.bat]             # calibration-free
│   ├── eval-calib-model[.bat] / eval-all-calib-model[.bat] # calibration-based
│   ├── generate-train-status[.bat] / generate-all-train-status[.bat]
│   ├── collect-result[.bat] / summarize-result[.bat] / generate-overview[.bat]
│   └── share-data[.bat] / download-shared-data[.bat] # share data/dataset
├── scripts/                          # one .py per bin/ entry above
├── bpe/                              # package
│   ├── io/                           # WFDB record/segment reading helpers
│   ├── preprocess/                   # resample, window, QC, peak-based labeling
│   ├── features/                     # spectrogram (STFT) + PSD computation
│   ├── models/                       # calibration-free CNN, Siamese, registry
│   ├── dataset.py                     # PyTorch Dataset/DataLoader (+ calib pairing)
│   ├── trainer.py
│   └── metrics.py                     # MAE/RMSE/ME/SD, BHS grade, AAMI pass/fail
├── docs/
│   ├── method.md                      # source methodology
│   ├── development-plan.md            # implementation plan & status
│   ├── data-cleaning.md               # implementation-level QC pipeline detail
│   └── evaluation-result*.md          # written once models are evaluated
├── data/                              # git-ignored, local only
│   ├── mimic3/                        # read-only symlink — DO NOT MODIFY
│   ├── mimic3_index.csv               # output of build-mimic3-index
│   ├── dataset/                       # output of construct-dataset
│   └── models/                        # training checkpoints & metrics
├── AGENTS.md                          # contribution rules for AI agents
├── pyproject.toml                     # uv project configuration
└── README.md
```

## Environment

| Tool             | Version                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Python           | ≥ 3.13                                                                                                                   |
| Package manager  | [uv](https://docs.astral.sh/uv/)                                                                                         |
| Key dependencies | `wfdb`, `torch`, `numpy`, `scipy`, `pandas`, `matplotlib` (to be added to `pyproject.toml` as each phase is implemented) |

```bash
# requires uv (https://docs.astral.sh/uv/)
uv sync
```

> **Do not use `pip install`.** All dependency management must go through
> `uv`. Run scripts with `uv run python <script>`. See [AGENTS.md](AGENTS.md).

## Experiment Results

Not available yet — no model has been trained. Once the pipeline in
[docs/development-plan.md](docs/development-plan.md) is implemented, results
(MAE / RMSE / ME / SD, BHS grade, AAMI pass/fail for both calibration-free
and calibration-based models) will be documented here and in
`docs/evaluation-result.md`.

## References

- Schlesinger, O., Vigderhouse, N., Eytan, D., and Moshe, Y. (2020). "Blood
  Pressure Estimation From PPG Signals Using Convolutional Neural Networks
  And Siamese Network."
- Johnson, A., Pollard, T., and Mark, R. (2016). "MIMIC-III Clinical
  Database" / "MIMIC-III Waveform Database Matched Subset."
  PhysioNet. DOI: 10.13026/c2294b
