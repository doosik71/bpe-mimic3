# Command Reference: Reproducing This Project's Experiments

This document collects every command needed to reproduce this project's
experiments end to end, grouped by execution stage: environment setup,
dataset construction, training, evaluation, and reporting. It also lists
the exact `--model` name for every architecture currently in the registry
(`bpe/models/registry.py`) so each one can be trained and evaluated
individually.

Every command below is run through the project's `run` launcher at the
repository root — `run <command> [options]` on Linux/macOS, the same
`run <command> [options]` (resolving to `run.bat`) on Windows. It simply
forwards to `uv run python scripts/<command>.py [options]`, so the two forms
are equivalent and the trailing `.py` is optional (`run eval-model` and
`run eval-model.py` are the same). Run `run` with no arguments to list every
available command. Pass `--help` to any command for the full flag list and
its defaults.

See the [README](../README.md#quick-start-end-to-end-pipeline) for a
narrative walkthrough of the same pipeline; this document is the
exhaustive reference, in particular the full per-model train/eval command
list in §4 and §5.

## 0. Environment Setup

```bash
# requires uv (https://docs.astral.sh/uv/)
uv sync
```

Do not use `pip install` — see [AGENTS.md](../AGENTS.md).

## 1. Build the MIMIC-III Index

Scans `data/mimic3` once for segments carrying both PPG (`PLETH`) and an
arterial BP channel, and writes `data/mimic3_index.csv`.

```bash
run build-mimic3-index
```

Useful flags: `--limit N` (scan only the first N records, for a quick
trial), `--workers N` (thread-pool size).

## 2. Construct the Dataset

Reads the index, then resamples/windows/labels/QC-filters every
qualifying segment and writes
`data/dataset/{train,val,test}/{subject_id}.npz`. This step is resumable
— re-running the same command after an interruption continues instead of
restarting (progress tracked in `data/dataset/_progress.csv`).

```bash
run construct-dataset
```

Useful flags: `--limit-subjects N` (quick trial run), `--force`
(reprocess everyone, e.g. after changing QC thresholds), `--skip-split` /
`--split-only` (rerun only one of the two internal phases).

## 3. Inspect the Dataset (optional)

GUI browser for PPG/ABP waveforms, spectrograms, and PSDs — useful for
sanity-checking QC thresholds or browsing an in-progress (unsplit) build.

```bash
run dataset-browser
```

## 4. Train Every Model

`--model` selects the architecture from the registry
(`bpe/models/registry.py`). Every calibration-free model shares the same
CLI (`run train-model`); the calibration-based (Siamese) model
uses the exact same command with `--model spectro_siamese`. Common flags:
`--epochs`, `--batch-size`, `--lr`, `--patience`, `--device
auto|cpu|cuda`, `--resume <checkpoint.pt>`.

### 4.1 Calibration-free models

| `--model`     | Architecture                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------- |
| `spectro_cnn` | AlexNet-inspired spectrogram CNN -- this project's baseline (docs/method-spectrogram-cnn.md) |
| `acfa`        | Adaptive Cross-domain Fusion Architecture (DyCASNet + xLSTM + Transformer + FKAN)            |
| `ae_lstm`     | Autoencoder-LSTM (encoder/decoder LSTM) **+ reconstruction loss**                            |
| `bpnet_cf`    | Calibration-free dual-scale depthwise-separable CNN + self-attention                         |
| `conv_reg`    | Simple 6-stage 1D CNN regression baseline                                                    |
| `mtae`        | Multi-Task AutoEncoder, shared linear BP head **+ reconstruction loss**                      |
| `mtae_mlp`    | Multi-Task AutoEncoder, separate SBP/DBP MLP heads **+ reconstruction loss**                 |
| `pctn`        | Parallel CNN-Transformer Network (ResNet + Transformer branches, CBAM fusion)                |
| `ppnet`       | PP-Net: CNN-LSTM (LRCN)                                                                      |
| `resnet1d13`  | 1D ResNet, 13 layers (1 stage, 1 block)                                                      |
| `resnet1d21`  | 1D ResNet, 21 layers (2 stages, 1 block each)                                                |
| `resnet1d37`  | 1D ResNet, 37 layers (4 stages, 1 block each)                                                |
| `resnet1d61`  | 1D ResNet, 61 layers (4 stages, 2 blocks each)                                               |
| `st_resnet`   | Spectro-temporal ResNet over derived PPG/VPG/APG channels                                    |

Models marked **+ reconstruction loss** define a `compute_loss` method
(see `bpe/models/mtae.py`) that `bpe/trainer.py` calls automatically when
present, mixing in a PPG-reconstruction auxiliary loss alongside the
SBP/DBP regression loss — no extra flags needed, it's on by default for
those three.

```bash
run train-model --model spectro_cnn
run train-model --model acfa
run train-model --model ae_lstm
run train-model --model bpnet_cf
run train-model --model conv_reg
run train-model --model mtae
run train-model --model mtae_mlp
run train-model --model pctn
run train-model --model ppnet
run train-model --model resnet1d13
run train-model --model resnet1d21
run train-model --model resnet1d37
run train-model --model resnet1d61
run train-model --model st_resnet
```

### 4.2 Calibration-based model (Siamese)

| `--model`         | Architecture                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `spectro_siamese` | Weight-shared twin backbone, regresses `ΔBP` vs. a stored calibration window (docs/method-spectrogram-cnn.md §3) |

```bash
run train-model --model spectro_siamese
```

## 5. Evaluate Every Model

`run eval-model` is for calibration-free models; `run eval-calib-model` is
for calibration-based models (it evaluates using each patient's stored
calibration pair). Both take the run directory as a positional argument
and report MAE, RMSE, ME, SD, BHS grade, and AAMI pass/fail for SBP/DBP,
writing `eval_results.json`, `eval_plot.png`, and `error_hist.png` next to
the checkpoint. Common flags: `--split val|test`, `--checkpoint
<name>.pt`.

### 5.1 Calibration-free models

```bash
run eval-model data/models/spectro_cnn
run eval-model data/models/acfa
run eval-model data/models/ae_lstm
run eval-model data/models/bpnet_cf
run eval-model data/models/conv_reg
run eval-model data/models/mtae
run eval-model data/models/mtae_mlp
run eval-model data/models/pctn
run eval-model data/models/ppnet
run eval-model data/models/resnet1d13
run eval-model data/models/resnet1d21
run eval-model data/models/resnet1d37
run eval-model data/models/resnet1d61
run eval-model data/models/st_resnet
```

### 5.2 Calibration-based model (Siamese)

```bash
run eval-calib-model data/models/spectro_siamese
```

## 6. Check Training Progress

Plots per-epoch loss/MAE curves from `metrics.csv` and prints a summary,
either for one run or every run under `data/models/`:

```bash
run generate-train-status data/models/spectro_cnn
run generate-all-train-status
```

## 7. Collect and Compare Results Across Models

Once several models are trained and evaluated, gather their results into
`data/results/` for easy comparison/sharing:

```bash
run collect-result       # copies eval_results.json + plots into data/results/<model>/
run summarize-result     # writes data/results/summary.csv, one row per model
run generate-overview    # writes overview_mae.png / overview_rmse.png (params vs. accuracy)
```
