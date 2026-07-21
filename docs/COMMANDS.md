# Command Reference: Reproducing This Project's Experiments

This document collects every command needed to reproduce this project's
experiments end to end, grouped by execution stage: environment setup,
dataset construction, training, evaluation, and reporting. It also lists
the exact `--model` name for every architecture currently in the registry
(`bpe/models/registry.py`) so each one can be trained and evaluated
individually.

Every script below has a matching launcher in [bin/](../bin/)
(`bin/<name>` on Linux/macOS, `bin\<name>.bat` on Windows) that just
forwards to `uv run python scripts/<name>.py`; the commands here use the
`uv run` form directly since it works identically on every platform. Pass
`--help` to any script for the full flag list and its defaults.

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
uv run python scripts/build-mimic3-index.py
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
uv run python scripts/construct-dataset.py
```

Useful flags: `--limit-subjects N` (quick trial run), `--force`
(reprocess everyone, e.g. after changing QC thresholds), `--skip-split` /
`--split-only` (rerun only one of the two internal phases).

## 3. Inspect the Dataset (optional)

GUI browser for PPG/ABP waveforms, spectrograms, and PSDs — useful for
sanity-checking QC thresholds or browsing an in-progress (unsplit) build.

```bash
uv run python scripts/dataset-browser.py
```

## 4. Train Every Model

`--model` selects the architecture from the registry
(`bpe/models/registry.py`). Every calibration-free model shares the same
CLI (`scripts/train-model.py`); the calibration-based (Siamese) model
uses the exact same script with `--model spectro_siamese`. Common flags:
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
uv run python scripts/train-model.py --model spectro_cnn
uv run python scripts/train-model.py --model acfa
uv run python scripts/train-model.py --model ae_lstm
uv run python scripts/train-model.py --model bpnet_cf
uv run python scripts/train-model.py --model conv_reg
uv run python scripts/train-model.py --model mtae
uv run python scripts/train-model.py --model mtae_mlp
uv run python scripts/train-model.py --model pctn
uv run python scripts/train-model.py --model ppnet
uv run python scripts/train-model.py --model resnet1d13
uv run python scripts/train-model.py --model resnet1d21
uv run python scripts/train-model.py --model resnet1d37
uv run python scripts/train-model.py --model resnet1d61
uv run python scripts/train-model.py --model st_resnet
```

### 4.2 Calibration-based model (Siamese)

| `--model`         | Architecture                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- |
| `spectro_siamese` | Weight-shared twin backbone, regresses `ΔBP` vs. a stored calibration window (docs/method-spectrogram-cnn.md §3) |

```bash
uv run python scripts/train-model.py --model spectro_siamese
```

## 5. Evaluate Every Model

`eval-model.py` is for calibration-free models; `eval-calib-model.py` is
for calibration-based models (it evaluates using each patient's stored
calibration pair). Both take the run directory as a positional argument
and report MAE, RMSE, ME, SD, BHS grade, and AAMI pass/fail for SBP/DBP,
writing `eval_results.json`, `eval_plot.png`, and `error_hist.png` next to
the checkpoint. Common flags: `--split val|test`, `--checkpoint
<name>.pt`.

### 5.1 Calibration-free models

```bash
uv run python scripts/eval-model.py data/models/spectro_cnn
uv run python scripts/eval-model.py data/models/acfa
uv run python scripts/eval-model.py data/models/ae_lstm
uv run python scripts/eval-model.py data/models/bpnet_cf
uv run python scripts/eval-model.py data/models/conv_reg
uv run python scripts/eval-model.py data/models/mtae
uv run python scripts/eval-model.py data/models/mtae_mlp
uv run python scripts/eval-model.py data/models/pctn
uv run python scripts/eval-model.py data/models/ppnet
uv run python scripts/eval-model.py data/models/resnet1d13
uv run python scripts/eval-model.py data/models/resnet1d21
uv run python scripts/eval-model.py data/models/resnet1d37
uv run python scripts/eval-model.py data/models/resnet1d61
uv run python scripts/eval-model.py data/models/st_resnet
```

### 5.2 Calibration-based model (Siamese)

```bash
uv run python scripts/eval-calib-model.py data/models/spectro_siamese
```

## 6. Check Training Progress

Plots per-epoch loss/MAE curves from `metrics.csv` and prints a summary,
either for one run or every run under `data/models/`:

```bash
uv run python scripts/generate-train-status.py data/models/spectro_cnn
uv run python scripts/generate-all-train-status.py
```

## 7. Collect and Compare Results Across Models

Once several models are trained and evaluated, gather their results into
`data/results/` for easy comparison/sharing:

```bash
uv run python scripts/collect-result.py       # copies eval_results.json + plots into data/results/<model>/
uv run python scripts/summarize-result.py     # writes data/results/summary.csv, one row per model
uv run python scripts/generate-overview.py    # writes overview_mae.png / overview_rmse.png (params vs. accuracy)
```
