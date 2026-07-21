# construct-dataset

Detailed design and user guide for the `construct-dataset` module
(`scripts/construct-dataset.py` → `bpe/preprocess/pipeline.py`), which turns
the raw MIMIC-III waveform segments listed in `data/mimic3_index.csv` into the
PyTorch-ready per-subject arrays under `data/dataset/`.

This is the implementation/usage companion to
[data-cleaning.md](data-cleaning.md) (the exact per-window/per-patient QC
rules) and [development-plan.md](development-plan.md) §4 (design rationale).
The consumer of the output is documented in [train-model.md](train-model.md).

## What it does

For every subject in the index it reads the `PLETH` (PPG) and arterial
pressure (`ABP`/`ART`) channels, resamples to the target rate (a no-op at
MIMIC-III's native 125 Hz), slices into 8 s windows, labels each window's
`[SBP, DBP]` from the ABP peaks/troughs, discards windows and patients that
fail quality control, keeps each patient's surviving window closest to their
own median SBP/DBP as a **calibration pair**, and finally splits patients
into train/val/test. See [data-cleaning.md](data-cleaning.md) for the
precise filter chain.

## Usage

```bash
uv run python scripts/construct-dataset.py
```

The step is the slowest in the pipeline and is **resumable**: each subject is
converted, written to disk, and recorded in `data/dataset/_progress.csv` as
soon as it finishes, so re-running the same command after an interruption
continues instead of restarting.

### Two phases

1. **Convert** (`convert_dataset`): process each subject and write its npz
   *flat* under `data/dataset/` (pre-split staging state).
2. **Split** (`finalize_split`): assign kept subjects to
   `data/dataset/{train,val,test}/` by patient (never by window, to prevent
   leakage) and move their npz into the split subdirectory.

### Key flags

| Flag                                                          | Default        | Purpose                                                                                                                           |
| ------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--output-dir`                                                | `data/dataset` | Where the npz files are written.                                                                                                  |
| `--limit-subjects N`                                          | (no limit)     | Process only the first N subjects, for a quick trial.                                                                             |
| `--force`                                                     | off            | Reprocess every subject even if already in `_progress.csv` (needed after changing QC parameters — the ledger doesn't track them). |
| `--skip-split`                                                | off            | Run only the conversion phase; leave npz flat/unsplit.                                                                            |
| `--split-only`                                                | off            | Skip conversion; only (re-)assign already-converted subjects into splits.                                                         |
| `--workers`                                                   | 8              | Process-pool size for concurrent subject conversion.                                                                              |
| `--target-fs`                                                 | 125.0          | Target sample rate in Hz.                                                                                                         |
| `--window-sec` / `--stride-sec`                               | 8.0 / 4.0      | Window length and stride in seconds.                                                                                              |
| `--sbp-range` / `--dbp-range`                                 | 75–165 / 40–85 | Physiologically plausible BP bounds (mmHg).                                                                                       |
| `--pulse-pressure-range`                                      | 20–100         | Plausible pulse-pressure (SBP-DBP) bounds (mmHg), rejecting windows where SBP/DBP each individually pass but their difference doesn't. |
| `--ppg-periodicity-threshold` / `--abp-periodicity-threshold` | 0.05 / 0.05    | Minimum autocorrelation periodicity score to keep a window.                                                                       |
| `--min-ppg-std`                                               | 0.005          | Minimum PPG window std, to reject flatline/disconnected-sensor windows.                                                           |
| `--min-valid-windows`                                         | 375            | Minimum surviving windows to keep a subject.                                                                                      |
| `--max-reject-fraction`                                       | 0.95           | Maximum fraction of a subject's windows that may be rejected before dropping the subject.                                         |
| `--max-bp-deviation`                                          | 40.0           | Max mmHg deviation from a patient's first valid window before a later window is dropped as an outlier.                            |
| `--split`                                                     | 0.6 0.2 0.2    | Patient-level train/val/test ratios (must sum to 1.0).                                                                            |
| `--seed`                                                      | 42             | Random seed for the split.                                                                                                        |

Pass `--verbose` to log per-subject/segment failures, `--no-progress` to
disable the tqdm bar. Run `--help` for the complete list.

## Output format

Each kept subject becomes one file `data/dataset/{split}/{subject_id}.npz`,
written by `write_patient_npz`. It packs five arrays:

| Key       | Shape       | dtype   | Meaning                                                      |
| --------- | ----------- | ------- | ------------------------------------------------------------ |
| `x`       | `(N, 1000)` | float32 | N PPG windows, 1000 samples each (8 s × 125 Hz).             |
| `y`       | `(N, 2)`    | float32 | `[SBP, DBP]` label (mmHg) for each window.                   |
| `calib_x` | `(1000,)`   | float32 | The patient's calibration window (their surviving window closest to their own median SBP/DBP). |
| `calib_y` | `(2,)`      | float32 | `[SBP, DBP]` of the calibration window.                      |
| `fs`      | scalar      | float32 | Sample rate (Hz).                                            |

`x` is by far the largest array (up to hundreds of MB per subject); the other
four are a few KB.

## The dataset must be stored **uncompressed**

`write_patient_npz` uses plain `np.savez`, **not** `np.savez_compressed`, and
this is a hard requirement, not a preference:

- **Memory.** The training loader memory-maps the large `x` array in place so
  a whole split is never read into RAM (see
  [train-model.md](train-model.md) and `bpe/dataset.py:_memmap_npz_array`).
  Memory-mapping an npz member only works when it is stored **uncompressed**
  (`ZIP_STORED`) — its raw `.npy` bytes then sit contiguously in the zip and
  can be addressed by file offset. A compressed member cannot be mapped and
  would have to be fully decoded into memory, which is exactly the behavior
  that exhausted RAM and got training processes OOM-killed when several runs
  ran at once.
- **Speed.** Uncompressed members are read/faulted straight from disk (an
  NVMe SSD here) into the OS page cache with no decode step, and that page
  cache is shared across concurrent training runs of the same split. Adding
  compression would trade a modest disk saving for per-load decode cost on
  every epoch and would break the memory-map path.

The array data is what dominates file size, and it is not compressible enough
to be worth it here anyway. If you rebuild or post-process the dataset, keep
`np.savez` (uncompressed). The loader guards against accidental compression:
it raises a clear error naming the offending file if it finds a compressed
`x` member, pointing back here.

## Console summary

On completion the script prints how many subjects were scanned / already done
/ processed / kept / excluded, the window retention rate, and the per-split
subject and window counts, followed by the output directory. Errors are
counted (not fatal) and retried on the next run.
