# train-model

Detailed design and user guide for the `train-model` module
(`scripts/train-model.py`, `bpe/trainer.py`, `bpe/dataset.py`), which trains a
calibration-free or calibration-based (Siamese) blood-pressure estimator on
`data/dataset/` (built by [construct-dataset](construct-dataset.md)).

The full command reference for every model lives in [COMMANDS.md](COMMANDS.md)
§4; this document is the design/behavior companion.

## Usage

```bash
uv run python scripts/train-model.py --model spectro_cnn      # calibration-free
uv run python scripts/train-model.py --model spectro_siamese  # calibration-based
```

`--model` selects the architecture from the registry
(`bpe/models/registry.py`), which also determines the dataset flavor and
training step used:

- **Calibration-free** (`acfa`, `ae_lstm`, `bpnet_cf`, `conv_reg`, `mtae`,
  `mtae_mlp`, `pctn`, `ppnet`, `resnet1d13`, `resnet1d21`, `resnet1d37`,
  `resnet1d61`, `spectro_cnn`, `st_resnet`): predicts `[SBP, DBP]` directly
  from a PPG window. Uses `CalibrationFreeDataset` + `calibration_free_step`.
- **Calibration-based** (`spectro_siamese`): a Siamese network that regresses
  `ΔBP` between the current window and the patient's calibration window, then
  reports `calibration_BP + ΔBP`. Uses `CalibrationPairDataset` +
  `siamese_step`.

### Key flags

| Flag                            | Default        | Purpose                                                        |
| ------------------------------- | -------------- | -------------------------------------------------------------- |
| `--model`                       | (required)     | Architecture name from the registry.                           |
| `--dataset-dir`                 | `data/dataset` | Directory holding `{train,val,test}/*.npz`.                    |
| `--models-dir`                  | `data/models`  | Checkpoints/metrics are written under `<models-dir>/<model>/`. |
| `--epochs`                      | 100            | Maximum training epochs.                                       |
| `--batch-size`                  | 32             | Paper default.                                                 |
| `--lr`                          | 1e-3           | Adam learning rate.                                            |
| `--weight-decay`                | 1e-4           | Adam weight decay.                                             |
| `--patience`                    | 15             | Early-stopping patience on validation loss.                    |
| `--embedding-dim` / `--dropout` | model default  | Backbone overrides.                                            |
| `--seed`                        | 42             | RNG seed (python/numpy/torch).                                 |
| `--device`                      | auto           | `auto` \| `cpu` \| `cuda` \| `cuda:N`.                         |
| `--workers`                     | 0              | DataLoader worker processes (see below).                       |
| `--no-normalize`                | off            | Skip per-window z-score normalization.                         |
| `--no-subject-balanced-sampling`| off            | Sample training windows uniformly instead of the default subject-balanced sampler (see below). |
| `--resume`                      | (fresh)        | Path to a checkpoint `.pt` to resume from.                     |

### Subject-balanced training sampler

Windows per subject are highly skewed (`dataset-statistic` found up to 40x
max/median, top 10% of subjects holding ~46% of all windows -- see
[dataset-statistic.md](dataset-statistic.md) §5), so plain uniform per-window
shuffling would let a handful of long-stay subjects dominate every training
batch. By default, `train_loader` instead uses
`bpe.dataset.SubjectBalancedSampler(train_set)`, which draws each subject
with equal probability and then a window uniformly within it, so every
subject has the same expected number of draws per epoch regardless of how
many windows they contributed. This is the exact distribution of a
`WeightedRandomSampler` weighting each window by `1/(its subject's window
count)`, but drawn two-level instead of from a length-N weight tensor --
which `torch.multinomial` rejects once a split exceeds `2**24` windows. This
only affects the **training** loader -- validation
still iterates every window once, unweighted, so eval metrics reflect the
true window distribution. Pass `--no-subject-balanced-sampling` to fall back
to plain uniform `shuffle=True` sampling.

## Training loop (`bpe/trainer.py`)

`train()` is a generic loop shared by both model families; the model-specific
batch handling (how to get predictions and the loss) is injected via a
`step_fn`, so the loop itself knows nothing about any architecture. Per the
source methodology it uses **Adam + L1 loss**, with:

- **Early stopping** on validation loss with patience `--patience`.
- **Checkpoints** written every epoch to `<models-dir>/<model>/`:
  `last.pt` (always) and `best.pt` (whenever val loss improves). Each holds
  `epoch`, `model_state_dict`, `optimizer_state_dict`, `best_val_loss`.
- **`metrics.csv`** appended every epoch:
  `epoch, train_loss, val_loss, train_sbp_mae, train_dbp_mae, val_sbp_mae,
  val_dbp_mae`. Reported MAE is always in absolute `[SBP, DBP]` mmHg for both
  families (the Siamese step converts its delta back), so runs are
  comparable.
- **`--resume`** restores model/optimizer/epoch/best-val-loss and appends to
  the existing `metrics.csv`.

Both `y`/`pred` MAE and the loss are accumulated sample-weighted across each
epoch (`_run_epoch`).

## Data loading and memory (`bpe/dataset.py`)

This is the part most likely to surprise you at scale, so it is documented in
full.

### The large array is memory-mapped, not read into RAM

The processed splits are big — on the order of `train ≈ 197 GB`,
`val ≈ 67 GB`, `test ≈ 62 GB` on disk. A naive "read the whole split into
RAM" loader made one training run hold **~264 GB** (train + val, both loaded
eagerly), and running four at once (e.g. in separate `tmux` panes) requested
roughly **1 TB** against 503 GB of RAM — the Linux OOM killer then sent
`SIGKILL` to the processes, which by definition die silently with no Python
traceback (observed as a run vanishing mid "loading val").

`load_split` now **memory-maps the large per-window `x` array in place**
(`_memmap_npz_array`) and reads only the small arrays (`y`, `calib_x`,
`calib_y`, `fs` — a few KB each) into RAM. `__getitem__` copies just the one
requested window (~4 KB) out of the map, so only the pages actually touched
are faulted in from disk. Measured effect: loading the `val` split peaks at
~2 GB resident instead of ~67 GB.

Why hand-rolled: `np.load(path, mmap_mode='r')` does **not** memory-map `.npz`
members — it reads the whole array into memory. `_memmap_npz_array` instead
locates the uncompressed member's byte offset inside the zip and `np.memmap`s
it directly. This requires the dataset to be stored **uncompressed** (see
[construct-dataset.md](construct-dataset.md)); the loader raises a clear error
if it finds a compressed `x` member.

### Consequences

- **Concurrent runs are now safe.** Multiple processes memory-mapping the
  same files share one copy of the data in the OS page cache (a single ~264 GB
  cache for any number of runs, not ×N), and file-backed pages are clean and
  reclaimable, so they do not drive the anonymous-memory growth that triggered
  the OOM kill. This holds on Linux and Windows alike.
- **Startup is instant** — no multi-minute upfront load. The first epoch is
  disk-bound as pages fault in (a one-time cost, ~1–2 min for 264 GB on this
  NVMe); later epochs are served from the page cache at RAM speed.
- **DataLoader workers.** `--workers` defaults to `0`. With memory-mapping the
  old concern that workers duplicate the in-memory arrays no longer applies
  (the data is file-backed and shared), so `--workers > 0` is now cheap and
  lets the first-epoch fault-in overlap with GPU compute. On Windows workers
  use `spawn` and re-map the files independently, which is fine.

### Open-file limit (handled automatically)

One memory-map (and thus one open file handle) is held per subject for the
run's lifetime. A training run holds **every loaded split open at once** —
train (~1541 subjects) *and* val (~514) together, ~2000+ handles — so the
count that matters is the sum across splits, not any single split. That
exceeds the common default soft `ulimit -n` of `1024`, and the soft limit
varies per shell/session (a login shell, a tmux pane, or a systemd service
may each differ), so requiring the caller to run `ulimit -n` first is fragile.

You do **not** need to raise it yourself. `load_split` raises the process's
own soft open-file limit generously toward its hard limit at startup
(`_raise_open_file_limit`) — a process may always do this without privileges —
so several splits open concurrently stay well within budget. This applies to
every consumer of the dataset (train, eval, dataset-browser).

Manual action is only needed in the rare case that the **hard** limit itself
is too low for the combined total (raising it needs admin: systemd
`LimitNOFILE=` or `/etc/security/limits.conf`). If so, the loader raises a
clear error naming the file, the subject count, and the current soft/hard
limits.

(Windows has no comparable low default and no `resource` module, so this is a
no-op there; its handle limit is far higher.)

## Crash / error reporting

Because an OOM kill leaves no message, the entrypoint installs handlers so
every *catchable* failure says why it happened instead of the run vanishing:

- `faulthandler.enable()` — dumps a Python traceback on a native fatal error
  (segfault/abort inside a C extension such as torch or numpy), which would
  otherwise terminate the process silently.
- **`SIGTERM` / `SIGHUP` handlers** — print which signal arrived, dump the
  stack, report peak memory, and exit `128 + signum` (covers scheduler kills
  and soft terminations).
- **Exception wrapper around `main()`** — a `MemoryError` prints an
  out-of-memory hint pointing at the loader; a `KeyboardInterrupt` (Ctrl-C)
  reports the interrupt and exits `130`; any other exception is re-printed
  under a clear header. Argparse's normal `--help` / bad-argument exits pass
  through unaffected.
- On any abnormal exit the **peak resident memory** is printed (POSIX), so an
  out-of-memory situation is easy to confirm.

**Limitation:** a hard `SIGKILL` (the OOM killer's `-9`, or `kill -9`) cannot
be caught by any program and still terminates silently — the peak-memory line
on other exits and the memory design above are the defense against reaching
that point. If a run still disappears with no message, check the kernel log
for an OOM event:

```bash
sudo dmesg | grep -i oom          # or: journalctl -k | grep -i oom
```

## Output

Everything is written under `data/models/<model>/`:

- `best.pt`, `last.pt` — checkpoints.
- `metrics.csv` — per-epoch training history.

Check progress with `generate-train-status` and evaluate with
`eval-model` / `eval-calib-model` (see [COMMANDS.md](COMMANDS.md) §5–6).
