"""Train a calibration-free or calibration-based (Siamese) BP estimator on
data/dataset. `--model` selects which registry the run belongs to
(bpe/models/registry.py), which in turn determines the dataset flavor and
training step used -- see bpe/dataset.py and bpe/trainer.py.
"""

from __future__ import annotations

import argparse
import faulthandler
import random
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from bpe.dataset import CalibrationFreeDataset, CalibrationPairDataset, DEFAULT_DATASET_DIR
from bpe.models.registry import (
    build_calibration_based_model,
    build_calibration_free_model,
    list_calibration_based_models,
    list_calibration_free_models,
)
from bpe.reporting import print_run_info
from bpe.trainer import (
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    calibration_free_step,
    siamese_step,
    train,
)

DEFAULT_MODELS_DIR = Path("data/models")


def _peak_memory_mb() -> Optional[float]:
    """Best-effort peak resident memory of this process, or None if it
    can't be determined (e.g. on Windows, which has no `resource` module).
    Reported on abnormal exit so an out-of-memory kill is easy to spot."""
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kibibytes on Linux.
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _print_peak_memory() -> None:
    peak = _peak_memory_mb()
    if peak is not None:
        print(f"train-model: peak memory use was {peak:.0f} MiB.", file=sys.stderr, flush=True)


def _install_fault_handlers() -> None:
    """Make an abnormal termination say *something* instead of dying
    silently mid-run (as happened during val loading with no message).

    Covers the failures a process can actually observe:
      - a native fatal error (segfault/abort in a C extension such as
        torch or numpy) -- faulthandler dumps a Python traceback that the
        interpreter would otherwise skip;
      - a catchable termination signal (SIGTERM/SIGHUP from a scheduler or
        a soft OOM kill) -- reported below before we exit.
    A hard `kill -9` / OOM-killer SIGKILL cannot be caught by any program,
    so it necessarily stays silent; the peak-memory line helps flag it."""
    faulthandler.enable()

    def _on_signal(signum, frame):
        name = signal.Signals(signum).name
        print(f"\ntrain-model: received {name}; terminating.", file=sys.stderr, flush=True)
        traceback.print_stack(frame, file=sys.stderr)
        _print_peak_memory()
        sys.exit(128 + signum)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            # e.g. not on the main thread, or unsupported on this platform.
            pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    all_models = list_calibration_free_models() + list_calibration_based_models()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=all_models, help="Model name from the registry")
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Directory holding train/val/test npz files (default: %(default)s)"
    )
    parser.add_argument(
        "--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory checkpoints/metrics are written under (default: %(default)s)"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Maximum training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=32, help="Paper default (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (default: %(default)s)")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Adam weight decay (default: %(default)s)")
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stopping patience on val loss (default: %(default)s)"
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=None, help="Override the backbone embedding dim (default: model default)"
    )
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout probability (default: model default)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: %(default)s)")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N (default: %(default)s)")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: %(default)s). Default 0: the dataset is already fully "
        "in-memory numpy arrays, so extra worker processes mostly just duplicate that "
        "memory (Windows uses spawn) for little benefit.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-window z-score normalization")
    parser.add_argument(
        "--resume", type=Path, default=None, help="Path to a checkpoint .pt to resume from (default: start a fresh run)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    normalize = not args.no_normalize
    out_dir = args.models_dir / args.model

    print_run_info(
        "train-model",
        {
            "model": args.model,
            "dataset dir": args.dataset_dir,
            "output dir": out_dir,
            "device": device,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "lr": args.lr,
            "weight decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "normalize": normalize,
            "workers": args.workers,
            "resume": args.resume if args.resume is not None else "(none, fresh run)",
        },
    )

    model_kwargs = {}
    if args.embedding_dim is not None:
        model_kwargs["embedding_dim"] = args.embedding_dim
    if args.dropout is not None:
        model_kwargs["dropout"] = args.dropout

    if args.model in list_calibration_based_models():
        model = build_calibration_based_model(args.model, **model_kwargs)
        train_set = CalibrationPairDataset(args.dataset_dir, "train", normalize=normalize)
        val_set = CalibrationPairDataset(args.dataset_dir, "val", normalize=normalize)
        step_fn = siamese_step
    else:
        model = build_calibration_free_model(args.model, **model_kwargs)
        train_set = CalibrationFreeDataset(args.dataset_dir, "train", normalize=normalize)
        val_set = CalibrationFreeDataset(args.dataset_dir, "val", normalize=normalize)
        step_fn = calibration_free_step

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    history = train(
        model,
        step_fn,
        train_loader,
        val_loader,
        out_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
        resume=args.resume,
    )
    print(f"finished after {len(history)} epoch(s); checkpoints + metrics.csv written to {out_dir}")


if __name__ == "__main__":
    _install_fault_handlers()
    try:
        main()
    except KeyboardInterrupt:
        print("\ntrain-model: interrupted by user (Ctrl-C).", file=sys.stderr, flush=True)
        sys.exit(130)
    except MemoryError:
        print(
            "\ntrain-model: out of memory. Each split is loaded into RAM eagerly "
            "(see bpe.dataset.load_split); use a machine with more memory or a "
            "smaller dataset split.",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        _print_peak_memory()
        sys.exit(1)
    except Exception:
        # Any other error would normally print a traceback, but the tqdm
        # progress bar can bury it; re-print it explicitly with a clear
        # header so the run never ends without a reason.
        print("\ntrain-model: terminated by an unhandled error:", file=sys.stderr, flush=True)
        traceback.print_exc()
        _print_peak_memory()
        sys.exit(1)
