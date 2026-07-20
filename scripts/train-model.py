"""Train a calibration-free or calibration-based (Siamese) BP estimator on
data/dataset. `--model` selects which registry the run belongs to
(bpe/models/registry.py), which in turn determines the dataset flavor and
training step used -- see bpe/dataset.py and bpe/trainer.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

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
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32, help="Paper default (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stopping patience on val loss")
    parser.add_argument(
        "--embedding-dim", type=int, default=None, help="Override the backbone embedding dim (default: model default)"
    )
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout probability (default: model default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Default 0: the dataset is already fully "
        "in-memory numpy arrays, so extra worker processes mostly just duplicate that "
        "memory (Windows uses spawn) for little benefit.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-window z-score normalization")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint .pt to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    normalize = not args.no_normalize

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

    out_dir = args.models_dir / args.model
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
    main()
