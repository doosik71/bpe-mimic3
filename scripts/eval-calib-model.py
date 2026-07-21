"""Evaluate a trained calibration-based (Siamese) model on a data/dataset
split, using each patient's stored calibration pair: MAE, RMSE, ME, SD, BHS
cumulative-error grade, and AAMI pass/fail for SBP and DBP. Writes
eval_results.json, eval_plot.png, error_hist.png next to the checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bpe.dataset import DEFAULT_DATASET_DIR, CalibrationPairDataset
from bpe.evaluate import run_and_report, siamese_predict
from bpe.models.registry import build_calibration_based_model, list_calibration_based_models
from bpe.reporting import print_run_info


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Run directory containing a checkpoint, e.g. data/models/siamese")
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename inside model_dir (default: %(default)s)")
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Directory holding train/val/test npz files (default: %(default)s)"
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate on (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size (default: %(default)s)")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (default: %(default)s)")
    parser.add_argument("--no-normalize", action="store_true", help="Skip per-window z-score normalization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    model_name = args.model_dir.name
    if model_name not in list_calibration_based_models():
        raise SystemExit(
            f"{model_name!r} is not a calibration-based model name "
            f"(available: {list_calibration_based_models()}); use eval-model for calibration-free models"
        )

    print_run_info(
        "eval-calib-model",
        {
            "model dir": args.model_dir,
            "checkpoint": args.checkpoint,
            "dataset dir": args.dataset_dir,
            "split": args.split,
            "device": device,
            "batch size": args.batch_size,
            "workers": args.workers,
            "normalize": not args.no_normalize,
        },
    )

    print(f"loading checkpoint {args.model_dir / args.checkpoint}...")
    checkpoint = torch.load(args.model_dir / args.checkpoint, map_location=device)
    model = build_calibration_based_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    dataset = CalibrationPairDataset(args.dataset_dir, args.split, normalize=not args.no_normalize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    run_and_report(model, loader, siamese_predict, device, args.model_dir)


if __name__ == "__main__":
    main()
