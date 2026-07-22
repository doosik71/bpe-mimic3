"""Generic training loop shared by the calibration-free and
calibration-based (Siamese) training runs. Model-specific batch handling
(how to get predictions and the loss) is injected via a `step_fn` so this
file has no knowledge of any particular architecture.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# step_fn(model, batch, device) -> (loss, pred_bp[B, 2], true_bp[B, 2]).
# pred_bp/true_bp are always in absolute [SBP, DBP] mmHg, even if the model
# itself regresses on a delta scale (see siamese_step) -- this keeps the MAE
# reported here comparable across both model families.
StepFn = Callable[[nn.Module, tuple, torch.device],
                  tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 5

_L1 = nn.L1Loss()


def calibration_free_step(model: nn.Module, batch, device: torch.device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    if hasattr(model, "compute_loss"):
        # Multi-task models (e.g. MTAE, MTAE_MLP, AE_LSTM) define this to mix
        # in a reconstruction loss alongside the BP regression loss -- see
        # bpe/models/mtae.py for the convention.
        loss, pred = model.compute_loss(x, y, _L1)
    else:
        pred = model(x)
        loss = _L1(pred, y)
    return loss, pred, y


def siamese_step(model: nn.Module, batch, device: torch.device):
    x, y, calib_x, calib_y = batch
    x, y, calib_x, calib_y = x.to(device), y.to(
        device), calib_x.to(device), calib_y.to(device)
    delta_pred = model(x, calib_x)
    delta_true = y - calib_y
    loss = _L1(delta_pred, delta_true)
    pred_bp = calib_y + delta_pred
    return loss, pred_bp, y


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_sbp_mae: float
    train_dbp_mae: float
    val_sbp_mae: float
    val_dbp_mae: float


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    step_fn: StepFn,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    desc: str = "",
) -> tuple[float, float, float]:
    from tqdm import tqdm

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs_err = torch.zeros(2)
    n_samples = 0
    for batch in tqdm(loader, desc=desc, unit="batch", ncols=100, ascii=True, leave=False):
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss, pred_bp, true_bp = step_fn(model, batch, device)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = true_bp.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_abs_err += (pred_bp.detach() - true_bp.detach()
                          ).abs().sum(dim=0).cpu()
        n_samples += batch_size
    mean_loss = total_loss / max(1, n_samples)
    sbp_mae = float(total_abs_err[0] / max(1, n_samples))
    dbp_mae = float(total_abs_err[1] / max(1, n_samples))
    return mean_loss, sbp_mae, dbp_mae


def train(
    model: nn.Module,
    step_fn: StepFn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = DEFAULT_PATIENCE,
    device: Optional[torch.device] = None,
    resume: Optional[Path] = None,
) -> list[EpochMetrics]:
    """Train `model` with Adam + L1 loss (docs/method-spectrogram-cnn.md §2/§3), early
    stopping on validation loss, and per-epoch checkpoints under `out_dir`.
    Returns the per-epoch metrics history."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []

    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(
        f"training on {device} for epoch(s) {start_epoch}-{epochs} "
        f"({len(train_loader)} train batch(es), {len(val_loader)} val batch(es) per epoch)..."
    )

    with metrics_path.open("a" if resume is not None else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if resume is None:
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "train_sbp_mae",
                    "train_dbp_mae", "val_sbp_mae", "val_dbp_mae"]
            )
            f.flush()

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss, train_sbp_mae, train_dbp_mae = _run_epoch(
                model, train_loader, step_fn, device, optimizer, desc=f"epoch {epoch}/{epochs} train"
            )
            val_loss, val_sbp_mae, val_dbp_mae = _run_epoch(
                model, val_loader, step_fn, device, None, desc=f"epoch {epoch}/{epochs} val"
            )
            elapsed = time.time() - t0

            metrics = EpochMetrics(
                epoch, train_loss, val_loss, train_sbp_mae, train_dbp_mae, val_sbp_mae, val_dbp_mae)
            history.append(metrics)
            writer.writerow([epoch, train_loss, val_loss, train_sbp_mae,
                            train_dbp_mae, val_sbp_mae, val_dbp_mae])
            f.flush()

            print(
                f"epoch {epoch:4d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_sbp_mae={val_sbp_mae:.2f}  val_dbp_mae={val_dbp_mae:.2f}  ({elapsed:.1f}s)"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, out_dir / "last.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                checkpoint["best_val_loss"] = best_val_loss
                torch.save(checkpoint, out_dir / "best.pt")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

    return history
