"""Shared helpers for the reporting/analysis tooling: reading per-run
metrics.csv and eval_results.json files across data/models/*, and plotting
training-status curves. Kept as one small module since these are all just
different views over the same two artifact types (bpe.trainer's metrics.csv,
bpe.evaluate's eval_results.json).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_RESULTS_DIR = Path("data/results")


def list_model_dirs(models_dir: Path) -> list[Path]:
    """Every immediate subdirectory of `models_dir` -- each is one model's
    training run."""
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    return sorted(p for p in models_dir.iterdir() if p.is_dir())


def read_metrics_csv(model_dir: Path) -> list[dict]:
    """Read metrics.csv (written by bpe.trainer.train) as a list of
    per-epoch dicts. Empty if the run hasn't produced one yet."""
    path = Path(model_dir) / "metrics.csv"
    if not path.is_file():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {"epoch": int(row["epoch"])}
            parsed.update({key: float(value) for key, value in row.items() if key != "epoch"})
            rows.append(parsed)
    return rows


def read_eval_results(model_dir: Path) -> Optional[dict]:
    """Read eval_results.json (written by bpe.evaluate.run_and_report), or
    None if the model hasn't been evaluated yet."""
    path = Path(model_dir) / "eval_results.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_parameters(model_dir: Path) -> Optional[int]:
    """Best-effort parameter count from a saved checkpoint, without
    needing to know the model's architecture (a state_dict's tensors carry
    their own shapes)."""
    for name in ("best.pt", "last.pt"):
        path = Path(model_dir) / name
        if path.is_file():
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            return sum(tensor.numel() for tensor in state_dict.values())
    return None


def plot_train_status(model_dir: Path, rows: list[dict]) -> None:
    """Write loss_graph.png and mae_graph.png into `model_dir`."""
    model_dir = Path(model_dir)
    epochs = [r["epoch"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train_loss")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (L1)")
    ax.set_title(f"{model_dir.name} -- loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "loss_graph.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for key in ("train_sbp_mae", "train_dbp_mae", "val_sbp_mae", "val_dbp_mae"):
        ax.plot(epochs, [r[key] for r in rows], label=key)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title(f"{model_dir.name} -- MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "mae_graph.png", dpi=150)
    plt.close(fig)


def summarize_train_status(model_dir: Path, rows: list[dict]) -> Optional[dict]:
    """A single-epoch summary (best + last), or None if `rows` is empty."""
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val_loss"])
    last = rows[-1]
    return {
        "model": Path(model_dir).name,
        "epochs_run": len(rows),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_sbp_mae": best["val_sbp_mae"],
        "best_val_dbp_mae": best["val_dbp_mae"],
        "last_epoch": last["epoch"],
        "last_val_loss": last["val_loss"],
    }


def print_train_status_summary(summary: dict) -> None:
    print(f"model            : {summary['model']}")
    print(f"epochs run       : {summary['epochs_run']}")
    print(f"best epoch       : {summary['best_epoch']} (val_loss={summary['best_val_loss']:.4f})")
    print(f"best val_sbp_mae : {summary['best_val_sbp_mae']:.2f}")
    print(f"best val_dbp_mae : {summary['best_val_dbp_mae']:.2f}")
    print(f"last epoch       : {summary['last_epoch']} (val_loss={summary['last_val_loss']:.4f})")
