"""Shared evaluation utilities: run inference over a test split, compute
clinical metrics, and save results/plots -- for both calibration-free and
calibration-based (Siamese) models. Model-specific inference is injected
via a `predict_fn`, mirroring bpe/trainer.py's `step_fn` pattern.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bpe.metrics import aami_pass, bhs_cumulative_percentages, bhs_grade, compute_error_stats

# predict_fn(model, batch, device) -> (pred_bp[B, 2], true_bp[B, 2]), both on CPU.
PredictFn = Callable[[nn.Module, tuple, torch.device], tuple[torch.Tensor, torch.Tensor]]


def calibration_free_predict(model: nn.Module, batch, device: torch.device):
    x, y = batch
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
    return pred.cpu(), y


def siamese_predict(model: nn.Module, batch, device: torch.device):
    x, y, calib_x, calib_y = batch
    x, calib_x, calib_y = x.to(device), calib_x.to(device), calib_y.to(device)
    with torch.no_grad():
        pred_bp = model.predict_bp(x, calib_x, calib_y)
    return pred_bp.cpu(), y


def collect_predictions(
    model: nn.Module, loader: DataLoader, predict_fn: PredictFn, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    from tqdm import tqdm

    model.eval()
    preds, trues = [], []
    print(f"running inference over {len(loader)} batch(es)...")
    for batch in tqdm(loader, desc="evaluating", unit="batch", ncols=100, ascii=True):
        pred, true = predict_fn(model, batch, device)
        preds.append(pred.numpy())
        trues.append(true.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def evaluate(pred: np.ndarray, true: np.ndarray) -> dict:
    """`pred`/`true`: `(N, 2)` `[SBP, DBP]` arrays -> per-target metrics."""
    results: dict = {}
    for i, name in enumerate(("sbp", "dbp")):
        stats = compute_error_stats(pred[:, i], true[:, i])
        pct5, pct10, pct15 = bhs_cumulative_percentages(pred[:, i], true[:, i])
        results[name] = {
            **asdict(stats),
            "bhs_grade": bhs_grade(pred[:, i], true[:, i]),
            "bhs_pct_5mmHg": pct5,
            "bhs_pct_10mmHg": pct10,
            "bhs_pct_15mmHg": pct15,
            "aami_pass": aami_pass(stats),
        }
    results["n_windows"] = int(len(true))
    return results


def save_results_json(results: dict, out_path: Path) -> None:
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")


def save_scatter_plot(pred: np.ndarray, true: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, i, name in zip(axes, (0, 1), ("SBP", "DBP")):
        ax.scatter(true[:, i], pred[:, i], s=4, alpha=0.3)
        lo = float(min(true[:, i].min(), pred[:, i].min()))
        hi = float(max(true[:, i].max(), pred[:, i].max()))
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
        ax.set_xlabel(f"True {name} (mmHg)")
        ax.set_ylabel(f"Predicted {name} (mmHg)")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_error_histograms(pred: np.ndarray, true: np.ndarray, out_path: Path) -> None:
    error = pred - true
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, i, name in zip(axes, (0, 1), ("SBP", "DBP")):
        ax.hist(error[:, i], bins=50, color="tab:blue", alpha=0.8)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel(f"{name} error (pred - true) mmHg")
        ax.set_ylabel("Count")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_and_report(
    model: nn.Module,
    loader: DataLoader,
    predict_fn: PredictFn,
    device: torch.device,
    model_dir: Path,
) -> dict:
    """Run inference, compute metrics, write eval_results.json / eval_plot.png
    / error_hist.png into `model_dir`, print a summary, and return the
    results dict."""
    model_dir = Path(model_dir)
    pred, true = collect_predictions(model, loader, predict_fn, device)
    results = evaluate(pred, true)

    save_results_json(results, model_dir / "eval_results.json")
    save_scatter_plot(pred, true, model_dir / "eval_plot.png")
    save_error_histograms(pred, true, model_dir / "error_hist.png")

    for name in ("sbp", "dbp"):
        r = results[name]
        print(
            f"{name.upper()}: MAE={r['mae']:.2f} RMSE={r['rmse']:.2f} ME={r['me']:.2f} SD={r['sd']:.2f} "
            f"BHS={r['bhs_grade']} AAMI={'PASS' if r['aami_pass'] else 'FAIL'}"
        )
    print(f"n_windows={results['n_windows']}")
    print(f"results written to {model_dir}")
    return results
