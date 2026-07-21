"""Generate overview scatter plots comparing every trained model's
parameter count against its evaluation accuracy (MAE, RMSE), colored by
BHS grade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from bpe.reporting import (
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    count_parameters,
    list_model_dirs,
    print_run_info,
    read_eval_results,
)

_GRADE_COLORS = {"A": "tab:green", "B": "tab:blue", "C": "tab:orange", "D": "tab:red"}


def _collect(models_dir: Path) -> list[dict]:
    from tqdm import tqdm

    model_dirs = list_model_dirs(models_dir)
    print(f"found {len(model_dirs)} model run(s) under {models_dir}")
    rows = []
    for model_dir in tqdm(model_dirs, desc="collecting", unit="model", ncols=100, ascii=True):
        results = read_eval_results(model_dir)
        if results is None:
            continue
        n_params = count_parameters(model_dir)
        if n_params is None:
            continue
        rows.append({"model": model_dir.name, "n_params": n_params, "sbp": results["sbp"], "dbp": results["dbp"]})
    return rows


def _scatter(rows: list[dict], metric: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, target in zip(axes, ("sbp", "dbp")):
        for row in rows:
            grade = row[target]["bhs_grade"]
            color = _GRADE_COLORS.get(grade, "gray")
            ax.scatter(row["n_params"], row[target][metric], color=color, s=40)
            ax.annotate(
                row["model"],
                (row["n_params"], row[target][metric]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_ylabel(f"{metric.upper()} (mmHg)")
        ax.set_title(target.upper())
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=g, markersize=8)
        for g, c in _GRADE_COLORS.items()
    ]
    fig.legend(handles=handles, title="BHS grade", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing per-model training runs (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory the overview PNGs are written into (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_run_info("generate-overview", {"models dir": args.models_dir, "output dir": args.output_dir})
    rows = _collect(args.models_dir)
    if not rows:
        print(f"no evaluated models found under {args.models_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _scatter(rows, "mae", args.output_dir / "overview_mae.png")
    _scatter(rows, "rmse", args.output_dir / "overview_rmse.png")
    print(f"overview_mae.png / overview_rmse.png written to {args.output_dir} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
