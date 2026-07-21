"""Generate training-status graphs and summaries for every trained model
under data/models/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe.reporting import (
    DEFAULT_MODELS_DIR,
    list_model_dirs,
    plot_train_status,
    print_run_info,
    print_train_status_summary,
    read_metrics_csv,
    summarize_train_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing per-model training runs (default: %(default)s)")
    parser.add_argument("--no-save", action="store_true", help="Print summaries only; skip writing PNG files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_run_info("generate-all-train-status", {"models dir": args.models_dir, "save plots": not args.no_save})

    model_dirs = list_model_dirs(args.models_dir)
    if not model_dirs:
        print(f"no model runs found under {args.models_dir}")
        return
    print(f"found {len(model_dirs)} model run(s) under {args.models_dir}")

    for i, model_dir in enumerate(model_dirs, start=1):
        print(f"[{i}/{len(model_dirs)}] {model_dir.name}")
        rows = read_metrics_csv(model_dir)
        summary = summarize_train_status(model_dir, rows)
        if summary is None:
            print(f"{model_dir.name}: no metrics.csv found, skipping")
            print("-" * 40)
            continue
        print_train_status_summary(summary)
        if not args.no_save:
            plot_train_status(model_dir, rows)
        print("-" * 40)


if __name__ == "__main__":
    main()
