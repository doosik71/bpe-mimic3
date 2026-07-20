"""Plot per-epoch loss/MAE curves for one training run
(data/models/<model>) from its metrics.csv, and print a summary table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe.reporting import plot_train_status, print_train_status_summary, read_metrics_csv, summarize_train_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Run directory containing metrics.csv, e.g. data/models/cnn")
    parser.add_argument("--no-save", action="store_true", help="Print the summary only; skip writing PNG files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_metrics_csv(args.model_dir)
    summary = summarize_train_status(args.model_dir, rows)
    if summary is None:
        print(f"{args.model_dir.name}: no metrics.csv found")
        return
    print_train_status_summary(summary)
    if not args.no_save:
        plot_train_status(args.model_dir, rows)
        print(f"loss_graph.png / mae_graph.png written to {args.model_dir}")


if __name__ == "__main__":
    main()
