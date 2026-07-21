"""Summarize every trained model's eval_results.json into a single
summary.csv (one row per model, one column per sbp/*, dbp/* metric).
Models missing eval_results.json, or reporting a different subset of
fields, are handled transparently: missing files are skipped and missing
columns are left blank.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bpe.reporting import DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, list_model_dirs, print_run_info, read_eval_results


def _flatten(model_name: str, results: dict) -> dict:
    row = {"model": model_name, "n_windows": results.get("n_windows")}
    for target in ("sbp", "dbp"):
        for key, value in results.get(target, {}).items():
            row[f"{target}/{key}"] = value
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory containing per-model training runs (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory summary.csv is written into (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    from tqdm import tqdm

    args = parse_args()
    print_run_info("summarize-result", {"models dir": args.models_dir, "results dir": args.results_dir})

    model_dirs = list_model_dirs(args.models_dir)
    print(f"found {len(model_dirs)} model run(s) under {args.models_dir}")

    rows = []
    skipped = []
    for model_dir in tqdm(model_dirs, desc="summarizing", unit="model", ncols=100, ascii=True):
        results = read_eval_results(model_dir)
        if results is None:
            skipped.append(model_dir.name)
            continue
        rows.append(_flatten(model_dir.name, results))

    if not rows:
        print(f"no eval_results.json found under {args.models_dir}")
        return

    fieldnames = ["model", "n_windows"]
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"summarized {len(rows)} model(s) into {out_path}")
    if skipped:
        print(f"skipped {len(skipped)} model(s) with no eval_results.json: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
