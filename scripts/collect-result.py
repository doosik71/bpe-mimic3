"""Collect eval_results.json (and its plots) from every trained model
under data/models/ into a single data/results/ directory, for easy
archiving/sharing separately from the (much larger) checkpoint files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from bpe.reporting import DEFAULT_MODELS_DIR, DEFAULT_RESULTS_DIR, list_model_dirs

_RESULT_FILES = ("eval_results.json", "eval_plot.png", "error_hist.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = list_model_dirs(args.models_dir)

    collected = 0
    skipped = []
    for model_dir in model_dirs:
        if not (model_dir / "eval_results.json").is_file():
            skipped.append(model_dir.name)
            continue
        dst_dir = args.results_dir / model_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in _RESULT_FILES:
            src_file = model_dir / name
            if src_file.is_file():
                shutil.copy2(src_file, dst_dir / name)
        collected += 1

    print(f"collected {collected} model result(s) into {args.results_dir}")
    if skipped:
        print(f"skipped {len(skipped)} model(s) with no eval_results.json: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
