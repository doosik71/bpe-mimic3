"""Scan data/mimic3 for waveform segments carrying both PLETH and an
arterial BP channel, and write the result to data/mimic3_index.csv.

This is the pruning step that keeps the rest of the pipeline from having to
re-scan the full ~22k-record matched subset on every run; see
docs/development-plan.md §4 step 1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bpe.io.mimic3 import DEFAULT_INDEX_CSV, DEFAULT_MIMIC3_DIR, build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mimic3-dir",
        type=Path,
        default=DEFAULT_MIMIC3_DIR,
        help="Root of the MIMIC-III waveform matched subset (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_CSV,
        help="Output index CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N records from RECORDS-waveforms (for quick trials; default: no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread-pool size for concurrent header scanning (default: %(default)s)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar")
    parser.add_argument("--verbose", action="store_true", help="Log per-record/segment scan failures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    summary = build_index(
        mimic3_dir=args.mimic3_dir,
        output_csv=args.output,
        limit=args.limit,
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    print(f"records scanned       : {summary['records_scanned']}")
    print(f"qualifying segments   : {summary['qualifying_segments']}")
    print(f"qualifying records    : {summary['qualifying_records']}")
    print(f"qualifying subjects   : {summary['qualifying_subjects']}")
    print(f"total qualifying time : {summary['total_duration_hr']:.1f} h")
    print(f"errors                : {len(summary['errors'])}")
    print(f"index written to      : {args.output}")


if __name__ == "__main__":
    main()
