"""Build the PyTorch-ready dataset from the segments in data/mimic3_index.csv
(produced by build-mimic3-index): resample/window/label/filter every
qualifying segment per docs/development-plan.md §4, then write
data/dataset/{train,val,test}/{subject_id}.npz.

Runs in two resumable phases (bpe.preprocess.pipeline.convert_dataset then
finalize_split): each subject is converted and written to disk as soon as
it's done, and recorded in data/dataset/_progress.csv, so re-running this
after an interruption picks up where it left off instead of starting over.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bpe.io.mimic3 import DEFAULT_INDEX_CSV, DEFAULT_MIMIC3_DIR
from bpe.preprocess.pipeline import (
    DEFAULT_ABP_PERIODICITY_THRESHOLD,
    DEFAULT_DATASET_DIR,
    DEFAULT_MAX_BP_DEVIATION,
    DEFAULT_MAX_REJECT_FRACTION,
    DEFAULT_MIN_VALID_WINDOWS,
    DEFAULT_PPG_PERIODICITY_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    DEFAULT_STRIDE_SEC,
    DEFAULT_TARGET_FS,
    DEFAULT_WINDOW_SEC,
    build_dataset,
)
from bpe.preprocess.quality import DEFAULT_DBP_RANGE, DEFAULT_SBP_RANGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic3-dir", type=Path, default=DEFAULT_MIMIC3_DIR)
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--stride-sec", type=float, default=DEFAULT_STRIDE_SEC)
    parser.add_argument(
        "--sbp-range",
        type=float,
        nargs=2,
        default=list(DEFAULT_SBP_RANGE),
        metavar=("MIN", "MAX"),
        help="Physiologically plausible SBP range in mmHg (default: %(default)s)",
    )
    parser.add_argument(
        "--dbp-range",
        type=float,
        nargs=2,
        default=list(DEFAULT_DBP_RANGE),
        metavar=("MIN", "MAX"),
        help="Physiologically plausible DBP range in mmHg (default: %(default)s)",
    )
    parser.add_argument(
        "--ppg-periodicity-threshold",
        type=float,
        default=DEFAULT_PPG_PERIODICITY_THRESHOLD,
        help="Minimum PPG periodicity score to keep a window (default: %(default)s; unvalidated, see docs/development-plan.md §7)",
    )
    parser.add_argument(
        "--abp-periodicity-threshold",
        type=float,
        default=DEFAULT_ABP_PERIODICITY_THRESHOLD,
        help="Minimum ABP periodicity score to keep a window (default: %(default)s; unvalidated, see docs/development-plan.md §7)",
    )
    parser.add_argument("--min-valid-windows", type=int, default=DEFAULT_MIN_VALID_WINDOWS)
    parser.add_argument("--max-reject-fraction", type=float, default=DEFAULT_MAX_REJECT_FRACTION)
    parser.add_argument("--max-bp-deviation", type=float, default=DEFAULT_MAX_BP_DEVIATION)
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=list(DEFAULT_SPLIT),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Patient-level train/val/test ratios, must sum to 1.0 (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit-subjects",
        type=int,
        default=None,
        help="Only process the first N subjects from the index (for quick trials)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar")
    parser.add_argument("--verbose", action="store_true", help="Log per-subject/segment failures")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every subject even if already recorded in _progress.csv "
        "(needed after changing QC parameters, since the ledger doesn't track them)",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Only run the conversion phase; leave converted npz files flat, unsplit",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Skip conversion; only (re-)assign already-converted subjects into train/val/test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    summary = build_dataset(
        mimic3_dir=args.mimic3_dir,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        target_fs=args.target_fs,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        sbp_range=tuple(args.sbp_range),
        dbp_range=tuple(args.dbp_range),
        ppg_periodicity_threshold=args.ppg_periodicity_threshold,
        abp_periodicity_threshold=args.abp_periodicity_threshold,
        min_valid_windows=args.min_valid_windows,
        max_reject_fraction=args.max_reject_fraction,
        max_bp_deviation=args.max_bp_deviation,
        split=tuple(args.split),
        seed=args.seed,
        limit_subjects=args.limit_subjects,
        workers=args.workers,
        show_progress=not args.no_progress,
        force=args.force,
        skip_split=args.skip_split,
        split_only=args.split_only,
    )

    convert = summary["convert"]
    if convert:
        print(f"subjects scanned          : {convert['subjects_scanned']}")
        print(f"subjects already done     : {convert['subjects_already_done']}")
        print(f"subjects processed now    : {convert['subjects_processed_this_run']}")
        print(f"  kept                    : {convert['subjects_kept_this_run']}")
        print(f"  excluded                : {convert['subjects_excluded_this_run']}")
        print(f"windows attempted (now)   : {convert['total_windows_attempted_this_run']}")
        print(f"windows kept (now)        : {convert['total_windows_kept_this_run']}")
        if convert["total_windows_attempted_this_run"]:
            rate = convert["total_windows_kept_this_run"] / convert["total_windows_attempted_this_run"]
            print(f"retention rate (now)      : {rate:.2%}")
        print(f"errors (retried next run) : {len(convert['errors'])}")

    split_summary = summary["split"]
    if split_summary:
        print(f"subjects kept (total)     : {split_summary['subjects_kept']}")
        for split_name in ("train", "val", "test"):
            n_subj = split_summary["subjects_by_split"].get(split_name, 0)
            n_win = split_summary["windows_by_split"].get(split_name, 0)
            print(f"  {split_name:<5s}                   : {n_subj} subjects, {n_win} windows")
        print(f"moved this run            : {split_summary['moved']}")
        print(f"already in place          : {split_summary['already_in_place']}")
        if split_summary["missing"]:
            print(
                f"missing (kept but no npz) : {len(split_summary['missing'])} "
                f"e.g. {split_summary['missing'][:5]}"
            )

    print(f"dataset directory         : {args.output_dir}")


if __name__ == "__main__":
    main()
