"""Retrofit newly added QC rules onto an already-built data/dataset without
re-running the full (slow) raw-signal pipeline (build-mimic3-index +
construct-dataset), which re-reads every WFDB segment from data/mimic3.

Applies two changes made after the dataset was originally built (see
docs/data-cleaning.md):

1. Drops windows whose pulse pressure (SBP - DBP) falls outside
   `--pulse-pressure-range` (bpe.preprocess.quality.pulse_pressure_ok).
2. Recomputes each surviving patient's calibration pair with the current
   `calibration_index` selection -- the surviving window closest to that
   patient's own median SBP/DBP -- instead of whatever rule produced the
   `calib_x`/`calib_y` already stored in the npz.

Patient-level exclusion (`should_exclude_patient`) is re-checked against
each subject's *original* `n_windows_total` from `data/dataset/_progress.csv`
(unaffected by this script), so a subject who drops below
`--min-valid-windows` or above `--max-reject-fraction` once the new filter
is applied is fully excluded to match what a from-scratch rebuild would do:
their npz is deleted and `_progress.csv` gets a new row marking them
"excluded". Subjects that stay kept get a new row recording their updated
`n_windows_kept` (last row per subject_id wins -- see `read_progress`).

Known approximation: in a from-scratch rebuild, the pulse-pressure filter
runs *before* per-patient outlier removal (data-cleaning.md step 4), so a
window it would reject can never become that patient's outlier-removal
reference point. This script can only re-filter what already survived the
*old* outlier removal -- the windows that old removal rejected are gone,
not stored in the npz -- so if the *old* reference window (the
chronologically-first survivor) itself fails the new pulse-pressure check,
the downstream set of windows that removal treated as "outliers" is not
re-derived here. This should be rare in practice (the pulse-pressure filter
was found to affect very few windows in dataset-statistic.md §5), but it
means this migration is an approximation, not a bit-exact match for a full
rebuild. Use --dry-run to inspect impact before committing, and re-run
`construct-dataset --force` from raw data if exact parity matters.

WARNING: with no --output-dir, this OVERWRITES data/dataset IN PLACE
(each subject's npz is replaced, and newly-excluded subjects' npz files are
deleted). Writes are atomic per-subject (written to a .tmp file then
os.replace'd), so a crash mid-run leaves at most one subject file untouched
rather than corrupted, but there is no way to undo an already-committed
change other than rebuilding from data/mimic3. Run with --dry-run first.

Usage:
    uv run python scripts/migrate_dataset.py --dry-run
    uv run python scripts/migrate_dataset.py
    uv run python scripts/migrate_dataset.py --output-dir data/dataset_migrated
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from bpe.preprocess.patient import (
    DEFAULT_MAX_REJECT_FRACTION,
    DEFAULT_MIN_VALID_WINDOWS,
    calibration_index,
    should_exclude_patient,
)
from bpe.preprocess.pipeline import (
    DEFAULT_DATASET_DIR,
    PROGRESS_FILENAME,
    ProgressRow,
    _append_progress,
    read_progress,
)
from bpe.preprocess.quality import DEFAULT_PULSE_PRESSURE_RANGE, pulse_pressure_ok
from bpe.reporting import print_run_info

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset directory to migrate, holding _progress.csv and train/val/test/ (default: %(default)s)")
    p.add_argument("--output-dir", type=Path, default=None, help="Where migrated npz + _progress.csv are written (default: --dataset-dir, i.e. overwrite in place)")
    p.add_argument(
        "--pulse-pressure-range",
        type=float,
        nargs=2,
        default=list(DEFAULT_PULSE_PRESSURE_RANGE),
        metavar=("MIN", "MAX"),
        help="Plausible pulse-pressure (SBP-DBP) range in mmHg (default: %(default)s)",
    )
    p.add_argument("--min-valid-windows", type=int, default=DEFAULT_MIN_VALID_WINDOWS, help="Re-exclusion threshold (default: %(default)s)")
    p.add_argument("--max-reject-fraction", type=float, default=DEFAULT_MAX_REJECT_FRACTION, help="Re-exclusion threshold (default: %(default)s)")
    p.add_argument("--limit-subjects", type=int, default=None, help="Only migrate the first N kept subjects, for a quick trial (default: no limit)")
    p.add_argument("--workers", type=int, default=8, help="Thread-pool size for concurrent subject migration (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="Report what would change without writing or deleting anything")
    p.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar")
    p.add_argument("--verbose", action="store_true", help="Print one line per subject (in addition to the summary)")
    return p.parse_args()


def _find_subject_npz(dataset_dir: Path, subject_id: str) -> Optional[tuple[Path, Optional[str]]]:
    """Locate a kept subject's npz, whether it's already split into
    train/val/test or still flat (pre-`finalize_split`) under `dataset_dir`.
    Returns `(path, split_name)`, with `split_name=None` for the flat case."""
    for split in SPLITS:
        path = dataset_dir / split / f"{subject_id}.npz"
        if path.is_file():
            return path, split
    path = dataset_dir / f"{subject_id}.npz"
    if path.is_file():
        return path, None
    return None


def _write_npz_atomic(path: Path, x: np.ndarray, y: np.ndarray, calib_x: np.ndarray, calib_y: np.ndarray, fs: float) -> None:
    """Write uncompressed (np.savez, matching write_patient_npz's format
    requirement -- see docs/construct-dataset.md) to a temp file in the same
    directory, then atomically replace `path`. Passing an open file object
    (not a path/string) to np.savez avoids its automatic '.npz' suffix
    logic, which would otherwise mangle the '.tmp' name."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        np.savez(fh, x=x, y=y, calib_x=calib_x, calib_y=calib_y, fs=np.float32(fs))
    tmp_path.replace(path)


def migrate_subject(
    subject_id: str,
    path: Path,
    split_name: Optional[str],
    n_windows_total: int,
    pulse_pressure_range: tuple[float, float],
    min_valid_windows: int,
    max_reject_fraction: float,
    dry_run: bool,
    progress_lock: threading.Lock,
) -> dict:
    """Apply the pulse-pressure filter and recompute the calibration window
    for one subject's already-built npz. Returns a per-subject result dict;
    never raises for QC outcomes (only for I/O/format errors), so the
    caller can tally results without special-casing exclusion.

    The heavy per-subject work (npz read, numpy filtering, npz write) needs
    no locking -- each subject owns a distinct file. Only the shared
    `_progress.csv` append is serialized via `progress_lock`, since
    concurrent appends from multiple threads could otherwise interleave.
    """
    with np.load(path) as data:
        x = np.array(data["x"])
        y = np.array(data["y"])
        fs = float(data["fs"])

    n_before = int(y.shape[0])
    keep_mask = np.array([pulse_pressure_ok(sbp, dbp, pulse_pressure_range) for sbp, dbp in y], dtype=bool)
    n_after = int(keep_mask.sum())

    result = {
        "subject_id": subject_id,
        "split": split_name,
        "n_before": n_before,
        "n_after": n_after,
        "n_dropped": n_before - n_after,
    }

    progress_dir = path.parents[1] if split_name else path.parent

    if should_exclude_patient(n_after, n_windows_total, min_valid_windows, max_reject_fraction):
        result["outcome"] = "excluded"
        if not dry_run:
            path.unlink()
            with progress_lock:
                _append_progress(progress_dir, ProgressRow(subject_id, "excluded", n_windows_total, 0))
        return result

    new_x = x[keep_mask]
    new_y = y[keep_mask]
    labels = [(float(sbp), float(dbp)) for sbp, dbp in new_y]
    calib_idx = calibration_index(labels, [True] * len(labels))
    # Can't be None: should_exclude_patient above already guarantees
    # n_after >= 1 (min_valid_windows is always >= 1 in practice).
    new_calib_x = new_x[calib_idx]
    new_calib_y = new_y[calib_idx]

    result["outcome"] = "kept"
    if not dry_run:
        _write_npz_atomic(path, new_x, new_y, new_calib_x, new_calib_y, fs)
        with progress_lock:
            _append_progress(progress_dir, ProgressRow(subject_id, "kept", n_windows_total, n_after))
    return result


def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    output_dir: Path = args.output_dir or dataset_dir
    pulse_pressure_range = tuple(args.pulse_pressure_range)

    print_run_info(
        "migrate-dataset",
        {
            "dataset dir": dataset_dir,
            "output dir": output_dir,
            "pulse pressure range": pulse_pressure_range,
            "min valid windows": args.min_valid_windows,
            "max reject fraction": args.max_reject_fraction,
            "workers": args.workers,
            "dry run": args.dry_run,
        },
    )

    if output_dir != dataset_dir:
        raise NotImplementedError(
            "writing to a different --output-dir is not implemented -- this script overwrites "
            "--dataset-dir in place. Copy data/dataset to --output-dir yourself first if you want "
            "to migrate a copy, then pass that path as --dataset-dir instead."
        )

    progress = read_progress(dataset_dir)
    if not progress:
        print(f"ERROR: no {PROGRESS_FILENAME} found under {dataset_dir}; nothing to migrate.")
        return

    kept_ids = sorted(sid for sid, row in progress.items() if row.status == "kept")
    if args.limit_subjects is not None:
        kept_ids = kept_ids[: args.limit_subjects]
    print(f"found {len(kept_ids)} kept subject(s) to migrate...")

    jobs: list[tuple[str, Path, Optional[str], int]] = []
    missing: list[str] = []
    for subject_id in kept_ids:
        located = _find_subject_npz(dataset_dir, subject_id)
        if located is None:
            missing.append(subject_id)
            continue
        path, split_name = located
        jobs.append((subject_id, path, split_name, progress[subject_id].n_windows_total))

    progress_bar = None
    if not args.no_progress:
        from tqdm import tqdm

        progress_bar = tqdm(total=len(jobs), desc="migrating subjects", unit="subj", ncols=100, ascii=True)

    progress_lock = threading.Lock()
    results: list[dict] = []
    errors: list[tuple[str, str]] = []

    def _run(job: tuple[str, Path, Optional[str], int]) -> dict:
        subject_id, path, split_name, n_windows_total = job
        return migrate_subject(
            subject_id,
            path,
            split_name,
            n_windows_total,
            pulse_pressure_range,
            args.min_valid_windows,
            args.max_reject_fraction,
            args.dry_run,
            progress_lock,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, job): job for job in jobs}
        for future in as_completed(futures):
            subject_id = futures[future][0]
            try:
                result = future.result()
                results.append(result)
                if args.verbose:
                    print(
                        f"  {result['subject_id']:<10} {result['outcome']:<8} "
                        f"{result['n_before']:>8} -> {result['n_after']:>8} windows "
                        f"(-{result['n_dropped']})"
                    )
            except Exception as exc:
                errors.append((subject_id, str(exc)))
            if progress_bar is not None:
                progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    kept = [r for r in results if r["outcome"] == "kept"]
    excluded = [r for r in results if r["outcome"] == "excluded"]
    windows_before = sum(r["n_before"] for r in results)
    windows_after = sum(r["n_after"] for r in kept)

    print()
    print(f"subjects migrated         : {len(results)} / {len(jobs)}")
    print(f"  still kept              : {len(kept)}")
    print(f"  newly excluded          : {len(excluded)}")
    if excluded:
        print(f"    e.g. {[r['subject_id'] for r in excluded[:5]]}")
    print(f"windows before             : {windows_before:,}")
    print(f"windows after (kept only)  : {windows_after:,}")
    print(f"windows dropped            : {windows_before - windows_after:,}")
    if missing:
        print(f"missing npz (in progress but no file found): {len(missing)} e.g. {missing[:5]}")
    print(f"errors                     : {len(errors)}")
    if errors:
        for subject_id, msg in errors[:10]:
            print(f"  {subject_id}: {msg}")
    if args.dry_run:
        print("\n--dry-run: no files were written or deleted.")


if __name__ == "__main__":
    main()
