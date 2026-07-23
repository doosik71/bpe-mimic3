"""End-to-end per-patient dataset construction.

Reads the segments listed in data/mimic3_index.csv (built by
build-mimic3-index), applies resample -> window -> label -> QC per
docs/development-plan.md §4, aggregates per patient across all of their
records in chronological order, applies the patient-level exclusion and
outlier rules, and writes one npz per surviving patient into
data/dataset/{train,val,test}/.

This runs in two resumable phases so an interrupted run doesn't have to
restart from scratch:

1. `convert_dataset` processes one subject at a time and immediately writes
   its npz flat under `output_dir` (pre-split), recording every outcome
   (kept or excluded) in `output_dir/_progress.csv` as it goes. A subject
   already in that ledger is skipped on the next run.
2. `finalize_split` moves each converted (kept) subject's flat npz into
   `output_dir/{train,val,test}/` using the same deterministic
   `split_subjects`, so the result is identical to doing everything in one
   in-memory pass regardless of how many resumed runs it took to get there.

`build_dataset` runs both phases back to back for convenience.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

from bpe.preprocess.labels import compute_sbp_dbp
from bpe.preprocess.patient import (
    DEFAULT_MAX_BP_DEVIATION,
    DEFAULT_MAX_REJECT_FRACTION,
    DEFAULT_MIN_VALID_WINDOWS,
    calibration_index,
    outlier_keep_mask,
    should_exclude_patient,
)
from bpe.preprocess.quality import (
    DEFAULT_DBP_RANGE,
    DEFAULT_MIN_PPG_STD,
    DEFAULT_PULSE_PRESSURE_RANGE,
    DEFAULT_SBP_RANGE,
    has_sufficient_amplitude,
    is_periodic,
    physiological_range_ok,
    pulse_pressure_ok,
)
from bpe.preprocess.resample import resample_signal
from bpe.preprocess.window import window_signal

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path("data/dataset")
DEFAULT_TARGET_FS = 125.0  # MIMIC-III's native rate -- no resampling needed
DEFAULT_WINDOW_SEC = 8.0
DEFAULT_STRIDE_SEC = 4.0
DEFAULT_SPLIT = (0.6, 0.2, 0.2)
DEFAULT_SEED = 42

# Unvalidated starting points -- see docs/development-plan.md §7
# "QC threshold retuning". Tune once dataset-statistic (a later phase)
# reports real retention rates against these values.
DEFAULT_PPG_PERIODICITY_THRESHOLD = 0.05
DEFAULT_ABP_PERIODICITY_THRESHOLD = 0.05

_RECORD_NAME_RE = re.compile(r"^p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")


@dataclass
class IndexEntry:
    subject_id: str
    record_name: str
    record_dir: str
    segment_name: str
    segment_index: int
    sample_offset: int
    n_samples: int
    fs: float
    bp_signal: str
    sig_names: str


@dataclass
class PatientResult:
    subject_id: str
    x: np.ndarray  # (N, window_samples) float32 -- PPG windows
    y: np.ndarray  # (N, 2) float32 -- [SBP, DBP] mmHg per window
    calib_x: np.ndarray  # (window_samples,) float32
    calib_y: np.ndarray  # (2,) float32
    fs: float  # sample rate the windows were built at (target_fs)


def read_index_csv(index_csv: Path) -> dict[str, list[IndexEntry]]:
    """Group indexed segments by subject_id."""
    by_subject: dict[str, list[IndexEntry]] = {}
    with Path(index_csv).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            entry = IndexEntry(
                subject_id=row["subject_id"],
                record_name=row["record_name"],
                record_dir=row["record_dir"],
                segment_name=row["segment_name"],
                segment_index=int(row["segment_index"]),
                sample_offset=int(row["sample_offset"]),
                n_samples=int(row["n_samples"]),
                fs=float(row["fs"]),
                bp_signal=row["bp_signal"],
                sig_names=row["sig_names"],
            )
            by_subject.setdefault(entry.subject_id, []).append(entry)
    return by_subject


def _record_start_time(record_name: str) -> Optional[datetime]:
    """Parse the admission timestamp embedded in a MIMIC-III record name
    (e.g. "p000109-2142-01-14-18-53" -> 2142-01-14 18:53), used to order a
    patient's segments chronologically across multiple records/stays."""
    match = _RECORD_NAME_RE.match(record_name)
    if match is None:
        return None
    year, month, day, hour, minute = (int(g) for g in match.groups())
    return datetime(year, month, day, hour, minute)


def _segment_sort_key(entry: IndexEntry) -> tuple:
    start = _record_start_time(entry.record_name)
    offset_sec = entry.sample_offset / entry.fs
    if start is not None:
        return (0, start + timedelta(seconds=offset_sec), entry.segment_index)
    # Fallback for a record name that doesn't match the expected pattern:
    # sort after every parseable record, grouped and ordered by segment.
    return (1, entry.record_name, entry.segment_index)


def _read_segment_channels(mimic3_dir: Path, entry: IndexEntry) -> tuple[np.ndarray, np.ndarray]:
    """Read the PPG and BP channels of one data segment. Column order is
    looked up by name rather than assumed, since wfdb does not guarantee
    `channel_names` request order matches output column order."""
    path = str(Path(mimic3_dir) / entry.record_dir / entry.segment_name)
    record = wfdb.rdrecord(path, channel_names=["PLETH", entry.bp_signal])
    ppg_idx = record.sig_name.index("PLETH")
    bp_idx = record.sig_name.index(entry.bp_signal)
    return record.p_signal[:, ppg_idx], record.p_signal[:, bp_idx]


def process_patient(
    mimic3_dir: Path,
    subject_id: str,
    entries: list[IndexEntry],
    target_fs: float = DEFAULT_TARGET_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    sbp_range: tuple[float, float] = DEFAULT_SBP_RANGE,
    dbp_range: tuple[float, float] = DEFAULT_DBP_RANGE,
    pulse_pressure_range: tuple[float, float] = DEFAULT_PULSE_PRESSURE_RANGE,
    ppg_periodicity_threshold: float = DEFAULT_PPG_PERIODICITY_THRESHOLD,
    abp_periodicity_threshold: float = DEFAULT_ABP_PERIODICITY_THRESHOLD,
    min_ppg_std: float = DEFAULT_MIN_PPG_STD,
    min_valid_windows: int = DEFAULT_MIN_VALID_WINDOWS,
    max_reject_fraction: float = DEFAULT_MAX_REJECT_FRACTION,
    max_bp_deviation: float = DEFAULT_MAX_BP_DEVIATION,
) -> tuple[Optional[PatientResult], int]:
    """Run resample -> window -> label -> QC over every segment of one
    patient, in chronological order across all of their records, then
    apply the patient-level exclusion and outlier-removal rules.

    Returns `(result, n_windows_attempted)`; `result` is `None` if the
    patient is excluded, but `n_windows_attempted` is always reported so
    callers can compute an accurate overall retention rate.
    """
    ordered_entries = sorted(entries, key=_segment_sort_key)

    valid_windows: list[np.ndarray] = []
    valid_labels: list[tuple[float, float]] = []
    n_total = 0

    for entry in ordered_entries:
        try:
            ppg, bp = _read_segment_channels(mimic3_dir, entry)
        except Exception:
            logger.warning(
                "failed to read segment %s/%s for subject %s",
                entry.record_dir,
                entry.segment_name,
                subject_id,
                exc_info=True,
            )
            continue

        # NB: resampling a segment that contains NaN gaps (sensor dropouts)
        # can smear the NaNs across nearby samples via the polyphase FIR
        # filter. The per-window NaN check below still catches contaminated
        # windows; it just means a gap can cost slightly more than its own
        # width. Acceptable given the heavy attrition already expected, but
        # flagged as a tuning candidate in docs/development-plan.md §7.
        ppg = resample_signal(ppg, entry.fs, target_fs)
        bp = resample_signal(bp, entry.fs, target_fs)

        ppg_windows = window_signal(ppg, target_fs, window_sec, stride_sec)
        bp_windows = window_signal(bp, target_fs, window_sec, stride_sec)
        n_windows = min(len(ppg_windows), len(bp_windows))
        n_total += n_windows

        for i in range(n_windows):
            ppg_win = ppg_windows[i]
            bp_win = bp_windows[i]
            if np.isnan(ppg_win).any() or np.isnan(bp_win).any():
                continue
            # A disconnected/malfunctioning PPG sensor can produce a
            # near-flatline reading whose tiny quantization jitter still
            # scores as "periodic" (periodicity_score is scale-invariant),
            # so this absolute-amplitude gate is needed in addition to it.
            if not has_sufficient_amplitude(ppg_win, min_ppg_std):
                continue
            labels = compute_sbp_dbp(bp_win, target_fs)
            if labels is None:
                continue
            sbp, dbp = labels
            if not physiological_range_ok(sbp, dbp, sbp_range, dbp_range):
                continue
            if not pulse_pressure_ok(sbp, dbp, pulse_pressure_range):
                continue
            if not is_periodic(ppg_win, ppg_periodicity_threshold):
                continue
            if not is_periodic(bp_win, abp_periodicity_threshold):
                continue
            valid_windows.append(ppg_win.astype(np.float32))
            valid_labels.append((sbp, dbp))

    n_valid = len(valid_windows)
    if should_exclude_patient(n_valid, n_total, min_valid_windows, max_reject_fraction):
        return None, n_total

    keep_mask = outlier_keep_mask(valid_labels, max_bp_deviation)
    calib_idx = calibration_index(valid_labels, keep_mask)
    if calib_idx is None:
        return None, n_total  # defensive; outlier_keep_mask always keeps index 0

    calib_x = valid_windows[calib_idx]
    calib_y = np.array(valid_labels[calib_idx], dtype=np.float32)

    kept_windows = [w for w, keep in zip(valid_windows, keep_mask) if keep]
    kept_labels = [lbl for lbl, keep in zip(valid_labels, keep_mask) if keep]
    if not kept_windows:
        return None, n_total

    x = np.stack(kept_windows).astype(np.float32)
    y = np.array(kept_labels, dtype=np.float32)

    return PatientResult(subject_id=subject_id, x=x, y=y, calib_x=calib_x, calib_y=calib_y, fs=target_fs), n_total


def split_subjects(
    subject_ids: list[str],
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[str]]:
    """Deterministically shuffle and split subjects into train/val/test
    (patient-level, so no window from one subject can leak across splits)."""
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {split}")
    ids = sorted(subject_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(n * split[0]))
    n_val = int(round(n * split[1]))
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def write_patient_npz(result: PatientResult, output_dir: Path, split_name: Optional[str] = None) -> Path:
    """Write one patient's npz. With `split_name=None` (the default), it is
    written flat directly under `output_dir` -- the pre-split staging state
    `convert_dataset` produces; `finalize_split` later moves it into
    `output_dir/{split_name}/`."""
    output_dir = Path(output_dir)
    target_dir = output_dir / split_name if split_name else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{result.subject_id}.npz"
    # Uncompressed np.savez (NOT savez_compressed) is required, not just
    # preferred: the training loader memory-maps the large `x` array in place
    # so a whole split needn't be read into RAM (see bpe/dataset.py and
    # docs/construct-dataset.md), which only works when members are stored
    # uncompressed. Compression would also cost decode time on every load.
    np.savez(
        out_path,
        x=result.x,
        y=result.y,
        calib_x=result.calib_x,
        calib_y=result.calib_y,
        fs=np.float32(result.fs),
    )
    return out_path


PROGRESS_FILENAME = "_progress.csv"


@dataclass
class ProgressRow:
    subject_id: str
    status: str  # "kept" | "excluded"
    n_windows_total: int
    n_windows_kept: int


def read_progress(output_dir: Path) -> dict[str, ProgressRow]:
    """Read the resumability ledger written by `convert_dataset`: every
    subject already attempted, and its outcome."""
    path = Path(output_dir) / PROGRESS_FILENAME
    if not path.is_file():
        return {}
    rows: dict[str, ProgressRow] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["subject_id"]] = ProgressRow(
                subject_id=row["subject_id"],
                status=row["status"],
                n_windows_total=int(row["n_windows_total"]),
                n_windows_kept=int(row["n_windows_kept"]),
            )
    return rows


def _append_progress(output_dir: Path, row: ProgressRow) -> None:
    """Append one subject's outcome, opened and closed for this call alone
    (not held open across the run) so a completed row is fully flushed to
    disk before the next subject starts -- if the process is killed, at
    most the in-flight subject's work is lost, never an already-recorded
    one."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROGRESS_FILENAME
    is_new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["subject_id", "status", "n_windows_total", "n_windows_kept"])
        writer.writerow([row.subject_id, row.status, row.n_windows_total, row.n_windows_kept])


def convert_dataset(
    mimic3_dir: Path,
    index_csv: Path,
    output_dir: Path = DEFAULT_DATASET_DIR,
    target_fs: float = DEFAULT_TARGET_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    sbp_range: tuple[float, float] = DEFAULT_SBP_RANGE,
    dbp_range: tuple[float, float] = DEFAULT_DBP_RANGE,
    pulse_pressure_range: tuple[float, float] = DEFAULT_PULSE_PRESSURE_RANGE,
    ppg_periodicity_threshold: float = DEFAULT_PPG_PERIODICITY_THRESHOLD,
    abp_periodicity_threshold: float = DEFAULT_ABP_PERIODICITY_THRESHOLD,
    min_ppg_std: float = DEFAULT_MIN_PPG_STD,
    min_valid_windows: int = DEFAULT_MIN_VALID_WINDOWS,
    max_reject_fraction: float = DEFAULT_MAX_REJECT_FRACTION,
    max_bp_deviation: float = DEFAULT_MAX_BP_DEVIATION,
    limit_subjects: Optional[int] = None,
    workers: int = 8,
    show_progress: bool = True,
    force: bool = False,
) -> dict:
    """Phase 1: process every indexed subject not already converted,
    writing each kept patient's npz flat under `output_dir` (pre-split) as
    soon as it's ready, and recording every outcome (kept or excluded) in
    `output_dir/_progress.csv`.

    Resumable by construction: a subject already present in `_progress.csv`
    is skipped (pass `force=True` to reprocess everyone regardless of what
    is already recorded). Note the ledger is only valid for a fixed set of
    QC parameters -- changing e.g. the periodicity thresholds between runs
    without `force=True` mixes results computed under different settings.
    A subject that raises an exception is *not* recorded (an error is more
    likely transient than a stable outcome), so it is retried next run.
    """
    mimic3_dir = Path(mimic3_dir)
    output_dir = Path(output_dir)
    by_subject = read_index_csv(index_csv)
    subject_ids = sorted(by_subject.keys())
    if limit_subjects is not None:
        subject_ids = subject_ids[:limit_subjects]

    previous_progress = {} if force else read_progress(output_dir)
    pending_ids = [sid for sid in subject_ids if sid not in previous_progress]
    already_done = len(subject_ids) - len(pending_ids)

    print(
        f"converting {len(pending_ids)} pending subject(s) ({already_done} already done "
        f"of {len(subject_ids)} indexed) using {workers} worker(s)..."
    )

    progress_bar = None
    if show_progress:
        from tqdm import tqdm

        progress_bar = tqdm(total=len(pending_ids), desc="converting patients", unit="pt", ncols=90, ascii=True)

    def _process(subject_id: str) -> tuple[Optional[PatientResult], int]:
        return process_patient(
            mimic3_dir,
            subject_id,
            by_subject[subject_id],
            target_fs=target_fs,
            window_sec=window_sec,
            stride_sec=stride_sec,
            sbp_range=sbp_range,
            dbp_range=dbp_range,
            pulse_pressure_range=pulse_pressure_range,
            ppg_periodicity_threshold=ppg_periodicity_threshold,
            abp_periodicity_threshold=abp_periodicity_threshold,
            min_ppg_std=min_ppg_std,
            min_valid_windows=min_valid_windows,
            max_reject_fraction=max_reject_fraction,
            max_bp_deviation=max_bp_deviation,
        )

    errors: list[tuple[str, str]] = []
    n_kept = 0
    n_excluded = 0
    total_windows_attempted = 0
    total_windows_kept = 0

    def _handle(subject_id: str, result: Optional[PatientResult], n_total: int) -> None:
        nonlocal n_kept, n_excluded, total_windows_attempted, total_windows_kept
        total_windows_attempted += n_total
        if result is not None:
            # Write before recording progress: if the process dies between
            # the two, the subject is just retried next run instead of
            # being marked done with no file to show for it.
            write_patient_npz(result, output_dir, split_name=None)
            _append_progress(output_dir, ProgressRow(subject_id, "kept", n_total, result.x.shape[0]))
            n_kept += 1
            total_windows_kept += result.x.shape[0]
        else:
            _append_progress(output_dir, ProgressRow(subject_id, "excluded", n_total, 0))
            n_excluded += 1

    if workers <= 1:
        for subject_id in pending_ids:
            try:
                result, n_total = _process(subject_id)
                _handle(subject_id, result, n_total)
            except Exception as exc:
                errors.append((subject_id, str(exc)))
                logger.warning("failed to process subject %s", subject_id, exc_info=True)
            if progress_bar is not None:
                progress_bar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, sid): sid for sid in pending_ids}
            for future in as_completed(futures):
                subject_id = futures[future]
                try:
                    result, n_total = future.result()
                    _handle(subject_id, result, n_total)
                except Exception as exc:
                    errors.append((subject_id, str(exc)))
                    logger.warning("failed to process subject %s", subject_id, exc_info=True)
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    return {
        "subjects_scanned": len(subject_ids),
        "subjects_already_done": already_done,
        "subjects_processed_this_run": len(pending_ids),
        "subjects_kept_this_run": n_kept,
        "subjects_excluded_this_run": n_excluded,
        "total_windows_attempted_this_run": total_windows_attempted,
        "total_windows_kept_this_run": total_windows_kept,
        "errors": errors,
    }


def finalize_split(
    output_dir: Path = DEFAULT_DATASET_DIR,
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Phase 2: move every converted (kept) subject's flat npz from
    `output_dir/{subject_id}.npz` into `output_dir/{train,val,test}/`,
    using the same deterministic `split_subjects` as a single in-memory
    pass would (docs/development-plan.md §4 step 10) -- sorted subject IDs
    then seeded shuffle, so the assignment depends only on *which* subjects
    were kept, never on the order they were converted in. Safe to re-run:
    a subject already moved into place is left alone.
    """
    output_dir = Path(output_dir)
    progress = read_progress(output_dir)
    kept_subjects = [sid for sid, row in progress.items() if row.status == "kept"]

    print(f"splitting {len(kept_subjects)} kept subject(s) into train/val/test under {output_dir}...")

    splits = split_subjects(kept_subjects, split=split, seed=seed)

    moved = 0
    already_in_place = 0
    missing: list[str] = []
    windows_by_split: dict[str, int] = {}

    for split_name, subject_ids in splits.items():
        n_windows = 0
        for subject_id in subject_ids:
            dest = output_dir / split_name / f"{subject_id}.npz"
            if dest.is_file():
                already_in_place += 1
            else:
                src = output_dir / f"{subject_id}.npz"
                if not src.is_file():
                    missing.append(subject_id)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dest)
                moved += 1
            with np.load(dest) as data:
                n_windows += int(data["x"].shape[0])
        windows_by_split[split_name] = n_windows

    return {
        "subjects_kept": len(kept_subjects),
        "subjects_by_split": {k: len(v) for k, v in splits.items()},
        "windows_by_split": windows_by_split,
        "moved": moved,
        "already_in_place": already_in_place,
        "missing": missing,
    }


def build_dataset(
    mimic3_dir: Path,
    index_csv: Path,
    output_dir: Path = DEFAULT_DATASET_DIR,
    target_fs: float = DEFAULT_TARGET_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    sbp_range: tuple[float, float] = DEFAULT_SBP_RANGE,
    dbp_range: tuple[float, float] = DEFAULT_DBP_RANGE,
    pulse_pressure_range: tuple[float, float] = DEFAULT_PULSE_PRESSURE_RANGE,
    ppg_periodicity_threshold: float = DEFAULT_PPG_PERIODICITY_THRESHOLD,
    abp_periodicity_threshold: float = DEFAULT_ABP_PERIODICITY_THRESHOLD,
    min_ppg_std: float = DEFAULT_MIN_PPG_STD,
    min_valid_windows: int = DEFAULT_MIN_VALID_WINDOWS,
    max_reject_fraction: float = DEFAULT_MAX_REJECT_FRACTION,
    max_bp_deviation: float = DEFAULT_MAX_BP_DEVIATION,
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    limit_subjects: Optional[int] = None,
    workers: int = 8,
    show_progress: bool = True,
    force: bool = False,
    skip_split: bool = False,
    split_only: bool = False,
) -> dict:
    """Convenience wrapper: run `convert_dataset` (unless `split_only`),
    then `finalize_split` (unless `skip_split`). Never writes to
    `mimic3_dir`. See those two functions for the resumable two-phase
    design."""
    convert_summary: dict = {}
    if not split_only:
        convert_summary = convert_dataset(
            mimic3_dir,
            index_csv,
            output_dir,
            target_fs=target_fs,
            window_sec=window_sec,
            stride_sec=stride_sec,
            sbp_range=sbp_range,
            dbp_range=dbp_range,
            pulse_pressure_range=pulse_pressure_range,
            ppg_periodicity_threshold=ppg_periodicity_threshold,
            abp_periodicity_threshold=abp_periodicity_threshold,
            min_ppg_std=min_ppg_std,
            min_valid_windows=min_valid_windows,
            max_reject_fraction=max_reject_fraction,
            max_bp_deviation=max_bp_deviation,
            limit_subjects=limit_subjects,
            workers=workers,
            show_progress=show_progress,
            force=force,
        )

    split_summary: dict = {}
    if not skip_split:
        split_summary = finalize_split(output_dir, split=split, seed=seed)

    return {"convert": convert_summary, "split": split_summary}
