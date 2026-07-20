"""End-to-end per-patient dataset construction.

Reads the segments listed in data/mimic3_index.csv (built by
build-mimic3-index), applies resample -> window -> label -> QC per
docs/development-plan.md §4, aggregates per patient across all of their
records in chronological order, applies the patient-level exclusion and
outlier rules, and writes one npz per surviving patient into
data/dataset/{train,val,test}/.
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
    DEFAULT_SBP_RANGE,
    is_periodic,
    physiological_range_ok,
)
from bpe.preprocess.resample import resample_signal
from bpe.preprocess.window import window_signal

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path("data/dataset")
DEFAULT_TARGET_FS = 100.0
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
    ppg_periodicity_threshold: float = DEFAULT_PPG_PERIODICITY_THRESHOLD,
    abp_periodicity_threshold: float = DEFAULT_ABP_PERIODICITY_THRESHOLD,
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
            labels = compute_sbp_dbp(bp_win, target_fs)
            if labels is None:
                continue
            sbp, dbp = labels
            if not physiological_range_ok(sbp, dbp, sbp_range, dbp_range):
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
    calib_idx = calibration_index(keep_mask)
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

    return PatientResult(subject_id=subject_id, x=x, y=y, calib_x=calib_x, calib_y=calib_y), n_total


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


def write_patient_npz(result: PatientResult, output_dir: Path, split_name: str) -> Path:
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    out_path = split_dir / f"{result.subject_id}.npz"
    np.savez(out_path, x=result.x, y=result.y, calib_x=result.calib_x, calib_y=result.calib_y)
    return out_path


def build_dataset(
    mimic3_dir: Path,
    index_csv: Path,
    output_dir: Path = DEFAULT_DATASET_DIR,
    target_fs: float = DEFAULT_TARGET_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    sbp_range: tuple[float, float] = DEFAULT_SBP_RANGE,
    dbp_range: tuple[float, float] = DEFAULT_DBP_RANGE,
    ppg_periodicity_threshold: float = DEFAULT_PPG_PERIODICITY_THRESHOLD,
    abp_periodicity_threshold: float = DEFAULT_ABP_PERIODICITY_THRESHOLD,
    min_valid_windows: int = DEFAULT_MIN_VALID_WINDOWS,
    max_reject_fraction: float = DEFAULT_MAX_REJECT_FRACTION,
    max_bp_deviation: float = DEFAULT_MAX_BP_DEVIATION,
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = DEFAULT_SEED,
    limit_subjects: Optional[int] = None,
    workers: int = 8,
    show_progress: bool = True,
) -> dict:
    """Process every indexed subject and write data/dataset/{train,val,test}.
    Returns a summary dict; never writes to `mimic3_dir`."""
    mimic3_dir = Path(mimic3_dir)
    by_subject = read_index_csv(index_csv)
    subject_ids = sorted(by_subject.keys())
    if limit_subjects is not None:
        subject_ids = subject_ids[:limit_subjects]

    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=len(subject_ids), desc="processing patients", unit="pt")

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
            ppg_periodicity_threshold=ppg_periodicity_threshold,
            abp_periodicity_threshold=abp_periodicity_threshold,
            min_valid_windows=min_valid_windows,
            max_reject_fraction=max_reject_fraction,
            max_bp_deviation=max_bp_deviation,
        )

    results: dict[str, PatientResult] = {}
    errors: list[tuple[str, str]] = []
    total_windows_attempted = 0

    def _handle(subject_id: str, result: Optional[PatientResult], n_total: int) -> None:
        nonlocal total_windows_attempted
        total_windows_attempted += n_total
        if result is not None:
            results[subject_id] = result

    if workers <= 1:
        for subject_id in subject_ids:
            try:
                result, n_total = _process(subject_id)
                _handle(subject_id, result, n_total)
            except Exception as exc:
                errors.append((subject_id, str(exc)))
                logger.warning("failed to process subject %s", subject_id, exc_info=True)
            if progress is not None:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, sid): sid for sid in subject_ids}
            for future in as_completed(futures):
                subject_id = futures[future]
                try:
                    result, n_total = future.result()
                    _handle(subject_id, result, n_total)
                except Exception as exc:
                    errors.append((subject_id, str(exc)))
                    logger.warning("failed to process subject %s", subject_id, exc_info=True)
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    splits = split_subjects(list(results.keys()), split=split, seed=seed)

    windows_by_split: dict[str, int] = {}
    for split_name, ids in splits.items():
        n_windows = 0
        for subject_id in ids:
            result = results[subject_id]
            write_patient_npz(result, output_dir, split_name)
            n_windows += result.x.shape[0]
        windows_by_split[split_name] = n_windows

    total_windows_kept = sum(r.x.shape[0] for r in results.values())

    return {
        "subjects_scanned": len(subject_ids),
        "subjects_kept": len(results),
        "subjects_by_split": {k: len(v) for k, v in splits.items()},
        "windows_by_split": windows_by_split,
        "total_windows_attempted": total_windows_attempted,
        "total_windows_kept": total_windows_kept,
        "retention_rate": (total_windows_kept / total_windows_attempted) if total_windows_attempted else 0.0,
        "errors": errors,
    }
