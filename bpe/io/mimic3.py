"""WFDB helpers for scanning the MIMIC-III Waveform Database Matched Subset.

`data/mimic3` is read-only. Everything here only opens small `.hea` header
files (never `.dat` signal data) to decide which segments are worth
processing later; see docs/development-plan.md §2 and §4.
"""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Optional

import wfdb

logger = logging.getLogger(__name__)

DEFAULT_MIMIC3_DIR = Path("data/mimic3")
DEFAULT_INDEX_CSV = Path("data/mimic3_index.csv")

PPG_SIGNAL = "PLETH"
# Arterial pressure channel name varies by record; either counts as ground truth.
BP_SIGNALS = ("ABP", "ART")


@dataclass
class SegmentInfo:
    """One WFDB data segment that carries both a PPG and an arterial BP
    channel, i.e. a unit of raw data the preprocessing pipeline can later
    read and window."""

    subject_id: str
    record_name: str
    record_dir: str
    segment_name: str
    segment_index: int
    sample_offset: int
    n_samples: int
    fs: float
    bp_signal: str
    sig_names: str  # ';'-joined signal names actually present in this segment


def read_waveform_record_list(mimic3_dir: Path) -> list[str]:
    """Read `RECORDS-waveforms`, the list of waveform record paths relative
    to `mimic3_dir` (e.g. "p00/p000020/p000020-2183-04-28-17-47")."""
    records_file = Path(mimic3_dir) / "RECORDS-waveforms"
    with records_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _bp_signal_present(sig_names: Iterable[str]) -> Optional[str]:
    sig_names = set(sig_names)
    for name in BP_SIGNALS:
        if name in sig_names:
            return name
    return None


def _segment_info_if_qualifying(
    header,
    *,
    subject_id: str,
    record_name: str,
    record_dir: str,
    segment_name: str,
    segment_index: int,
    sample_offset: int,
) -> list[SegmentInfo]:
    sig_names = header.sig_name or []
    bp_signal = _bp_signal_present(sig_names)
    if PPG_SIGNAL not in sig_names or bp_signal is None:
        return []
    return [
        SegmentInfo(
            subject_id=subject_id,
            record_name=record_name,
            record_dir=record_dir,
            segment_name=segment_name,
            segment_index=segment_index,
            sample_offset=sample_offset,
            n_samples=header.sig_len,
            fs=header.fs,
            bp_signal=bp_signal,
            sig_names=";".join(sig_names),
        )
    ]


def scan_record(mimic3_dir: Path, record_rel_path: str) -> list[SegmentInfo]:
    """Return every data segment of one waveform record that carries both
    PLETH and an arterial pressure channel.

    Multi-segment (variable layout) records declare their full signal set
    in a shared zero-length "layout" segment; individual data segments may
    each carry only a subset of it, so availability is checked per segment,
    not just against the layout. The layout is still used as a cheap
    pre-filter: if it never declares both PLETH and a BP channel, no
    individual segment can either, so the whole record is skipped without
    opening every segment header.
    """
    record_path = Path(mimic3_dir) / record_rel_path
    subject_id = record_path.parent.name
    record_dir = str(record_path.parent.relative_to(mimic3_dir))
    record_name = record_path.name

    header = wfdb.rdheader(str(record_path))

    if getattr(header, "n_seg", 1) <= 1:
        return _segment_info_if_qualifying(
            header,
            subject_id=subject_id,
            record_name=record_name,
            record_dir=record_dir,
            segment_name=record_name,
            segment_index=0,
            sample_offset=0,
        )

    seg_names = header.seg_name
    seg_lens = header.seg_len

    layout_idx = next((i for i, n in enumerate(seg_lens) if n == 0), None)
    if layout_idx is not None:
        layout_header = wfdb.rdheader(str(record_path.parent / seg_names[layout_idx]))
        layout_sig_names = layout_header.sig_name or []
        if PPG_SIGNAL not in layout_sig_names or _bp_signal_present(layout_sig_names) is None:
            return []

    results: list[SegmentInfo] = []
    offset = 0
    for idx, (seg_name, seg_len) in enumerate(zip(seg_names, seg_lens)):
        if seg_name == "~" or seg_len == 0:
            # '~' marks a gap (no data); seg_len == 0 marks the shared
            # layout header itself, which carries no samples of its own.
            offset += seg_len
            continue
        try:
            seg_header = wfdb.rdheader(str(record_path.parent / seg_name))
        except Exception:
            logger.warning("failed to read segment header %s/%s", record_dir, seg_name, exc_info=True)
            offset += seg_len
            continue
        results.extend(
            _segment_info_if_qualifying(
                seg_header,
                subject_id=subject_id,
                record_name=record_name,
                record_dir=record_dir,
                segment_name=seg_name,
                segment_index=idx,
                sample_offset=offset,
            )
        )
        offset += seg_len
    return results


def write_index_csv(segments: list[SegmentInfo], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    field_names = [f.name for f in fields(SegmentInfo)]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg.__dict__)


def build_index(
    mimic3_dir: Path = DEFAULT_MIMIC3_DIR,
    output_csv: Path = DEFAULT_INDEX_CSV,
    limit: Optional[int] = None,
    workers: int = 8,
    show_progress: bool = True,
) -> dict:
    """Scan `mimic3_dir` for PLETH+BP-carrying segments and write them to
    `output_csv`. Returns a summary dict; never touches `mimic3_dir` beyond
    reading it."""
    mimic3_dir = Path(mimic3_dir)
    record_list = read_waveform_record_list(mimic3_dir)
    if limit is not None:
        record_list = record_list[:limit]

    print(f"scanning {len(record_list)} record(s) under {mimic3_dir} for PLETH+BP segments ({workers} worker(s))...")

    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=len(record_list), desc="scanning records", unit="rec", ncols=90, ascii=True)

    all_segments: list[SegmentInfo] = []
    errors: list[tuple[str, str]] = []

    def _scan_and_report(rel_path: str) -> list[SegmentInfo]:
        try:
            return scan_record(mimic3_dir, rel_path)
        except Exception as exc:
            errors.append((rel_path, str(exc)))
            logger.warning("failed to scan record %s", rel_path, exc_info=True)
            return []

    if workers <= 1:
        for rel_path in record_list:
            all_segments.extend(_scan_and_report(rel_path))
            if progress is not None:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_scan_and_report, rel_path): rel_path for rel_path in record_list}
            for future in as_completed(futures):
                all_segments.extend(future.result())
                if progress is not None:
                    progress.update(1)

    if progress is not None:
        progress.close()

    write_index_csv(all_segments, output_csv)

    qualifying_subjects = {s.subject_id for s in all_segments}
    qualifying_records = {(s.subject_id, s.record_name) for s in all_segments}
    total_duration_hr = sum(s.n_samples / s.fs for s in all_segments) / 3600.0

    return {
        "records_scanned": len(record_list),
        "qualifying_segments": len(all_segments),
        "qualifying_records": len(qualifying_records),
        "qualifying_subjects": len(qualifying_subjects),
        "total_duration_hr": total_duration_hr,
        "errors": errors,
    }
