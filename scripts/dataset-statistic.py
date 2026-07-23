"""Analyze the constructed BP dataset end to end: the pre-QC segment index,
the per-subject QC retention ledger, and the final train/val/test npz
splits. See docs/dataset-analysis.md for the full catalogue of analyses
this was planned from (and the ones deferred to later, heavier tooling).

Reads:
    data/mimic3_index.csv           pre-QC segment index (optional -- skipped
                                     if not found; build with
                                     `run build-mimic3-index`)
    data/dataset/_progress.csv      per-subject QC outcome ledger
    data/dataset/{train,val,test}/*.npz   final PyTorch-ready windows

Writes, under --output-dir (default: --dataset-dir):
    statistic.json          numerical summary of every analysis below
    index_overview.png       segments/duration per subject, BP channel naming
                             (skipped if the index csv isn't found)
    retention_overview.png   kept/excluded counts, per-subject retention rate,
                             attempted-window counts for kept vs. excluded
    bp_distribution.png     SBP / DBP / pulse-pressure density per split
    windows_per_subject.png windows-per-subject concentration check per split
    bp_sd_per_subject.png   within-subject SBP/DBP variability per split
    calibration_offset.png  calibration-window BP vs. subject's own mean BP
    ppg_amplitude.png       per-window PPG std vs. the min_ppg_std QC gate
                             (skipped with --skip-ppg-amplitude)

Deferred (not implemented here -- see docs/dataset-analysis.md):
    #10 exclusion-reason breakdown (too few windows vs. too noisy) --
        _progress.csv always logs excluded subjects with n_windows_kept=0,
        so the true n_valid used for the check isn't recoverable from the
        ledger alone.
    #12 retention-rate-vs-threshold sensitivity -- needs re-running the
        per-window QC filters (bpe/preprocess/quality.py) against raw
        signals, not just reading stored outputs.
    #22 superimposed pulse-shape consistency -- a visual/qualitative check,
        better suited to `run dataset-browser`.

Usage:
    uv run python scripts/dataset-statistic.py
    uv run python scripts/dataset-statistic.py --dataset-dir data/dataset --limit-subjects 50
    uv run python scripts/dataset-statistic.py --skip-ppg-amplitude --no-plots
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bpe.io.mimic3 import DEFAULT_INDEX_CSV
from bpe.preprocess.patient import DEFAULT_MAX_REJECT_FRACTION, DEFAULT_MIN_VALID_WINDOWS
from bpe.preprocess.pipeline import DEFAULT_DATASET_DIR, DEFAULT_STRIDE_SEC, DEFAULT_WINDOW_SEC, read_progress
from bpe.preprocess.quality import DEFAULT_MIN_PPG_STD
from bpe.reporting import print_run_info

SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute dataset statistics and generate distribution plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Root dataset directory, holding _progress.csv and train/val/test/ (default: %(default)s)")
    p.add_argument("--index-csv", type=Path, default=DEFAULT_INDEX_CSV, help="Pre-QC segment index csv; skipped if not found (default: %(default)s)")
    p.add_argument("--output-dir", type=Path, default=None, help="Where statistic.json and *.png are written (default: --dataset-dir)")
    p.add_argument("--limit-subjects", type=int, default=None, help="Only load the first N subjects per split (for a quick trial run; default: no limit)")
    p.add_argument("--skip-ppg-amplitude", action="store_true", help="Skip the per-window PPG amplitude check -- avoids loading `x`, the most expensive part of this script")
    p.add_argument("--no-plots", action="store_true", help="Only write statistic.json; skip generating PNG plots")
    p.add_argument("--workers", type=int, default=8, help="Thread-pool size for concurrent npz loading (default: %(default)s)")
    p.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bars")
    return p.parse_args()


# -- Generic stat helpers -------------------------------------------------------

def _summary_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "p25": None, "p50": None, "p75": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _concentration_stats(counts: np.ndarray, subject_ids: list[str]) -> dict:
    """Generalizes _summary_stats with a "does a few subjects dominate the
    total?" view: max/median ratio and how much of the total the top 10% of
    subjects hold."""
    stats = _summary_stats(counts)
    n_total = int(counts.sum())
    median = float(np.median(counts)) if counts.size else 0.0
    order = np.argsort(counts)[::-1]
    top10_n = max(1, int(np.ceil(len(counts) * 0.10)))
    top10_sum = int(counts[order[:top10_n]].sum()) if counts.size else 0
    stats.update(
        {
            "max_to_median_ratio": round(float(counts.max()) / median, 1) if median > 0 else None,
            "top10pct_subjects_hold_pct_windows": round(top10_sum / n_total * 100, 1) if n_total else 0.0,
            "top5_subjects": [{"subject_id": subject_ids[i], "n_windows": int(counts[i])} for i in order[:5]],
        }
    )
    return stats


# -- 1. Pre-QC segment index (data/mimic3_index.csv) ----------------------------

def load_index_stats(index_csv: Path) -> Optional[dict]:
    """Pre-QC view over every WFDB segment that exposes both PLETH and an
    arterial pressure channel. Cheap: pandas over one flat csv, no WFDB
    access needed."""
    if not index_csv.is_file():
        return None

    df = pd.read_csv(index_csv)
    seg_duration_sec = (df["n_samples"] / df["fs"]).to_numpy()

    co_occurring: Counter[str] = Counter()
    for sig_names, bp_signal in zip(df["sig_names"], df["bp_signal"]):
        for name in str(sig_names).split(";"):
            if name and name not in ("PLETH", bp_signal):
                co_occurring[name] += 1

    summary = {
        "n_segments": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_records": int(df.groupby(["subject_id", "record_name"]).ngroups),
        "total_candidate_duration_hr": float(seg_duration_sec.sum() / 3600.0),
        "segment_duration_sec": _summary_stats(seg_duration_sec),
        "segments_per_subject": _summary_stats(df.groupby("subject_id").size().to_numpy()),
        "records_per_subject": _summary_stats(df.groupby("subject_id")["record_name"].nunique().to_numpy()),
        "segments_per_record": _summary_stats(df.groupby(["subject_id", "record_name"]).size().to_numpy()),
        "bp_signal_counts": {str(k): int(v) for k, v in df["bp_signal"].value_counts().items()},
        "top_co_occurring_channels": co_occurring.most_common(5),
    }
    return {"summary": summary, "raw": df}


def plot_index_overview(df: pd.DataFrame, out_path: Path) -> None:
    """Segment count/duration per subject, and BP-channel naming, from the
    pre-QC index -- context for how much raw material feeds the QC pipeline
    below."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Pre-QC Segment Index Overview (data/mimic3_index.csv)", fontsize=13)

    segments_per_subject = df.groupby("subject_id").size().to_numpy()
    axes[0].hist(segments_per_subject, bins=40, color="#607D8B", edgecolor="none")
    axes[0].axvline(float(np.median(segments_per_subject)), color="red", linestyle="--", linewidth=1.2, label=f"median = {np.median(segments_per_subject):.0f}")
    axes[0].set_xlabel("Qualifying segments per subject")
    axes[0].set_ylabel("Number of subjects")
    axes[0].set_title("Segments per Subject")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    seg_duration_min = (df["n_samples"] / df["fs"] / 60.0).to_numpy()
    axes[1].hist(seg_duration_min, bins=60, color="#795548", edgecolor="none")
    axes[1].set_xlabel("Segment duration (min)")
    axes[1].set_ylabel("Number of segments")
    axes[1].set_title("Segment Duration")
    axes[1].grid(True, alpha=0.3, axis="y")

    bp_counts = df["bp_signal"].value_counts()
    axes[2].bar(bp_counts.index.astype(str), bp_counts.to_numpy(), color=["#3F51B5", "#009688"][: len(bp_counts)])
    for i, v in enumerate(bp_counts.to_numpy()):
        axes[2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[2].set_ylabel("Number of segments")
    axes[2].set_title("Arterial BP Channel Naming")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- 2. QC retention ledger (data/dataset/_progress.csv) ------------------------

def load_progress_stats(dataset_dir: Path) -> Optional[dict]:
    """QC outcome ledger for every subject the pipeline has attempted --
    who was kept/excluded and how many windows survived, independent of
    which split a kept subject later landed in."""
    rows = read_progress(dataset_dir)
    if not rows:
        return None

    kept = [r for r in rows.values() if r.status == "kept"]
    excluded = [r for r in rows.values() if r.status == "excluded"]
    n_attempted = len(rows)
    windows_total = sum(r.n_windows_total for r in rows.values())
    windows_kept = sum(r.n_windows_kept for r in kept)
    retention_per_subject = np.array(
        [r.n_windows_kept / r.n_windows_total for r in kept if r.n_windows_total > 0], dtype=np.float64
    )

    summary = {
        "n_subjects_attempted": n_attempted,
        "n_subjects_kept": len(kept),
        "n_subjects_excluded": len(excluded),
        "kept_fraction": (len(kept) / n_attempted) if n_attempted else 0.0,
        "windows_total_attempted": windows_total,
        "windows_total_kept": windows_kept,
        "window_retention_rate_overall": (windows_kept / windows_total) if windows_total else 0.0,
        "retention_rate_per_kept_subject": _summary_stats(retention_per_subject),
        "n_windows_total_kept_subjects": _summary_stats(np.array([r.n_windows_total for r in kept], dtype=np.float64)),
        "n_windows_total_excluded_subjects": _summary_stats(np.array([r.n_windows_total for r in excluded], dtype=np.float64)),
        "note_exclusion_reason_breakdown": (
            "Not computable from _progress.csv: excluded subjects are always logged with "
            "n_windows_kept=0 (the true n_valid used for the min_valid_windows / "
            "max_reject_fraction check is discarded), so 'too few windows' vs. 'too noisy' "
            "can't be told apart from the ledger alone -- see docs/dataset-analysis.md #10."
        ),
    }
    return {"summary": summary, "raw": rows}


def plot_retention_overview(rows: dict, out_path: Path) -> None:
    """Kept/excluded counts, per-subject window-retention-rate distribution,
    and whether excluded subjects tended to have short recordings or
    long-but-noisy ones."""
    kept = [r for r in rows.values() if r.status == "kept"]
    excluded = [r for r in rows.values() if r.status == "excluded"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("QC Retention Overview (data/dataset/_progress.csv)", fontsize=13)

    axes[0].bar(["kept", "excluded"], [len(kept), len(excluded)], color=["#4CAF50", "#F44336"])
    for i, v in enumerate([len(kept), len(excluded)]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Number of subjects")
    axes[0].set_title("Subject-Level Outcome")
    axes[0].grid(True, alpha=0.3, axis="y")

    retention = np.array([r.n_windows_kept / r.n_windows_total for r in kept if r.n_windows_total > 0])
    if retention.size:
        axes[1].hist(retention * 100, bins=40, color="#2196F3", edgecolor="none")
        axes[1].axvline(float(np.median(retention)) * 100, color="red", linestyle="--", linewidth=1.2, label=f"median = {np.median(retention) * 100:.1f}%")
    axes[1].axvline((1 - DEFAULT_MAX_REJECT_FRACTION) * 100, color="black", linestyle=":", linewidth=1.0, label=f"reject-fraction floor = {(1 - DEFAULT_MAX_REJECT_FRACTION) * 100:.0f}%")
    axes[1].set_xlabel("Window retention rate (%)")
    axes[1].set_ylabel("Number of kept subjects")
    axes[1].set_title("Per-Subject Retention Rate")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    kept_totals = np.array([r.n_windows_total for r in kept], dtype=np.float64)
    excl_totals = np.array([r.n_windows_total for r in excluded], dtype=np.float64)
    max_total = max(kept_totals.max(initial=1.0), excl_totals.max(initial=1.0))
    bins = np.logspace(0, np.log10(max_total + 1), 40)
    if kept_totals.size:
        axes[2].hist(kept_totals + 1, bins=bins, alpha=0.6, color="#4CAF50", label=f"kept (n={len(kept_totals)})")
    if excl_totals.size:
        axes[2].hist(excl_totals + 1, bins=bins, alpha=0.6, color="#F44336", label=f"excluded (n={len(excl_totals)})")
    axes[2].axvline(DEFAULT_MIN_VALID_WINDOWS, color="black", linestyle=":", linewidth=1.0, label=f"min_valid_windows = {DEFAULT_MIN_VALID_WINDOWS}")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Windows attempted per subject (+1, log scale)")
    axes[2].set_ylabel("Number of subjects")
    axes[2].set_title("Attempted Windows: Kept vs. Excluded")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- 3. Final npz splits ---------------------------------------------------------

def _load_one_subject(path: Path, include_ppg_amplitude: bool) -> dict:
    """Load one subject's npz and immediately reduce it to small per-subject
    scalars/arrays. `x` (the largest array in the file, up to (N, 1000)) is
    only ever reduced to a per-window std here -- if `include_ppg_amplitude`
    -- and never held onto, so memory stays bounded regardless of how many
    windows a subject contributes."""
    with np.load(path) as f:
        y = f["y"]
        calib_y = f["calib_y"]
        fs = float(f["fs"])
        ppg_std = np.std(f["x"], axis=1).astype(np.float32) if include_ppg_amplitude else None

    sbp = y[:, 0].astype(np.float64)
    dbp = y[:, 1].astype(np.float64)
    n = len(y)

    result = {
        "subject_id": path.stem,
        "n_windows": n,
        "sbp": sbp,
        "dbp": dbp,
        "sbp_sd": float(np.std(sbp, ddof=1)) if n > 1 else 0.0,
        "dbp_sd": float(np.std(dbp, ddof=1)) if n > 1 else 0.0,
        "calib_sbp_offset": float(calib_y[0] - sbp.mean()),
        "calib_dbp_offset": float(calib_y[1] - dbp.mean()),
        "fs": fs,
        "sbp_drift_corr": None,
        "dbp_drift_corr": None,
        "ppg_std": ppg_std,
    }
    # Chronological drift: correlation between a window's position in the
    # subject's (already time-ordered, per docs/data-cleaning.md §1) window
    # list and its labeled BP. Guarded against constant-BP subjects, where
    # the correlation is undefined.
    if n > 2:
        idx = np.arange(n, dtype=np.float64)
        if np.std(sbp) > 0:
            result["sbp_drift_corr"] = float(np.corrcoef(idx, sbp)[0, 1])
        if np.std(dbp) > 0:
            result["dbp_drift_corr"] = float(np.corrcoef(idx, dbp)[0, 1])
    return result


def load_split(
    split_dir: Path,
    limit_subjects: Optional[int] = None,
    include_ppg_amplitude: bool = True,
    workers: int = 8,
    show_progress: bool = True,
) -> dict:
    """Load every kept subject's npz in one split directory, in parallel."""
    npz_files = sorted(split_dir.glob("*.npz"))
    if limit_subjects is not None:
        npz_files = npz_files[:limit_subjects]
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")

    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=len(npz_files), desc=f"loading {split_dir.name}", unit="subj", ncols=90, ascii=True)

    results: list[dict] = []
    if workers <= 1:
        for path in npz_files:
            results.append(_load_one_subject(path, include_ppg_amplitude))
            if progress is not None:
                progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_load_one_subject, path, include_ppg_amplitude) for path in npz_files]
            for future in as_completed(futures):
                results.append(future.result())
                if progress is not None:
                    progress.update(1)
    if progress is not None:
        progress.close()

    ppg_std_parts = [r["ppg_std"] for r in results if r["ppg_std"] is not None]
    sbp_drift = [r["sbp_drift_corr"] for r in results if r["sbp_drift_corr"] is not None]
    dbp_drift = [r["dbp_drift_corr"] for r in results if r["dbp_drift_corr"] is not None]

    return {
        "n_subjects": len(results),
        "subject_ids": [r["subject_id"] for r in results],
        "window_counts": np.array([r["n_windows"] for r in results], dtype=np.int64),
        "sbp": np.concatenate([r["sbp"] for r in results]) if results else np.array([]),
        "dbp": np.concatenate([r["dbp"] for r in results]) if results else np.array([]),
        "sbp_sd_per_subject": np.array([r["sbp_sd"] for r in results], dtype=np.float64),
        "dbp_sd_per_subject": np.array([r["dbp_sd"] for r in results], dtype=np.float64),
        "calib_sbp_offset": np.array([r["calib_sbp_offset"] for r in results], dtype=np.float64),
        "calib_dbp_offset": np.array([r["calib_dbp_offset"] for r in results], dtype=np.float64),
        "sbp_drift_corr": np.array(sbp_drift, dtype=np.float64),
        "dbp_drift_corr": np.array(dbp_drift, dtype=np.float64),
        "fs_values": np.array([r["fs"] for r in results], dtype=np.float64),
        "ppg_std": np.concatenate(ppg_std_parts) if ppg_std_parts else np.array([], dtype=np.float32),
    }


def merge_splits(raw: dict[str, dict]) -> dict:
    """Pool every split's arrays into one "all" view, for dataset-wide
    numbers alongside the per-split breakdown."""
    return {
        "n_subjects": sum(d["n_subjects"] for d in raw.values()),
        "subject_ids": sum((d["subject_ids"] for d in raw.values()), []),
        "window_counts": np.concatenate([d["window_counts"] for d in raw.values()]),
        "sbp": np.concatenate([d["sbp"] for d in raw.values()]),
        "dbp": np.concatenate([d["dbp"] for d in raw.values()]),
        "sbp_sd_per_subject": np.concatenate([d["sbp_sd_per_subject"] for d in raw.values()]),
        "dbp_sd_per_subject": np.concatenate([d["dbp_sd_per_subject"] for d in raw.values()]),
        "calib_sbp_offset": np.concatenate([d["calib_sbp_offset"] for d in raw.values()]),
        "calib_dbp_offset": np.concatenate([d["calib_dbp_offset"] for d in raw.values()]),
        "sbp_drift_corr": np.concatenate([d["sbp_drift_corr"] for d in raw.values()]),
        "dbp_drift_corr": np.concatenate([d["dbp_drift_corr"] for d in raw.values()]),
        "fs_values": np.concatenate([d["fs_values"] for d in raw.values()]),
        "ppg_std": (
            np.concatenate([d["ppg_std"] for d in raw.values()])
            if any(d["ppg_std"].size for d in raw.values())
            else np.array([], dtype=np.float32)
        ),
    }


def compute_split_summary(
    raw: dict,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
) -> dict:
    sbp, dbp = raw["sbp"], raw["dbp"]
    pulse_pressure = sbp - dbp

    # Windows overlap (8 s window / 4 s stride, docs/data-cleaning.md §2), so
    # counting every window at its full window_sec double-counts the
    # overlapping half of each -- see docs/dataset-analysis.md §2.6. Each
    # window covers stride_sec of *new* real time on top of the previous one,
    # so window_count * stride_sec is the correct elapsed-time estimate
    # (equivalent to dividing the naive window_count * window_sec figure by
    # the overlap factor window_sec / stride_sec).
    total_retained_duration_hr = float(raw["window_counts"].sum() * stride_sec / 3600.0)

    summary = {
        "n_subjects": raw["n_subjects"],
        "n_windows": int(raw["window_counts"].sum()),
        "total_retained_duration_hr": total_retained_duration_hr,
        "sbp": _summary_stats(sbp),
        "dbp": _summary_stats(dbp),
        "pulse_pressure": _summary_stats(pulse_pressure),
        "windows_per_subject": _concentration_stats(raw["window_counts"], raw["subject_ids"]),
        "sbp_sd_per_subject": _summary_stats(raw["sbp_sd_per_subject"]),
        "dbp_sd_per_subject": _summary_stats(raw["dbp_sd_per_subject"]),
        "calib_sbp_offset_from_subject_mean": _summary_stats(raw["calib_sbp_offset"]),
        "calib_dbp_offset_from_subject_mean": _summary_stats(raw["calib_dbp_offset"]),
        "sbp_drift_corr_per_subject": _summary_stats(raw["sbp_drift_corr"]),
        "dbp_drift_corr_per_subject": _summary_stats(raw["dbp_drift_corr"]),
        "distinct_fs_values": sorted(set(raw["fs_values"].tolist())),
    }
    if raw["ppg_std"].size:
        summary["ppg_window_std"] = _summary_stats(raw["ppg_std"])
        summary["ppg_windows_below_min_std_pct"] = round(float((raw["ppg_std"] < DEFAULT_MIN_PPG_STD).mean() * 100), 3)
    return summary


def plot_bp_distribution(raw: dict[str, dict], out_path: Path) -> None:
    """Overlaid SBP / DBP / pulse-pressure density histograms across train,
    val, test -- confirms the patient-level split didn't skew any split's
    label distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle("BP Value Distribution Across Splits", fontsize=13)

    sbp_bins = np.linspace(60, 220, 80)
    dbp_bins = np.linspace(30, 140, 80)
    pp_bins = np.linspace(0, 140, 80)

    for ax, key, label, bins in [(axes[0], "sbp", "SBP", sbp_bins), (axes[1], "dbp", "DBP", dbp_bins)]:
        for split in SPLITS:
            if split not in raw:
                continue
            arr = raw[split][key]
            ax.hist(arr, bins=bins, density=True, histtype="step", linewidth=1.8, color=SPLIT_COLORS[split], label=f"{split}  (n={len(arr):,})")
        ax.set_xlabel(f"{label} (mmHg)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax = axes[2]
    for split in SPLITS:
        if split not in raw:
            continue
        pp = raw[split]["sbp"] - raw[split]["dbp"]
        ax.hist(pp, bins=pp_bins, density=True, histtype="step", linewidth=1.8, color=SPLIT_COLORS[split], label=f"{split}  (n={len(pp):,})")
    ax.set_xlabel("Pulse pressure, SBP-DBP (mmHg)")
    ax.set_ylabel("Density")
    ax.set_title("Pulse Pressure Distribution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_windows_per_subject(raw: dict[str, dict], out_path: Path) -> None:
    """Per-split histogram of how many windows each subject contributes --
    flags whether a handful of subjects dominate a split's total window
    count (relevant to per-subject vs. per-window sampling weight during
    training)."""
    present = [s for s in SPLITS if s in raw]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5))
    if len(present) == 1:
        axes = [axes]
    fig.suptitle("Windows per Subject -- Concentration Check", fontsize=13)

    for ax, split in zip(axes, present):
        counts = raw[split]["window_counts"]
        color = SPLIT_COLORS[split]

        ax.hist(counts, bins=40, color=color, alpha=0.8, edgecolor="none")

        mean_val = float(np.mean(counts))
        median_val = float(np.median(counts))
        ax.axvline(mean_val, color="black", linewidth=1.2, linestyle="--", label=f"Mean   = {mean_val:,.0f}")
        ax.axvline(median_val, color="red", linewidth=1.2, linestyle="-", label=f"Median = {median_val:,.0f}")

        top_idx = int(np.argmax(counts))
        top_subject = raw[split]["subject_ids"][top_idx]
        top_val = int(counts[top_idx])
        ax.annotate(
            f"max: {top_val:,}\n(subject {top_subject})",
            xy=(top_val, 0),
            xytext=(top_val, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1),
            fontsize=7,
            color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
            ha="right",
        )

        n_total = int(counts.sum())
        ax.set_xlabel("Windows per subject")
        ax.set_ylabel("Number of subjects")
        ax.set_title(f"{split.capitalize()}  ({len(counts):,} subjects | {n_total:,} windows)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bp_sd_per_subject(raw: dict[str, dict], out_path: Path) -> None:
    """Per-subject SBP/DBP SD sorted ascending -- within-subject variability
    plot. If a subject's within-subject SD is small, their mean BP is
    already a strong predictor, so calibration methods that learn a
    per-subject offset improve mainly by approximating that mean rather
    than by capturing true intra-subject dynamics."""
    all_sbp_sd: list[float] = []
    all_dbp_sd: list[float] = []
    all_splits: list[str] = []
    for split in SPLITS:
        if split not in raw:
            continue
        d = raw[split]
        for sbp_sd, dbp_sd in zip(d["sbp_sd_per_subject"], d["dbp_sd_per_subject"]):
            all_sbp_sd.append(sbp_sd)
            all_dbp_sd.append(dbp_sd)
            all_splits.append(split)

    all_sbp_sd = np.array(all_sbp_sd)
    all_dbp_sd = np.array(all_dbp_sd)
    all_splits = np.array(all_splits)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Within-Subject BP Variability (SD per Subject)\n- sorted ascending -", fontsize=12)

    for ax, sd_arr, bp_label in [(axes[0], all_sbp_sd, "SBP"), (axes[1], all_dbp_sd, "DBP")]:
        order = np.argsort(sd_arr)
        sorted_sd = sd_arr[order]
        sorted_splits = all_splits[order]
        xs = np.arange(len(sorted_sd))

        for split in SPLITS:
            mask = np.where(sorted_splits == split)[0]
            if len(mask) == 0:
                continue
            ax.scatter(xs[mask], sorted_sd[mask], c=SPLIT_COLORS[split], s=6, alpha=0.7, label=f"{split} (n={len(mask)})", rasterized=True)

        median_sd = float(np.median(sorted_sd))
        mean_sd = float(np.mean(sorted_sd))
        ax.axhline(median_sd, color="black", linewidth=1.2, linestyle="--", label=f"Median = {median_sd:.1f} mmHg")
        ax.axhline(mean_sd, color="dimgray", linewidth=1.0, linestyle=":", label=f"Mean   = {mean_sd:.1f} mmHg")

        threshold = 5.0  # mmHg -- rough "easy-to-calibrate" cutoff
        pct_below = float((sorted_sd < threshold).mean() * 100)
        ax.axvline(np.searchsorted(sorted_sd, threshold), color="#E91E63", linewidth=1.0, linestyle="-.", label=f"SD < {threshold:.0f} mmHg: {pct_below:.1f}% of subjects")

        ax.set_xlabel("Subject rank (sorted by SD, ascending)", fontsize=10)
        ax.set_ylabel(f"{bp_label} SD (mmHg)", fontsize=10)
        ax.set_title(f"{bp_label} Within-Subject SD", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_offset(raw: dict[str, dict], out_path: Path) -> None:
    """How far each subject's calibration-window BP sits from that same
    subject's own mean BP across all their kept windows -- checks whether
    the calibration reference (the outlier-filtered window closest to the
    subject's own median SBP/DBP, see bpe/preprocess/patient.py:calibration_index)
    is representative or a baseline outlier the Siamese model would need to
    correct hard from."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Calibration-Window BP Offset from Subject's Own Mean", fontsize=13)

    for ax, key, label in [(axes[0], "calib_sbp_offset", "SBP"), (axes[1], "calib_dbp_offset", "DBP")]:
        for split in SPLITS:
            if split not in raw:
                continue
            arr = raw[split][key]
            ax.hist(arr, bins=60, density=True, histtype="step", linewidth=1.8, color=SPLIT_COLORS[split], label=f"{split}  (n={len(arr):,})")
        ax.axvline(0.0, color="black", linewidth=1.0, linestyle=":")
        ax.set_xlabel(f"calib {label} - mean {label} (mmHg)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Calibration Offset")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ppg_amplitude(raw: dict[str, dict], out_path: Path) -> None:
    """Per-window PPG standard deviation across all kept windows -- sanity
    check that the min_ppg_std gate is doing its job and no near-flatline
    windows slipped through (see docs/data-cleaning.md's flatline case
    study)."""
    present = [s for s in SPLITS if s in raw and raw[s]["ppg_std"].size]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.logspace(np.log10(1e-5), np.log10(1.0), 100)
    for split in present:
        arr = raw[split]["ppg_std"]
        ax.hist(arr, bins=bins, density=True, histtype="step", linewidth=1.8, color=SPLIT_COLORS[split], label=f"{split}  (n={len(arr):,})")
    ax.axvline(DEFAULT_MIN_PPG_STD, color="red", linestyle="--", linewidth=1.2, label=f"min_ppg_std = {DEFAULT_MIN_PPG_STD}")
    ax.set_xscale("log")
    ax.set_xlabel("Per-window PPG std (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("PPG Window Amplitude -- min_ppg_std Gate Check")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- Main --------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    output_dir: Path = args.output_dir or dataset_dir

    print_run_info(
        "dataset-statistic",
        {
            "dataset dir": dataset_dir,
            "index csv": args.index_csv,
            "output dir": output_dir,
            "limit subjects": args.limit_subjects or "none",
            "ppg amplitude check": "skipped" if args.skip_ppg_amplitude else "enabled",
        },
    )

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        print("Run `run construct-dataset` first to build the dataset.")
        return

    report: dict = {}

    # -- 1. Pre-QC index --------------------------------------------------------
    index_result = load_index_stats(args.index_csv)
    if index_result is None:
        print(f"index csv not found at {args.index_csv}, skipping pre-QC index analysis.")
        report["index"] = None
    else:
        report["index"] = index_result["summary"]
        i = report["index"]
        print(f"index: {i['n_segments']:,} segments, {i['n_subjects']:,} subjects, {i['total_candidate_duration_hr']:.1f} h candidate signal")

    # -- 2. QC retention ledger --------------------------------------------------
    progress_result = load_progress_stats(dataset_dir)
    if progress_result is None:
        print(f"no _progress.csv found under {dataset_dir}, skipping QC retention analysis.")
        report["progress"] = None
    else:
        report["progress"] = progress_result["summary"]
        p = report["progress"]
        print(f"progress: {p['n_subjects_kept']}/{p['n_subjects_attempted']} subjects kept ({p['kept_fraction']:.1%}), window retention {p['window_retention_rate_overall']:.1%}")

    # -- 3. Final npz splits ------------------------------------------------------
    print("Loading dataset splits ...")
    raw: dict[str, dict] = {}
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"  {split}: directory not found, skipping.")
            continue
        d = load_split(
            split_dir,
            limit_subjects=args.limit_subjects,
            include_ppg_amplitude=not args.skip_ppg_amplitude,
            workers=args.workers,
            show_progress=not args.no_progress,
        )
        raw[split] = d
        print(f"  {split}: {d['n_subjects']} subjects, {int(d['window_counts'].sum()):,} windows")

    if not raw:
        print("no split data loaded under train/val/test; nothing more to report.")
    else:
        split_summary = {split: compute_split_summary(d) for split, d in raw.items()}
        split_summary["all"] = compute_split_summary(merge_splits(raw))
        report["splits"] = split_summary

        if report.get("index") is not None:
            candidate_hr = report["index"]["total_candidate_duration_hr"]
            retained_hr = split_summary["all"]["total_retained_duration_hr"]
            report["overall_yield_pct"] = round(retained_hr / candidate_hr * 100, 2) if candidate_hr else None
            print(f"overall yield: {retained_hr:.1f} h retained of {candidate_hr:.1f} h candidate ({report['overall_yield_pct']}%)")

        print()
        cols = f"  {'Split':<8}  {'Subjects':>8}  {'Windows':>12}  {'SBP mean+/-std':>16}  {'DBP mean+/-std':>16}  {'max/median':>10}  {'top10% holds':>13}"
        print(cols)
        print("-" * len(cols))
        for split in (*SPLITS, "all"):
            if split not in split_summary:
                continue
            s = split_summary[split]
            wpc = s["windows_per_subject"]
            print(
                f"  {split:<8}  {s['n_subjects']:>8}  {s['n_windows']:>12,}"
                f"  {s['sbp']['mean']:>7.1f} +/- {s['sbp']['std']:<6.1f}"
                f"  {s['dbp']['mean']:>7.1f} +/- {s['dbp']['std']:<6.1f}"
                f"  {wpc['max_to_median_ratio']:>10.1f}"
                f"  {wpc['top10pct_subjects_hold_pct_windows']:>12.1f}%"
            )
        print()

        if not args.no_plots:
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, fn in [
                ("bp_distribution.png", plot_bp_distribution),
                ("windows_per_subject.png", plot_windows_per_subject),
                ("bp_sd_per_subject.png", plot_bp_sd_per_subject),
                ("calibration_offset.png", plot_calibration_offset),
            ]:
                fn(raw, output_dir / name)
                print(f"Saved: {output_dir / name}")
            if not args.skip_ppg_amplitude:
                ppg_path = output_dir / "ppg_amplitude.png"
                plot_ppg_amplitude(raw, ppg_path)
                if ppg_path.exists():
                    print(f"Saved: {ppg_path}")

    if progress_result is not None and not args.no_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "retention_overview.png"
        plot_retention_overview(progress_result["raw"], path)
        print(f"Saved: {path}")

    if index_result is not None and not args.no_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "index_overview.png"
        plot_index_overview(index_result["raw"], path)
        print(f"Saved: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stat_path = output_dir / "statistic.json"
    stat_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {stat_path}")


if __name__ == "__main__":
    main()
