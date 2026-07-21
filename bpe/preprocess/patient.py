"""Patient-level QC: drop patients with too little surviving signal, and
drop per-patient outlier windows relative to their first valid window.
See docs/development-plan.md §4 steps 7-9.
"""

from __future__ import annotations

import statistics
from typing import Optional, Sequence

# The paper requires >=100 valid 30 s windows per patient; scaled to 8 s
# windows at roughly equivalent usable duration (~375). This is a starting
# point to validate empirically via dataset-statistic, not a fixed constant.
DEFAULT_MIN_VALID_WINDOWS = 375
DEFAULT_MAX_REJECT_FRACTION = 0.95
DEFAULT_MAX_BP_DEVIATION = 40.0


def should_exclude_patient(
    n_valid: int,
    n_total: int,
    min_valid_windows: int = DEFAULT_MIN_VALID_WINDOWS,
    max_reject_fraction: float = DEFAULT_MAX_REJECT_FRACTION,
) -> bool:
    """True if the patient has too few valid windows, or too high a
    fraction of their windows were rejected, to trust the remaining
    signal."""
    if n_total == 0:
        return True
    reject_fraction = 1.0 - (n_valid / n_total)
    return n_valid < min_valid_windows or reject_fraction > max_reject_fraction


def outlier_keep_mask(
    labels: Sequence[tuple[float, float]],
    max_deviation: float = DEFAULT_MAX_BP_DEVIATION,
) -> list[bool]:
    """`labels` must already be range/periodicity-valid windows in
    chronological order. The first window is the patient's reference point
    for outlier rejection (kept unconditionally, since it can't deviate from
    itself) -- a separate concern from which window becomes the calibration
    pair, see `calibration_index`."""
    if not labels:
        return []
    ref_sbp, ref_dbp = labels[0]
    return [
        abs(sbp - ref_sbp) <= max_deviation and abs(dbp - ref_dbp) <= max_deviation
        for sbp, dbp in labels
    ]


def calibration_index(
    labels: Sequence[tuple[float, float]],
    keep_mask: Sequence[bool],
) -> Optional[int]:
    """Index of the kept window whose (SBP, DBP) is closest to the patient's
    own median BP, used as the calibration anchor. Picking a "representative"
    window this way avoids anchoring the Siamese model's calibration pair on
    the chronologically-first window, which can itself be an atypical
    reading (e.g. taken during an unstable admission period) rather than
    typical of the patient. It is also kept in the regular training pool
    (docs/development-plan.md's calibration-window-reuse decision).

    Note this is independent of the outlier-removal reference point in
    `outlier_keep_mask`, which still anchors on the chronologically-first
    window -- only the calibration *pair* selection changes here.
    """
    kept_indices = [i for i, keep in enumerate(keep_mask) if keep]
    if not kept_indices:
        return None
    median_sbp = statistics.median(labels[i][0] for i in kept_indices)
    median_dbp = statistics.median(labels[i][1] for i in kept_indices)
    return min(
        kept_indices,
        key=lambda i: abs(labels[i][0] - median_sbp) + abs(labels[i][1] - median_dbp),
    )
