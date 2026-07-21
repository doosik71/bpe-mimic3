"""Per-window SBP/DBP labeling from an arterial BP window.

Follows docs/method-spectrogram-cnn.md: SBP/DBP are the mean of the window's systolic
peaks / diastolic troughs, not a single instantaneous reading.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks


def compute_sbp_dbp(
    abp_window: np.ndarray,
    fs: float,
    min_beat_interval_sec: float = 0.3,
    min_beats: int = 3,
) -> Optional[tuple[float, float]]:
    """Return `(SBP, DBP)` in mmHg, or `None` if fewer than `min_beats`
    systolic peaks or diastolic troughs were detected (too little rhythmic
    content to average reliably). `min_beat_interval_sec` caps the maximum
    plausible heart rate (default 0.3 s -> 200 bpm) to reject spurious
    peaks from high-frequency noise.
    """
    abp_window = np.asarray(abp_window, dtype=float)
    min_distance = max(1, int(round(min_beat_interval_sec * fs)))
    peak_idx, _ = find_peaks(abp_window, distance=min_distance)
    trough_idx, _ = find_peaks(-abp_window, distance=min_distance)
    if len(peak_idx) < min_beats or len(trough_idx) < min_beats:
        return None
    sbp = float(np.mean(abp_window[peak_idx]))
    dbp = float(np.mean(abp_window[trough_idx]))
    return sbp, dbp
