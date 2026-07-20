"""STFT spectrogram computation shared by the calibration-free CNN's input
pipeline (docs/development-plan.md §5.1) and the dataset inspection tooling.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram

DEFAULT_SUB_WINDOW_SEC = 1.0  # docs/development-plan.md §1 decision
DEFAULT_OVERLAP_FRACTION = 0.95


def compute_spectrogram(
    x: np.ndarray,
    fs: float,
    sub_window_sec: float = DEFAULT_SUB_WINDOW_SEC,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(frequencies, times, power)` from a Hamming-windowed STFT."""
    nperseg = max(1, int(round(sub_window_sec * fs)))
    noverlap = min(nperseg - 1, int(round(nperseg * overlap_fraction)))
    freqs, times, power = spectrogram(
        np.asarray(x, dtype=float),
        fs=fs,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    return freqs, times, power


def power_to_db(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(power + eps)
