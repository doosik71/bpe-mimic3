"""Power spectral density estimation for signal-quality inspection."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import welch


def compute_psd(x: np.ndarray, fs: float, nperseg: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Return `(frequencies, power)` via Welch's method."""
    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(len(x), 256)
    freqs, power = welch(x, fs=fs, nperseg=nperseg)
    return freqs, power
