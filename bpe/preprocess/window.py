"""Fixed-length windowing of resampled signals.

PPG and ABP channels from the same WFDB segment share length and sample
rate, so calling this independently on each with identical `fs`/`window_sec`
/`stride_sec` always yields the same number of windows at the same offsets --
no separate index-alignment step is needed.
"""

from __future__ import annotations

import numpy as np


def window_signal(x: np.ndarray, fs: float, window_sec: float, stride_sec: float) -> np.ndarray:
    """Slice a 1D signal into fixed-length windows of shape
    `(n_windows, round(window_sec * fs))`. A trailing partial window is
    dropped rather than padded."""
    x = np.asarray(x, dtype=float)
    window_len = int(round(window_sec * fs))
    stride_len = int(round(stride_sec * fs))
    if window_len <= 0 or stride_len <= 0:
        raise ValueError("window_sec and stride_sec must be positive")
    if len(x) < window_len:
        return np.empty((0, window_len), dtype=float)
    n_windows = 1 + (len(x) - window_len) // stride_len
    starts = np.arange(n_windows) * stride_len
    return np.stack([x[start : start + window_len] for start in starts])
