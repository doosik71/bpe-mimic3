"""Resample WFDB signals to the project's target sample rate (100 Hz)."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample a 1D signal from `orig_fs` to `target_fs` via polyphase
    filtering. `orig_fs`/`target_fs` are reduced to a small integer ratio
    (125 -> 100 Hz reduces exactly to up=4, down=5)."""
    if orig_fs == target_fs:
        return np.asarray(x, dtype=float)
    orig_fs_i = int(round(orig_fs))
    target_fs_i = int(round(target_fs))
    divisor = gcd(orig_fs_i, target_fs_i)
    up = target_fs_i // divisor
    down = orig_fs_i // divisor
    return resample_poly(np.asarray(x, dtype=float), up, down)
