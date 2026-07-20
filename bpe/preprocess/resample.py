"""Resample WFDB signals to the project's target sample rate.

MIMIC-III waveforms are natively 125 Hz, matching this project's target
rate, so in practice this is usually a no-op (`resample_signal` short-
circuits when `orig_fs == target_fs`). Kept as a general utility in case a
segment ever surfaces at a different native rate.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(x: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """Resample a 1D signal from `orig_fs` to `target_fs` via polyphase
    filtering. `orig_fs`/`target_fs` are reduced to a small integer ratio;
    if they're already equal (the common case here), returns `x` unchanged."""
    if orig_fs == target_fs:
        return np.asarray(x, dtype=float)
    orig_fs_i = int(round(orig_fs))
    target_fs_i = int(round(target_fs))
    divisor = gcd(orig_fs_i, target_fs_i)
    up = target_fs_i // divisor
    down = orig_fs_i // divisor
    return resample_poly(np.asarray(x, dtype=float), up, down)
