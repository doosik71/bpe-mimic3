"""Window-level validity checks from docs/method.md: physiological
plausibility of the labeled BP, and periodicity (noise) of the raw signal.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SBP_RANGE = (75.0, 165.0)
DEFAULT_DBP_RANGE = (40.0, 85.0)


def physiological_range_ok(
    sbp: float,
    dbp: float,
    sbp_range: tuple[float, float] = DEFAULT_SBP_RANGE,
    dbp_range: tuple[float, float] = DEFAULT_DBP_RANGE,
) -> bool:
    return sbp_range[0] <= sbp <= sbp_range[1] and dbp_range[0] <= dbp <= dbp_range[1]


def normalized_autocorrelation(x: np.ndarray) -> np.ndarray:
    """One-sided autocorrelation (lags 0..N-1) of the DC-removed signal,
    normalized so the lag-0 value is 1."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")
    ac = full[len(x) - 1 :]
    if ac[0] == 0:
        return np.zeros_like(ac)
    return ac / ac[0]


def periodicity_score(x: np.ndarray) -> float:
    """Area under the squared magnitude of the autocorrelation (lag 1
    onward -- the trivial lag-0 peak is excluded), normalized by the
    number of lags so the score is comparable across window lengths.
    A periodic signal (clean pulsatile PPG/ABP) keeps a high autocorrelation
    across many lags; noise decays to ~0 almost immediately.
    """
    ac = normalized_autocorrelation(x)
    if len(ac) <= 1:
        return 0.0
    return float(np.trapezoid(ac[1:] ** 2) / (len(ac) - 1))


def is_periodic(x: np.ndarray, threshold: float) -> bool:
    return periodicity_score(x) >= threshold
