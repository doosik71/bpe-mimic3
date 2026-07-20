"""Clinical accuracy metrics for BP estimation: basic error statistics,
BHS cumulative-error grading, and AAMI pass/fail -- the same standards
docs/method.md benchmarks its own MAD/STD results against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AAMI_MAX_ME = 5.0
AAMI_MAX_SD = 8.0

# grade -> (% within 5 mmHg, % within 10 mmHg, % within 15 mmHg) required
BHS_THRESHOLDS = {
    "A": (60.0, 85.0, 95.0),
    "B": (50.0, 75.0, 90.0),
    "C": (40.0, 65.0, 85.0),
}


@dataclass
class ErrorStats:
    me: float  # mean error (pred - true)
    mae: float  # mean absolute error
    rmse: float
    sd: float  # sample std of the signed error


def compute_error_stats(pred: np.ndarray, true: np.ndarray) -> ErrorStats:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    error = pred - true
    return ErrorStats(
        me=float(np.mean(error)),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        sd=float(np.std(error, ddof=1)) if len(error) > 1 else 0.0,
    )


def bhs_cumulative_percentages(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    """Return `(% within 5 mmHg, % within 10 mmHg, % within 15 mmHg)`."""
    abs_error = np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float))
    if len(abs_error) == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(abs_error <= 5.0) * 100.0),
        float(np.mean(abs_error <= 10.0) * 100.0),
        float(np.mean(abs_error <= 15.0) * 100.0),
    )


def bhs_grade(pred: np.ndarray, true: np.ndarray) -> str:
    pct5, pct10, pct15 = bhs_cumulative_percentages(pred, true)
    for grade, (t5, t10, t15) in BHS_THRESHOLDS.items():
        if pct5 >= t5 and pct10 >= t10 and pct15 >= t15:
            return grade
    return "D"


def aami_pass(stats: ErrorStats) -> bool:
    return abs(stats.me) <= AAMI_MAX_ME and stats.sd <= AAMI_MAX_SD
