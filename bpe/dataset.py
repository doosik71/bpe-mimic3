"""PyTorch Dataset classes over data/dataset/{split}/{subject_id}.npz.

Both the calibration-free and calibration-based (Siamese) training loops
read from the same underlying per-subject arrays; only how a window is
packaged into a batch differs -- see CalibrationFreeDataset and
CalibrationPairDataset below.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_DATASET_DIR = Path("data/dataset")


class SubjectArrays(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    calib_x: np.ndarray
    calib_y: np.ndarray
    fs: float


def load_split(dataset_dir: Path, split: str) -> dict[str, SubjectArrays]:
    """Eagerly load every subject npz in one split into memory. Dataset
    sizes here (a few hundred subjects x a few hundred windows) are small
    enough that this is simpler and fast enough than lazy per-window I/O."""
    from tqdm import tqdm

    split_dir = Path(dataset_dir) / split
    paths = sorted(split_dir.glob("*.npz"))
    print(f"loading {len(paths)} subject(s) from {split_dir}...")
    subjects: dict[str, SubjectArrays] = {}
    for path in tqdm(paths, desc=f"loading {split}", unit="subj", ncols=100, ascii=True):
        with np.load(path) as data:
            subjects[path.stem] = SubjectArrays(
                x=data["x"],
                y=data["y"],
                calib_x=data["calib_x"],
                calib_y=data["calib_y"],
                fs=float(data["fs"]),
            )
    return subjects


def _build_window_index(subjects: dict[str, SubjectArrays]) -> list[tuple[str, int]]:
    index: list[tuple[str, int]] = []
    for subject_id, arrays in subjects.items():
        for i in range(arrays.x.shape[0]):
            index.append((subject_id, i))
    return index


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-window z-score normalization. WFDB records use different ADC
    gain configurations per patient (seen directly in the .hea files), so
    raw PPG amplitude is not comparable across subjects without this."""
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


class _WindowDatasetBase(Dataset):
    def __init__(self, dataset_dir: Path = DEFAULT_DATASET_DIR, split: str = "train", normalize: bool = True):
        self.subjects = load_split(dataset_dir, split)
        self.index = _build_window_index(self.subjects)
        self.normalize = normalize
        if not self.index:
            raise ValueError(f"no windows found in {Path(dataset_dir) / split}")

    def __len__(self) -> int:
        return len(self.index)

    def _x(self, arrays: SubjectArrays, local_idx: int) -> torch.Tensor:
        x = torch.from_numpy(arrays.x[local_idx])
        return _normalize(x) if self.normalize else x

    def _calib_x(self, arrays: SubjectArrays) -> torch.Tensor:
        calib_x = torch.from_numpy(arrays.calib_x)
        return _normalize(calib_x) if self.normalize else calib_x


class CalibrationFreeDataset(_WindowDatasetBase):
    """Yields `(x, y)` -- a PPG window and its `[SBP, DBP]` label."""

    def __getitem__(self, i: int):
        subject_id, local_idx = self.index[i]
        arrays = self.subjects[subject_id]
        x = self._x(arrays, local_idx)
        y = torch.from_numpy(arrays.y[local_idx])
        return x, y


class CalibrationPairDataset(_WindowDatasetBase):
    """Yields `(x, y, calib_x, calib_y)` -- a PPG window and its label,
    paired with that patient's calibration window and calibration BP."""

    def __getitem__(self, i: int):
        subject_id, local_idx = self.index[i]
        arrays = self.subjects[subject_id]
        x = self._x(arrays, local_idx)
        y = torch.from_numpy(arrays.y[local_idx])
        calib_x = self._calib_x(arrays)
        calib_y = torch.from_numpy(arrays.calib_y)
        return x, y, calib_x, calib_y
