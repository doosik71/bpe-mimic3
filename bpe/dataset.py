"""PyTorch Dataset classes over data/dataset/{split}/{subject_id}.npz.

Both the calibration-free and calibration-based (Siamese) training loops
read from the same underlying per-subject arrays; only how a window is
packaged into a batch differs -- see CalibrationFreeDataset and
CalibrationPairDataset below.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.lib import format as _npy_format
import torch
from torch.utils.data import Dataset

DEFAULT_DATASET_DIR = Path("data/dataset")


class SubjectArrays(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    calib_x: np.ndarray
    calib_y: np.ndarray
    fs: float


def _memmap_npz_array(path: Path, name: str) -> np.memmap:
    """Memory-map one array stored *uncompressed* inside an .npz archive,
    without reading its bytes into RAM.

    The dataset is written with plain (uncompressed) ``np.savez`` -- see
    bpe/preprocess/pipeline.py:write_patient_npz and docs/construct-dataset.md
    -- which lays each member's .npy bytes out contiguously in the zip, so
    they can be addressed by file offset and mmap'd in place. This is
    deliberately hand-rolled: ``np.load(path, mmap_mode='r')`` does *not*
    memory-map .npz members (it reads the whole array into memory), which is
    exactly the behavior that made loading a whole split exhaust RAM when
    several training runs ran at once.
    """
    member = f"{name}.npy"
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(
                f"'{member}' in {path} is compressed (compress_type="
                f"{info.compress_type}); the dataset must be written with "
                f"uncompressed np.savez so each window can be memory-mapped "
                f"instead of read into RAM (see docs/construct-dataset.md). "
                f"Rebuild the dataset with construct-dataset."
            )
        header_offset = info.header_offset

    # Skip past the zip *local* file header (30 fixed bytes + filename +
    # extra field) to where the member's .npy payload begins, then parse the
    # .npy header to recover shape/dtype and the offset of the raw array data.
    with open(path, "rb") as f:
        f.seek(header_offset)
        local = f.read(30)
        if local[:4] != b"PK\x03\x04":
            raise RuntimeError(f"{path}: not a zip local file header for '{member}'")
        name_len, extra_len = struct.unpack("<HH", local[26:30])
        f.seek(header_offset + 30 + name_len + extra_len)
        version = _npy_format.read_magic(f)
        if version == (1, 0):
            shape, fortran_order, dtype = _npy_format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran_order, dtype = _npy_format.read_array_header_2_0(f)
        else:
            raise RuntimeError(f"{path}: unsupported .npy version {version} for '{member}'")
        data_offset = f.tell()

    # np.memmap aligns `offset` down to mmap.ALLOCATIONGRANULARITY internally
    # (4096 on Linux, 65536 on Windows), so an unaligned data_offset is fine
    # and this works identically on both platforms.
    return np.memmap(
        path, mode="r", dtype=dtype, shape=shape, offset=data_offset,
        order="F" if fortran_order else "C",
    )


def _raise_open_file_limit(n_needed: int) -> None:
    """Raise this process's soft open-file limit toward its hard limit when a
    split needs more file handles than currently allowed. Each subject's
    memory-map holds one open handle for the run's lifetime, and a large
    split (train has ~1541 subjects) easily exceeds the common default soft
    limit of 1024 -- which varies per shell/session (a login shell, tmux
    pane, or systemd service may differ), so relying on the caller to run
    `ulimit -n` first is fragile. A process may always raise its own soft
    limit up to the hard limit without privileges.

    No-op on platforms without `resource` (Windows), which don't impose the
    low default that makes this necessary."""
    try:
        import resource
    except ImportError:
        return
    want = n_needed + 256  # headroom for the interpreter's own fds, torch, tqdm, ...
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= want:
        return
    # Raise generously (toward the hard limit), not just to this split's
    # `want`: a training run keeps *several* splits open at once (train + val),
    # so the process's total handle count is the sum across splits. Sizing to
    # one split alone leaves the soft limit too low for the others and hits
    # EMFILE mid-run. 1<<20 is far above any realistic multi-split total.
    ceiling = max(want, 1 << 20)
    target = ceiling if hard == resource.RLIM_INFINITY else min(hard, ceiling)
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            pass  # best effort; the EMFILE handler in load_split explains if it wasn't enough


def load_split(dataset_dir: Path, split: str) -> dict[str, SubjectArrays]:
    """Load every subject in one split, memory-mapping the big per-window
    ``x`` array (see _memmap_npz_array) instead of reading it into RAM. The
    small per-subject arrays (``y``, ``calib_x``, ``calib_y``, ``fs`` -- a
    few KB each) are read eagerly. This keeps resident memory per process to
    a few GB regardless of split size, so several training runs can share
    the OS page cache instead of each copying the whole split into RAM."""
    from tqdm import tqdm

    split_dir = Path(dataset_dir) / split
    paths = sorted(split_dir.glob("*.npz"))
    _raise_open_file_limit(len(paths))
    print(f"loading {len(paths)} subject(s) from {split_dir}...")
    subjects: dict[str, SubjectArrays] = {}
    for path in tqdm(paths, desc=f"loading {split}", unit="subj", ncols=100, ascii=True):
        try:
            x = _memmap_npz_array(path, "x")
            with np.load(path) as data:
                subjects[path.stem] = SubjectArrays(
                    x=x,
                    y=data["y"],
                    calib_x=data["calib_x"],
                    calib_y=data["calib_y"],
                    fs=float(data["fs"]),
                )
        except OSError as exc:
            # One memory-map (and thus one open file handle) is held per
            # subject for the run's lifetime, and a training run keeps every
            # loaded split open at once (train + val together), so the limit
            # that matters is the *sum* across splits. load_split already
            # raised the soft limit toward the hard limit, so reaching EMFILE
            # means the hard limit itself is too low for that total -- raising
            # it needs admin action (limits.conf / systemd LimitNOFILE).
            import errno

            if getattr(exc, "errno", None) == errno.EMFILE:
                limits = "unknown"
                try:
                    import resource

                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    limits = f"soft={soft}, hard={hard}"
                except ImportError:
                    pass
                raise RuntimeError(
                    f"too many open files while loading {path}: each subject "
                    f"holds one memory-map handle, this split has {len(paths)} "
                    f"subjects, and other splits already loaded stay open too "
                    f"(current open-file limit: {limits}). Raise the hard "
                    f"limit (e.g. systemd LimitNOFILE= or "
                    f"/etc/security/limits.conf) -- see docs/train-model.md."
                ) from exc
            raise RuntimeError(f"failed to load subject file {path}: {exc}") from exc
        except Exception as exc:
            # Name the offending file so a corrupt/truncated npz doesn't
            # surface as an opaque error with no clue which subject it was.
            raise RuntimeError(f"failed to load subject file {path}: {exc}") from exc
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
        # arrays.x is a read-only memory-map of the whole subject; copy just
        # this one window (a few KB) into RAM so the tensor is writable and
        # no longer tied to the mapping. Only the touched window is faulted
        # in from disk, never the whole subject.
        x = torch.from_numpy(np.array(arrays.x[local_idx]))
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
