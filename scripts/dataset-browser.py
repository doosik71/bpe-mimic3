"""GUI dataset inspection tool.

Consolidates what would otherwise be three separate browsers (dataset,
spectrogram, PSD) into one: a left-hand list to pick a split / subject /
window, and a right-hand panel stacking the waveform, spectrogram, and power
spectral density of whatever window is currently selected.
"""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from bpe.features import compute_psd, compute_spectrogram, power_to_db

DEFAULT_DATASET_DIR = Path("data/dataset")
SPLIT_NAMES = ("train", "val", "test")
# Virtual "split": subjects converted by construct-dataset but not yet moved
# into train/val/test by finalize_split -- i.e. flat npz directly under
# dataset_dir. Lets this browser inspect a dataset that's still being built.
UNSPLIT_NAME = "unsplit"


def _peek_window_count(path: Path) -> int:
    with np.load(path, mmap_mode="r") as data:
        return int(data["x"].shape[0])


class DatasetBrowserApp(tk.Tk):
    def __init__(self, dataset_dir: Path):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.title("Dataset Browser")
        self.geometry("1280x920")

        self.current_subject_id: Optional[str] = None
        self.current_arrays: Optional[dict] = None
        self._last_plot: Optional[tuple[np.ndarray, float, str]] = None
        self._meta_queue: "queue.Queue[tuple[str, int, int]]" = queue.Queue()

        self._build_widgets()
        self._refresh_splits()
        self.after(100, self._poll_meta_queue)

    # -- layout ------------------------------------------------------------

    def _build_widgets(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        split_row = ttk.Frame(parent)
        split_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(split_row, text="Split:").pack(side=tk.LEFT)
        self.split_var = tk.StringVar()
        self.split_combo = ttk.Combobox(
            split_row, textvariable=self.split_var, state="readonly", values=SPLIT_NAMES
        )
        self.split_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.split_combo.bind("<<ComboboxSelected>>", lambda e: self._on_split_change())
        # Re-scans the available splits and the current split's subject list --
        # useful while construct-dataset is still running in another process.
        ttk.Button(split_row, text="Refresh", command=self._refresh_splits).pack(side=tk.LEFT, padx=(4, 0))

        vpaned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        vpaned.pack(fill=tk.BOTH, expand=True)

        subject_frame = ttk.Frame(vpaned)
        vpaned.add(subject_frame, weight=3)
        ttk.Label(subject_frame, text="Subjects").pack(anchor=tk.W)
        subject_tree_row = ttk.Frame(subject_frame)
        subject_tree_row.pack(fill=tk.BOTH, expand=True)
        self.subject_tree = ttk.Treeview(
            subject_tree_row, columns=("windows", "size"), show="tree headings", selectmode="browse"
        )
        self.subject_tree.heading("#0", text="subject_id", command=lambda: self._sort_subjects("#0"))
        self.subject_tree.heading("windows", text="windows", command=lambda: self._sort_subjects("windows"))
        self.subject_tree.heading("size", text="size (KB)", command=lambda: self._sort_subjects("size"))
        self.subject_tree.column("#0", width=110, anchor=tk.W)
        self.subject_tree.column("windows", width=70, anchor=tk.E)
        self.subject_tree.column("size", width=80, anchor=tk.E)
        subject_scroll = ttk.Scrollbar(subject_tree_row, orient=tk.VERTICAL, command=self.subject_tree.yview)
        self.subject_tree.configure(yscrollcommand=subject_scroll.set)
        self.subject_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        subject_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.subject_tree.bind("<<TreeviewSelect>>", lambda e: self._on_subject_select())

        segment_frame = ttk.Frame(vpaned)
        vpaned.add(segment_frame, weight=4)
        ttk.Label(segment_frame, text="Windows").pack(anchor=tk.W)
        segment_tree_row = ttk.Frame(segment_frame)
        segment_tree_row.pack(fill=tk.BOTH, expand=True)
        self.segment_tree = ttk.Treeview(
            segment_tree_row, columns=("index", "sbp", "dbp"), show="headings", selectmode="browse"
        )
        for col, label, width in (("index", "#", 70), ("sbp", "SBP", 70), ("dbp", "DBP", 70)):
            self.segment_tree.heading(col, text=label)
            self.segment_tree.column(col, width=width, anchor=tk.E)
        self.segment_tree.tag_configure("calib", background="#fff2cc")
        segment_scroll = ttk.Scrollbar(segment_tree_row, orient=tk.VERTICAL, command=self.segment_tree.yview)
        self.segment_tree.configure(yscrollcommand=segment_scroll.set)
        self.segment_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        segment_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.segment_tree.bind("<<TreeviewSelect>>", lambda e: self._on_segment_select())

        self.bind("<Up>", lambda e: self._step_subject(-1))
        self.bind("<Down>", lambda e: self._step_subject(1))
        self.bind("<Left>", lambda e: self._step_segment(-1))
        self.bind("<Right>", lambda e: self._step_segment(1))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top_row = ttk.Frame(parent)
        top_row.pack(fill=tk.X)
        self.info_var = tk.StringVar(value="Select a subject and window to inspect.")
        ttk.Label(top_row, textvariable=self.info_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Off by default: the colorbar otherwise shrinks the spectrogram axes,
        # so its time axis no longer spans the same width as the waveform
        # plot above it -- misleading when comparing them at the same time.
        self.show_colorbar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_row,
            text="Show spectrogram scale bar",
            variable=self.show_colorbar_var,
            command=self._on_toggle_colorbar,
        ).pack(side=tk.RIGHT)

        self.fig = Figure(figsize=(8, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X)

    # -- data loading --------------------------------------------------------

    def _split_dir(self, split: str) -> Path:
        """`unsplit` is a virtual split: subjects sit flat directly under
        `dataset_dir`, not yet moved into a split subdirectory."""
        return self.dataset_dir if split == UNSPLIT_NAME else self.dataset_dir / split

    def _refresh_splits(self) -> None:
        previous = self.split_var.get()
        available = [s for s in SPLIT_NAMES if (self.dataset_dir / s).is_dir()]
        if self.dataset_dir.is_dir() and any(self.dataset_dir.glob("*.npz")):
            available.append(UNSPLIT_NAME)
        if not available:
            messagebox.showwarning(
                "No dataset found",
                f"No npz files found under {self.dataset_dir} (flat or in train/val/test).\n"
                "Run construct-dataset first.",
            )
            self.split_combo["values"] = ()
            return
        self.split_combo["values"] = available
        self.split_var.set(previous if previous in available else available[0])
        self._on_split_change()

    def _on_split_change(self) -> None:
        split = self.split_var.get()
        self.subject_tree.delete(*self.subject_tree.get_children())
        self.segment_tree.delete(*self.segment_tree.get_children())
        self.current_subject_id = None
        self.current_arrays = None
        if not split:
            return
        paths = sorted(self._split_dir(split).glob("*.npz"))
        for path in paths:
            self.subject_tree.insert("", tk.END, iid=path.stem, text=path.stem, values=("...", "..."))
        threading.Thread(target=self._scan_metadata, args=(paths,), daemon=True).start()
        if paths:
            first = paths[0].stem
            self.subject_tree.selection_set(first)
            self.subject_tree.see(first)

    def _scan_metadata(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                n_windows = _peek_window_count(path)
            except Exception:
                n_windows = -1
            size_kb = path.stat().st_size / 1024
            self._meta_queue.put((path.stem, n_windows, size_kb))

    def _poll_meta_queue(self) -> None:
        try:
            while True:
                subject_id, n_windows, size_kb = self._meta_queue.get_nowait()
                if self.subject_tree.exists(subject_id):
                    self.subject_tree.set(subject_id, "windows", str(n_windows))
                    self.subject_tree.set(subject_id, "size", f"{size_kb:.0f}")
        except queue.Empty:
            pass
        self.after(100, self._poll_meta_queue)

    def _sort_subjects(self, col: str) -> None:
        items = [(self.subject_tree.set(iid, col) if col != "#0" else self.subject_tree.item(iid, "text"), iid)
                 for iid in self.subject_tree.get_children("")]

        def _key(pair):
            value, _ = pair
            try:
                return (0, float(value))
            except ValueError:
                return (1, value)

        items.sort(key=_key)
        for index, (_, iid) in enumerate(items):
            self.subject_tree.move(iid, "", index)

    # -- selection handlers --------------------------------------------------

    def _on_subject_select(self) -> None:
        selection = self.subject_tree.selection()
        if not selection:
            return
        subject_id = selection[0]
        split = self.split_var.get()
        path = self._split_dir(split) / f"{subject_id}.npz"
        try:
            with np.load(path) as data:
                arrays = {key: data[key] for key in ("x", "y", "calib_x", "calib_y", "fs")}
        except Exception as exc:
            messagebox.showerror("Load error", f"Failed to load {path}:\n{exc}")
            return
        self.current_subject_id = subject_id
        self.current_arrays = arrays
        self._populate_segments()

    def _populate_segments(self) -> None:
        self.segment_tree.delete(*self.segment_tree.get_children())
        arrays = self.current_arrays
        calib_sbp, calib_dbp = arrays["calib_y"]
        self.segment_tree.insert(
            "", tk.END, iid="calib", values=("calib", f"{calib_sbp:.1f}", f"{calib_dbp:.1f}"), tags=("calib",)
        )
        for i, (sbp, dbp) in enumerate(arrays["y"]):
            self.segment_tree.insert("", tk.END, iid=str(i), values=(str(i), f"{sbp:.1f}", f"{dbp:.1f}"))
        first_iid = "0" if len(arrays["y"]) > 0 else "calib"
        self.segment_tree.selection_set(first_iid)
        self.segment_tree.see(first_iid)

    def _on_segment_select(self) -> None:
        selection = self.segment_tree.selection()
        if not selection or self.current_arrays is None:
            return
        iid = selection[0]
        arrays = self.current_arrays
        fs = float(arrays["fs"])
        if iid == "calib":
            x = arrays["calib_x"]
            sbp, dbp = arrays["calib_y"]
            label = "calibration window"
        else:
            idx = int(iid)
            x = arrays["x"][idx]
            sbp, dbp = arrays["y"][idx]
            label = f"window {idx}"
        self.info_var.set(
            f"{self.current_subject_id}  |  {label}  |  SBP {sbp:.1f} mmHg  DBP {dbp:.1f} mmHg  |  "
            f"{len(x)} samples @ {fs:.0f} Hz"
        )
        self._plot_segment(x, fs, f"{self.current_subject_id} -- {label}")

    def _on_toggle_colorbar(self) -> None:
        if self._last_plot is not None:
            self._plot_segment(*self._last_plot)

    # -- keyboard navigation --------------------------------------------------

    def _step_subject(self, delta: int) -> None:
        children = self.subject_tree.get_children("")
        if not children:
            return
        selection = self.subject_tree.selection()
        index = children.index(selection[0]) if selection else 0
        index = max(0, min(len(children) - 1, index + delta))
        next_iid = children[index]
        self.subject_tree.selection_set(next_iid)
        self.subject_tree.see(next_iid)

    def _step_segment(self, delta: int) -> None:
        children = self.segment_tree.get_children("")
        if not children:
            return
        selection = self.segment_tree.selection()
        index = children.index(selection[0]) if selection else 0
        index = max(0, min(len(children) - 1, index + delta))
        next_iid = children[index]
        self.segment_tree.selection_set(next_iid)
        self.segment_tree.see(next_iid)

    # -- plotting --------------------------------------------------------

    def _plot_segment(self, x: np.ndarray, fs: float, title_suffix: str) -> None:
        self._last_plot = (x, fs, title_suffix)
        self.fig.clf()
        ax_wave, ax_spec, ax_psd = self.fig.subplots(3, 1)

        t = np.arange(len(x)) / fs
        ax_wave.plot(t, x, color="tab:blue", linewidth=0.8)
        ax_wave.set_title(f"PPG Waveform -- {title_suffix}")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")

        freqs, times, power = compute_spectrogram(x, fs)
        mesh = ax_spec.pcolormesh(times, freqs, power_to_db(power), shading="gouraud", cmap="viridis")
        ax_spec.set_title("Spectrogram (1 s Hamming, 95% overlap)")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        # The colorbar steals width from ax_spec, breaking its time-axis
        # alignment with ax_wave right above it -- keep it opt-in.
        if self.show_colorbar_var.get():
            self.fig.colorbar(mesh, ax=ax_spec, label="dB")

        freqs_psd, psd = compute_psd(x, fs)
        ax_psd.plot(freqs_psd, power_to_db(psd), color="tab:red", linewidth=0.8)
        ax_psd.set_title("Power Spectral Density (Welch)")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("Power (dB/Hz)")

        self.fig.tight_layout()
        self.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset directory to browse (default: %(default)s)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = DatasetBrowserApp(args.dataset_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
