"""Torch-native log-power spectrogram front end.

Uses the same parameters as bpe.features.spectrogram (1 s Hamming
sub-window, 95% overlap, docs/development-plan.md §1 decision) so the CNN
sees the same time-frequency representation the dataset browser displays,
but computed with `torch.stft` so it runs on-device as part of the forward
pass instead of a separate CPU preprocessing step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bpe.features.spectrogram import DEFAULT_OVERLAP_FRACTION, DEFAULT_SUB_WINDOW_SEC


class LogSpectrogram(nn.Module):
    def __init__(
        self,
        fs: float,
        sub_window_sec: float = DEFAULT_SUB_WINDOW_SEC,
        overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
        eps: float = 1e-12,
    ):
        super().__init__()
        n_fft = max(1, int(round(sub_window_sec * fs)))
        hop_length = max(1, n_fft - int(round(n_fft * overlap_fraction)))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer("window", torch.hamming_window(n_fft), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(batch, samples)` waveform -> `(batch, 1, freq_bins, time_frames)`
        log-power spectrogram."""
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs() ** 2
        log_power = torch.log(power + self.eps)
        return log_power.unsqueeze(1)
