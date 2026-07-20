"""Calibration-free BP estimator (docs/method.md §2, docs/development-plan.md
§5.1): predicts SBP/DBP directly from a single PPG window's spectrogram, no
patient-specific reference reading required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bpe.models.backbone import (
    DEFAULT_DROPOUT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INPUT_SAMPLES,
    PPGFeatureBackbone,
)
from bpe.preprocess.pipeline import DEFAULT_TARGET_FS


class CalibrationFreeCNN(nn.Module):
    def __init__(
        self,
        fs: float = DEFAULT_TARGET_FS,
        input_samples: int = DEFAULT_INPUT_SAMPLES,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.backbone = PPGFeatureBackbone(fs, input_samples, embedding_dim, dropout)
        self.head = nn.Linear(embedding_dim, 2)  # linear regression -> [SBP, DBP]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """`waveform`: `(batch, samples)` -> `(batch, 2)` `[SBP, DBP]` mmHg."""
        return self.head(self.backbone(waveform))
