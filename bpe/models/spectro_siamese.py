"""Siamese calibration-based BP estimator (docs/method-spectrogram-cnn.md §3,
docs/development-plan.md §5.2): two weight-sharing passes of the same
backbone process the current PPG window and the patient's calibration
window; their feature vectors are subtracted (signed, not an
absolute-value/Euclidean distance, so the direction of BP change is
preserved) and regressed to delta_BP = current_BP - calibration_BP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bpe.models._backbone import (
    DEFAULT_DROPOUT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_INPUT_SAMPLES,
    PPGFeatureBackbone,
)
from bpe.models.registry import register_model
from bpe.preprocess.pipeline import DEFAULT_TARGET_FS


@register_model("spectro_siamese", calibration_based=True)
class SpectroSiamese(nn.Module):
    def __init__(
        self,
        fs: float = DEFAULT_TARGET_FS,
        input_samples: int = DEFAULT_INPUT_SAMPLES,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        # A single backbone instance is called on both inputs below, so the
        # two "twin" branches share weights by construction.
        self.backbone = PPGFeatureBackbone(
            fs, input_samples, embedding_dim, dropout)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(embedding_dim, 2)  # -> [delta_SBP, delta_DBP]

    def forward(self, waveform: torch.Tensor, calib_waveform: torch.Tensor) -> torch.Tensor:
        """`waveform`, `calib_waveform`: `(batch, samples)` -> `(batch, 2)`
        predicted `[delta_SBP, delta_DBP]` relative to the calibration
        window."""
        current_features = self.backbone(waveform)
        calib_features = self.backbone(calib_waveform)
        diff = current_features - calib_features
        return self.head(self.relu(diff))

    def predict_bp(
        self,
        waveform: torch.Tensor,
        calib_waveform: torch.Tensor,
        calib_bp: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: absolute `[SBP, DBP]` = calibration BP + predicted delta."""
        return calib_bp + self.forward(waveform, calib_waveform)
