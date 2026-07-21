"""Simple CNN regression baseline for SBP/DBP estimation.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Already targets 1,000-sample (8 s @ 125 Hz) input -- no dimension changes
were needed for this project's MIMIC-III dataset.

Six Conv1d stages with BatchNorm + ReLU + AvgPool halve the time dimension at
each step (1000 -> 500 -> 250 -> 125 -> 62 -> 31 -> 15), followed by an
AdaptiveAvgPool1d(1) to collapse the remaining time axis.  A two-layer MLP
regressor then maps the 64-d feature vector to [SBP, DBP].
"""

import torch.nn as nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


@register_model("conv_reg")
class ConvReg(nn.Module):
    """Lightweight 1-D CNN regressor for blood-pressure estimation.

    Input shape:  (batch, 1, 1000)  -- 8 s PPG @ 125 Hz
    Output shape: (batch, 2)        -- [SBP, DBP] in mmHg
    """

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 1000 -> 500

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 500 -> 250

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 250 -> 125

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 125 -> 62

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 62 -> 31

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 31 -> 15

            nn.AdaptiveAvgPool1d(1),        # (B, 64, 15) -> (B, 64, 1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),                   # (B, 64, 1) -> (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),               # output = [SBP, DBP]
        )

    def forward(self, x):
        x = ensure_3d(x)                    # (B, L) -> (B, 1, L)
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x
