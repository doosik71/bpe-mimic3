"""Spectro-temporal ResNet for PPG/VPG/APG direct BP regression.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Fully length-agnostic (global/adaptive pooling throughout) -- no dimension
changes were needed for this project's 1,000-sample MIMIC-III input.
"""

import torch
from torch import nn

from bpe.models._blocks import ConvBnAct1d, DerivativeChannels, ensure_3d
from bpe.models.registry import register_model
from bpe.models.resnet1d61 import BasicBlock1D


class SignalBranch(nn.Module):
    """Small residual branch for one temporal signal channel."""

    def __init__(self, base_channels: int = 24, embedding_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnAct1d(1, base_channels, 15, stride=2),
            BasicBlock1D(base_channels, base_channels, stride=1),
            BasicBlock1D(base_channels, base_channels * 2, stride=2),
            BasicBlock1D(base_channels * 2, base_channels * 4, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("st_resnet")
class SpectroTemporalResNet(nn.Module):
    """Three-branch residual model over PPG, VPG, and APG channels."""

    def __init__(
        self,
        out_features: int = 2,
        base_channels: int = 24,
        embedding_dim: int = 96,
        dropout: float = 0.2,
        derive_channels: bool = True,
    ):
        super().__init__()
        self.derive = DerivativeChannels() if derive_channels else nn.Identity()
        self.ppg_branch = SignalBranch(base_channels, embedding_dim)
        self.vpg_branch = SignalBranch(base_channels, embedding_dim)
        self.apg_branch = SignalBranch(base_channels, embedding_dim)
        fused = embedding_dim * 3
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Dropout(dropout),
            nn.Linear(fused, fused // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused // 2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.derive(ensure_3d(x))
        if x.size(1) != 3:
            raise ValueError(
                "SpectroTemporalResNet expects raw 1-channel PPG with "
                "derive_channels=True or 3-channel PPG/VPG/APG input"
            )
        features = [
            self.ppg_branch(x[:, 0:1]),
            self.vpg_branch(x[:, 1:2]),
            self.apg_branch(x[:, 2:3]),
        ]
        return self.head(torch.cat(features, dim=1))
