"""ResNet1D13: 13-layer variant of ResNet1D61 (~10% of its layer count).

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Fully length-agnostic (global/adaptive pooling throughout) -- no dimension
changes were needed for this project's 1,000-sample MIMIC-III input.

ResNet1D61: 4 stages x 2 BasicBlock1D = 8 blocks, 61 layers.
ResNet1D13: 1 stage  x 1 BasicBlock1D = 1 block, 13 layers.
Channel progression is limited to a single stage (32 only).
"""

import torch
from torch import nn

from bpe.models._blocks import ConvBnAct1d, RegressionHead, ensure_3d
from bpe.models.registry import register_model
from bpe.models.resnet1d61 import BasicBlock1D


@register_model("resnet1d13")
class ResNet1D13(nn.Module):
    """Minimal 1D ResNet -- 1 residual block in a single stage."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct1d(in_channels, base_channels, 15, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(BasicBlock1D(base_channels, base_channels, stride=1))
        self.head = RegressionHead(base_channels, out_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = self.stem(x)
        x = self.stage1(x)
        return self.head(x)
