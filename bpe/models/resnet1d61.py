"""1D ResNet-style SBP/DBP regression models.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Fully length-agnostic (global/adaptive pooling throughout) -- no dimension
changes were needed for this project's 1,000-sample MIMIC-III input.
"""

import torch
from torch import nn

from bpe.models._blocks import ConvBnAct1d, RegressionHead, ensure_3d
from bpe.models.registry import register_model


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBnAct1d(in_channels, out_channels, 7, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = self._shortcut(in_channels, out_channels, stride)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _shortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if stride == 1 and in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class BottleneckBlock1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        hidden = out_channels
        expanded = out_channels * self.expansion
        self.conv1 = ConvBnAct1d(in_channels, hidden, 1)
        self.conv2 = ConvBnAct1d(hidden, hidden, 7, stride=stride)
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden, expanded, 1, bias=False),
            nn.BatchNorm1d(expanded),
        )
        self.shortcut = self._shortcut(in_channels, expanded, stride)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _shortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module:
        if stride == 1 and in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.act(x + residual)


@register_model("resnet1d61")
class ResNet1D61(nn.Module):
    """Residual 1D CNN for direct SBP/DBP regression."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 32,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        block: type[nn.Module] = BasicBlock1D,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            ConvBnAct1d(in_channels, base_channels, 15, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(block, base_channels, layers[0], stride=1)
        self.stage2 = self._make_stage(block, base_channels * 2, layers[1], stride=2)
        self.stage3 = self._make_stage(block, base_channels * 4, layers[2], stride=2)
        self.stage4 = self._make_stage(block, base_channels * 8, layers[3], stride=2)
        self.head = RegressionHead(self.in_channels, out_features, dropout)

    def _make_stage(
        self,
        block: type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)
