"""Shared 1D neural-network blocks for PPG models.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models
(see bpe/models/mtae.py, bpe/models/acfa.py etc.) -- generic building
blocks with no dependency on input length, so no adaptation was needed
when moving from VitalDB to this project's MIMIC-III dataset.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Return input as (batch, channels, samples)."""
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(x.shape)}")


def make_activation(name: str = "relu") -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBnAct1d(nn.Sequential):
    """Conv1d with same-ish padding, BatchNorm, and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: str = "relu",
    ):
        padding = ((kernel_size - 1) * dilation) // 2
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            make_activation(activation),
        )


class RegressionHead(nn.Module):
    """Global pooling head for SBP/DBP regression."""

    def __init__(self, in_features: int, out_features: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class DerivativeChannels(nn.Module):
    """Build PPG/VPG/APG channels from a single PPG input."""

    def __init__(self, normalize: bool = True, eps: float = 1e-6):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        if x.size(1) == 3:
            return self._normalize(x) if self.normalize else x
        if x.size(1) != 1:
            raise ValueError(
                "DerivativeChannels expects 1 raw PPG channel or precomputed "
                f"3-channel PPG/VPG/APG input, got {x.size(1)} channels"
            )

        ppg = x
        vpg = F.pad(ppg[:, :, 1:] - ppg[:, :, :-1], (1, 0))
        apg = F.pad(vpg[:, :, 1:] - vpg[:, :, :-1], (1, 0))
        out = torch.cat([ppg, vpg, apg], dim=1)
        return self._normalize(out) if self.normalize else out

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std


def validate_kernel_sizes(kernel_sizes: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(k) for k in kernel_sizes)
    if not values:
        raise ValueError("kernel_sizes must contain at least one value")
    for value in values:
        if value < 1 or value % 2 == 0:
            raise ValueError("kernel sizes must be positive odd integers")
    return values
