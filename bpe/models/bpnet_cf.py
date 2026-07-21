"""BPNet-CF: calibration-free dual-scale PPG model for SBP/DBP regression.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Fully length-agnostic (global/adaptive pooling throughout) -- no dimension
changes were needed for this project's 1,000-sample MIMIC-III input.

Implements the architecture described in ``docs/model-design-bpnet_cf.md``:

    input PPG
      -> short-scale depthwise-separable encoder
      -> long-scale depthwise-separable encoder
      -> shared 1x1 projection
      -> asymmetric SBP / DBP channel split
      -> per-head temporal self-attention
      -> per-head regression
      -> [SBP, DBP]

Input shape:  (batch, 1000) or (batch, 1, 1000)
Output shape: (batch, 2) = [SBP, DBP]
"""

import torch
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable Conv1d block without activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class _EncoderStage(nn.Module):
    """Depthwise-separable stage with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        pool: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SEBlock(nn.Module):
    """Squeeze-and-excitation channel reweighting."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = channels // reduction
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x).flatten(1)
        scale = self.fc(scale).unsqueeze(-1)
        return x * scale


class _BranchSummary(nn.Module):
    """Summarize a branch into a compact 1-channel temporal sequence."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.post = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1, bias=False),
            nn.AvgPool1d(kernel_size=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.spatial_proj = nn.Conv1d(32, 1, kernel_size=1)
        self.channel_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.post(x)                                    # (B, 32, 125)
        spatial = self.spatial_proj(x)                      # (B, 1, 125)
        channel = self.channel_pool(x).transpose(1, 2)      # (B, 1, 32)
        return torch.cat([spatial, channel], dim=-1)        # (B, 1, 157)


class _DualScaleBranch(nn.Module):
    """One branch of the dual-scale encoder."""

    def __init__(self, kernels: tuple[int, int, int], dropout: float = 0.2) -> None:
        super().__init__()
        self.stage1 = _EncoderStage(1, 32, kernels[0], pool=True)
        self.stage2 = _EncoderStage(32, 64, kernels[1], pool=True)
        self.stage3 = _EncoderStage(64, 128, kernels[2], pool=False)
        self.se = _SEBlock(128, reduction=4)
        self.summary = _BranchSummary(dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.se(x)
        return self.summary(x)


class _AttentionBlock(nn.Module):
    """Transformer-style self-attention block over temporal tokens."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class _RegressionHead(nn.Module):
    """Regress a scalar from temporal features."""

    def __init__(self, embed_dim: int, hidden_dims: tuple[int, ...], dropout: float = 0.2) -> None:
        super().__init__()
        self.time_proj = nn.Linear(embed_dim, 1)
        dims = [157 + embed_dim, *hidden_dims, 1]
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        for i in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(inplace=True),
            ])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_summary = self.time_proj(x).squeeze(-1)   # (B, 157)
        channel_summary = x.mean(dim=1)                # (B, C)
        fused = torch.cat([time_summary, channel_summary], dim=1)
        return self.mlp(fused)


@register_model("bpnet_cf")
class BPNetCF(nn.Module):
    """Calibration-free dual-scale PPG regressor for [SBP, DBP]."""

    def __init__(self, dropout: float = 0.2, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.short_encoder = _DualScaleBranch((5, 5, 3), dropout=dropout)
        self.long_encoder = _DualScaleBranch((15, 11, 7), dropout=dropout)

        self.shared_proj = nn.Conv1d(2, 32, kernel_size=1)
        self.asym_proj = nn.Conv1d(32, 48, kernel_size=1)

        self.sbp_attention = _AttentionBlock(32, num_heads=4, ff_dim=64, dropout=attention_dropout)
        self.dbp_attention = _AttentionBlock(16, num_heads=2, ff_dim=32, dropout=attention_dropout)

        self.sbp_head = _RegressionHead(32, hidden_dims=(128, 64, 32), dropout=dropout)
        self.dbp_head = _RegressionHead(16, hidden_dims=(64, 32), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)                                # (B, 1, 1000)

        short = self.short_encoder(x)                   # (B, 1, 157)
        long = self.long_encoder(x)                     # (B, 1, 157)
        fused = torch.cat([short, long], dim=1)         # (B, 2, 157)

        shared = self.shared_proj(fused)                # (B, 32, 157)
        split = self.asym_proj(shared)                  # (B, 48, 157)

        sbp_feat = split[:, :32, :].transpose(1, 2)     # (B, 157, 32)
        dbp_feat = split[:, 32:, :].transpose(1, 2)     # (B, 157, 16)

        sbp_feat = self.sbp_attention(sbp_feat)         # (B, 157, 32)
        dbp_feat = self.dbp_attention(dbp_feat)         # (B, 157, 16)

        sbp = self.sbp_head(sbp_feat)                   # (B, 1)
        dbp = self.dbp_head(dbp_feat)                   # (B, 1)
        return torch.cat([sbp, dbp], dim=1)             # (B, 2)
