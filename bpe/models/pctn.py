"""Parallel CNN-Transformer Network (PCTN) for PPG-based BP estimation.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Already targets 1,000-sample (8 s @ 125 Hz) input -- no dimension changes
were needed for this project's MIMIC-III dataset.

Reference: Tian et al., "A paralleled CNN and Transformer network for PPG-based
cuff-less blood pressure estimation," Biomed. Signal Process. Control 99 (2025) 106741.

Architecture:
  Stem -> (CNN branch || Transformer branch) -> CBAM-style Fusion -> Regressor

  Stem          Large-kernel Conv + BN + MaxPool for shallow feature extraction.
  CNN branch    1 ResNet-50 stage: 3 bottleneck blocks (1x1 -> 3x1 -> 1x1 + residual).
  Transformer   Linear projection + learnable positional encoding + 6 encoder layers
                (num_heads=4, FFN expansion 4x).
  Fusion        Spatial attention on each branch independently (to preserve the
                separation of local/global features), then concatenation, then
                modified SE channel attention (1x1 Conv instead of FC layers).
  Regressor     Global average pool -> FC -> FC -> 2 (SBP, DBP).
"""

import torch
from torch import nn

from bpe.models.blocks import ConvBnAct1d, ensure_3d
from bpe.models.registry import register_model


class _Bottleneck1D(nn.Module):
    """1D ResNet-50-style bottleneck (1x1 -> 3x1 -> 1x1 + residual connection)."""

    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1):
        super().__init__()
        out = mid_channels * self.expansion
        self.conv1 = ConvBnAct1d(in_channels, mid_channels, 1)
        self.conv2 = ConvBnAct1d(mid_channels, mid_channels, 3, stride=stride)
        self.conv3 = nn.Sequential(
            nn.Conv1d(mid_channels, out, 1, bias=False),
            nn.BatchNorm1d(out),
        )
        if in_channels != out or stride != 1:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(self.conv2(self.conv1(x))) + self.shortcut(x))


class _SpatialAttention1d(nn.Module):
    """1D CBAM spatial attention: channel-wise avg/max pool -> conv -> sigmoid."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class _ChannelAttention1d(nn.Module):
    """Modified SE channel attention using 1x1 Conv instead of FC layers.

    Replaces the FC-based squeeze-excitation with 1D convolutions (kernel=1)
    for better local abstraction and reduced overfitting, as described in the paper.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.excite = nn.Sequential(
            nn.Conv1d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excite(self.pool(x))


@register_model("pctn")
class PCTN(nn.Module):
    """Parallel CNN-Transformer Network for SBP/DBP regression from PPG.

    Input:  (B, 1000) or (B, 1, 1000) -- 8 s PPG waveform @ 125 Hz.
    Output: (B, 2)    -- [SBP, DBP] in mmHg.

    Key hyperparameters follow Section 4.3 of the paper:
      - CNN backbone: ResNet-50 (first stage, 3 bottleneck blocks)
      - Transformer: num_heads=4, depth (num_layers)=6
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        stem_channels: int = 64,
        cnn_mid: int = 64,       # bottleneck mid-dim; output dim = cnn_mid x 4 = 256
        cnn_blocks: int = 3,     # ResNet-50 conv2_x stage has 3 bottleneck blocks
        d_model: int = 256,      # shared feature dim for CNN output and Transformer
        num_heads: int = 4,
        num_tr_layers: int = 6,  # Transformer encoder depth (Section 4.3)
        ffn_ratio: int = 4,      # FFN hidden dim = d_model x ffn_ratio
        dropout: float = 0.1,
    ):
        super().__init__()

        # -- Stem ---------------------------------------------------------------
        # Large-kernel conv preserves more original waveform information.
        # Band-pass filtering (0.5-10 Hz) is handled upstream in the dataset.
        # Input (B,1,1000) -> (B,64,500) -> (B,64,250)
        self.stem = nn.Sequential(
            ConvBnAct1d(in_channels, stem_channels, 15, stride=2),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # -- CNN branch (1 ResNet-50 stage) ---------------------------------------
        # Pyramid bottleneck: stem_channels -> d_model via expansion-4 bottlenecks.
        cnn_out = cnn_mid * _Bottleneck1D.expansion  # 64 x 4 = 256
        cnn_layers: list[nn.Module] = []
        in_ch = stem_channels
        for _ in range(cnn_blocks):
            cnn_layers.append(_Bottleneck1D(in_ch, cnn_mid))
            in_ch = cnn_out
        if cnn_out != d_model:
            cnn_layers.append(ConvBnAct1d(cnn_out, d_model, 1))
        self.cnn_branch = nn.Sequential(*cnn_layers)

        # -- Transformer branch ---------------------------------------------------
        # Embedding: project stem features to d_model, add learnable positional
        # encoding, then pass through num_tr_layers TransformerEncoder layers.
        # Sequence length after stem is fixed at 250 for 1000-sample input.
        _seq_len = 250
        self.tr_proj = nn.Conv1d(stem_channels, d_model, 1, bias=False)
        self.tr_pos = nn.Parameter(torch.zeros(1, _seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * ffn_ratio,
            dropout=dropout,
            batch_first=True,
        )
        self.tr_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_tr_layers)

        # -- Fusion block ----------------------------------------------------------
        # Spatial attention applied to each branch independently to preserve the
        # separation of local (CNN) and global (Transformer) features.  Then
        # concatenate and apply channel attention on the joint representation.
        self.cnn_spatial = _SpatialAttention1d()
        self.tr_spatial = _SpatialAttention1d()
        self.channel_att = _ChannelAttention1d(d_model * 2)

        # -- Regressor ---------------------------------------------------------------
        # Global average pool aggregates all temporal positions, then two FC
        # layers map the fused representation to SBP and DBP.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)                                   # (B, 1, 1000)

        s = self.stem(x)                                   # (B, 64, 250)

        # CNN branch: local feature extraction via stacked bottlenecks
        cnn = self.cnn_branch(s)                           # (B, d_model, 250)

        # Transformer branch: global feature extraction via self-attention
        tr = self.tr_proj(s).permute(0, 2, 1)             # (B, 250, d_model)
        tr = tr + self.tr_pos[:, : tr.size(1), :]         # add positional encoding
        tr = self.tr_encoder(tr).permute(0, 2, 1)         # (B, d_model, 250)

        # Fusion: independent spatial attention, concat, channel attention
        cnn = self.cnn_spatial(cnn)
        tr = self.tr_spatial(tr)
        fused = self.channel_att(torch.cat([cnn, tr], dim=1))  # (B, 2*d_model, 250)

        # Regressor: global pool -> FC -> FC
        out = self.act(self.fc1(self.pool(fused).flatten(1)))
        out = self.drop(out)
        return self.fc2(out)                               # (B, 2)
