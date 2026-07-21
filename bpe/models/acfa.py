"""ACFA: Adaptive Cross-domain Fusion Architecture for blood pressure estimation.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Already targets 1,000-sample (8 s @ 125 Hz) input by default, with dynamic
sequence-length handling (interpolated positional embeddings) for other
lengths -- no dimension changes were needed for this project's MIMIC-III
dataset.

Reference:
    Li et al., "ACFA: A Hybrid Deep Learning Framework for Cuffless Continuous
    Blood Pressure Estimation Using Time-Frequency Adaptive PPG Features",
    IEEE Access, vol. 14, 2026. DOI: 10.1109/ACCESS.2026.3657471

Architecture (Section III-B):
    DyCASNet (CASB + DCB)  ->  xLSTM (sLSTM + mLSTM)  ->  Transformer  ->  FKAN  ->  [SBP, DBP]

    DyCASNet  performs adaptive dual-domain feature extraction:
      CASB  applies FFT, learnable adaptive frequency masking, complex-valued
            channel weights, and a Squeeze-and-Excitation channel attention
            before inverting back to the time domain with IFFT.
      DCB   uses two parallel 1-D convolutions (small / large kernel) fused
            by input-adaptive dynamic weights, followed by channel attention.

    xLSTM   alternates sLSTM blocks (causal-conv preprocessing + BiLSTM +
            residual) with mLSTM blocks (causal-conv + causal multi-head
            attention + output gate + residual) to capture both local
            short-term dynamics and longer-range sequence patterns.

    Transformer  models global contextual relationships across all time steps
                 via standard multi-head self-attention.

    FKAN    applies FastKAN layers (radial-basis-function basis functions +
            linear auxiliary path) as a nonlinear regression head to produce
            the final SBP / DBP estimates.

Adaptations from the paper:
    * Input: 1 000-sample PPG segments (8 s @ 125 Hz) vs. the paper's 789-sample
      segments (~6 s at the MIMIC-III native rate).
    * A stride-4 average-pool is applied after DyCASNet to reduce the sequence
      length to 250 time steps before the xLSTM / Transformer passes, keeping
      attention-matrix memory within practical limits for large batch sizes.
    * Output: [SBP, DBP] in mmHg (no MAP head; MAP is not a label in this dataset).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model

# -----------------------------------------------------------------------------
# Channel-Aware Spectral Block (CASB)   [paper Section III-B-2a]
# -----------------------------------------------------------------------------


class CASB(nn.Module):
    """Channel-Aware Spectral Block.

    Steps (Eq. 2-6 in the paper):
        1. rfft  -> complex spectrum F
        2. Power mask  P = |F|^2; adaptive threshold theta suppresses noise bands
        3. Learnable complex weights refine global and filtered spectra
        4. Squeeze-and-Excitation channel attention on spectral energy
        5. irfft returns enhanced temporal features
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Learnable per-channel frequency threshold theta (Eq. 4)
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1))
        # Learnable complex weights: stored as separate real / imag tensors
        # so they are always nn.Parameter objects (plain complex params are
        # not supported in older PyTorch versions).
        self.global_r = nn.Parameter(torch.ones(1, channels, 1))
        self.global_i = nn.Parameter(torch.zeros(1, channels, 1))
        self.local_r = nn.Parameter(torch.ones(1, channels, 1))
        self.local_i = nn.Parameter(torch.zeros(1, channels, 1))
        # Channel-attention (SE) over frequency-dimension energy (Eq. 5)
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, L)
        B, C, L = x.shape
        # Step 1 - rfft along temporal dim (Eq. 2)
        F_x = torch.fft.rfft(x, dim=-1)           # (B, C, L//2+1) complex
        # Step 2 - adaptive frequency masking (Eq. 3-4)
        P = F_x.abs() ** 2                         # power spectrum (real)
        theta = self.threshold.abs()               # (1, C, 1)  broadcast OK
        mask = (P > theta).to(x.dtype)             # (B, C, F_len) real float
        F_filt = F_x * mask                        # zero out noise bands
        # Step 3 - complex-weight refinement
        gw = torch.complex(self.global_r, self.global_i)   # (1, C, 1)
        lw = torch.complex(self.local_r, self.local_i)     # (1, C, 1)
        F_ref = F_x * gw + F_filt * lw            # (B, C, F_len) complex
        # Step 4 - channel attention on average spectral energy (Eq. 5)
        energy = F_ref.abs().mean(dim=-1)          # (B, C)
        scale = self.se(energy).unsqueeze(-1)      # (B, C, 1)
        F_att = F_ref * scale                      # (B, C, F_len) complex
        # Step 5 - irfft back to time domain (Eq. 6)
        return torch.fft.irfft(F_att, n=L, dim=-1)  # (B, C, L)


# -----------------------------------------------------------------------------
# Dynamic Convolution Block (DCB)   [paper Section III-B-2b]
# -----------------------------------------------------------------------------


class DCB(nn.Module):
    """Dynamic Convolution Block.

    Two parallel convolutions capture local (small kernel) and long-range
    (large kernel) patterns. Their outputs are fused with K sets of learnable
    channel-wise weights whose mixing coefficients alpha_k are predicted from the
    input (Eq. 7-10). A final SE block re-calibrates channel importance.
    """

    def __init__(
        self,
        channels: int,
        kernel_small: int = 3,
        kernel_large: int = 15,
        num_kernels: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        pad_s = kernel_small // 2
        pad_l = kernel_large // 2
        self.conv_s = nn.Conv1d(channels, channels, kernel_small, padding=pad_s, bias=False)
        self.conv_l = nn.Conv1d(channels, channels, kernel_large, padding=pad_l, bias=False)
        self.act = nn.GELU()
        # K learnable channel-wise weight tensors W_k (Eq. 8)
        self.num_kernels = num_kernels
        self.W = nn.Parameter(torch.ones(num_kernels, channels, 1))
        # Input-adaptive alpha predictor (Eq. 8)
        self.alpha_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, num_kernels),
            nn.Softmax(dim=-1),
        )
        # Channel attention (Eq. 9)
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, C, L)
        A1 = self.act(self.conv_s(x))   # local patterns
        A2 = self.act(self.conv_l(x))   # long-range patterns
        alpha = self.alpha_net(x)        # (B, K)
        # Fused output: O = Sum_k alpha_k * (A1 + A2) (dot) W_k  (Eq. 8, simplified)
        # alpha: (B, K) -> (B, K, 1, 1) for broadcast over (K, C, 1)
        W_mix = (alpha[:, :, None, None] * self.W[None]).sum(dim=1)  # (B, C, 1)
        O = (A1 + A2) * W_mix           # (B, C, L)
        # SE channel attention (Eq. 9)
        scale = self.se(O).unsqueeze(-1)
        return O * scale                 # (B, C, L)


# -----------------------------------------------------------------------------
# DyCASNet   [paper Section III-B-2]
# -----------------------------------------------------------------------------


class DyCASNet(nn.Module):
    """Dual-domain adaptive feature extractor combining CASB and DCB.

    A stem convolution first projects the raw 1-channel PPG to d_model channels.
    CASB and DCB are then applied in sequence, each with a residual shortcut,
    followed by Batch Normalisation.  Two CASB+DCB stages are stacked for
    richer multi-scale representation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 64,
        num_kernels: int = 4,
        reduction: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        # Two stages of CASB + DCB
        self.casb1 = CASB(d_model, reduction)
        self.dcb1 = DCB(d_model, num_kernels=num_kernels, reduction=reduction)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.casb2 = CASB(d_model, reduction)
        self.dcb2 = DCB(d_model, num_kernels=num_kernels, reduction=reduction)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, L)
        x = self.stem(x)                    # (B, d_model, L)
        # Stage 1
        x = self.bn1(self.dcb1(self.casb1(x) + x) + x)
        # Stage 2
        x = self.bn2(self.dcb2(self.casb2(x) + x) + x)
        return x                            # (B, d_model, L)


# -----------------------------------------------------------------------------
# xLSTM components   [paper Section III-B-3]
# -----------------------------------------------------------------------------


class sLSTMBlock(nn.Module):
    """Scalar LSTM (sLSTM) block.

    Captures local temporal dynamics with causal depthwise convolution
    preprocessing followed by a bidirectional LSTM and a residual connection.
    Uses Layer Normalisation for training stability (Eq. 11 in the paper).
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Causal depthwise conv: kernel=4, left-pad=3 -> slice to original L
        self.causal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=4, padding=3, groups=d_model, bias=False
        )
        self.norm_in = nn.LayerNorm(d_model)
        # BiLSTM doubles hidden dim; project back to d_model
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, D)
        L = x.size(1)
        # Causal conv in channel-last -> channel-first format
        xc = self.causal_conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x = self.norm_in(x + xc)           # pre-norm residual
        out, _ = self.lstm(x)
        return self.norm_out(x + self.proj(out))


class mLSTMBlock(nn.Module):
    """Matrix LSTM (mLSTM) block.

    Models higher-order nonlinear temporal interactions via causal multi-head
    self-attention (approximating the outer-product matrix-memory update of the
    original mLSTM) with an element-wise output gate inspired by the mLSTM
    gating mechanism.  F.scaled_dot_product_attention is used for efficient
    memory usage (flash attention when running on CUDA).
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Causal depthwise conv preprocessing (same as sLSTM)
        self.causal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=4, padding=3, groups=d_model, bias=False
        )
        self.norm_in = nn.LayerNorm(d_model)
        # Q / K / V projections + output projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Output gate (mLSTM-style learned gating)
        self.o_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, D)
        B, L, D = x.shape
        # Causal conv preprocessing
        xc = self.causal_conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_in = self.norm_in(x + xc)
        # Q, K, V
        qkv = self.qkv(x_in).view(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, head_dim)
        q, k, v = qkv.unbind(0)
        # Causal scaled dot-product attention (flash-attention when available)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).reshape(B, L, D)
        h = self.out_proj(h)
        # mLSTM output gate
        o = self.o_gate(x_in)
        return self.norm_out(x + o * h)


class xLSTMStack(nn.Module):
    """Stack of alternating sLSTM and mLSTM blocks (Eq. 11-12 in the paper)."""

    def __init__(self, d_model: int, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            if i % 2 == 0:
                layers.append(sLSTMBlock(d_model))
            else:
                layers.append(mLSTMBlock(d_model, num_heads))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------------------------------------------------------
# Transformer branch   [paper Section III-B-4]
# -----------------------------------------------------------------------------


class TransformerBranch(nn.Module):
    """Transformer encoder for global contextual modelling (Eq. 13-14).

    Adds a learnable positional embedding before the standard PyTorch
    TransformerEncoder so the model can distinguish time positions.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-LN (more stable training)
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, D); add positional embedding clipped / interpolated if needed
        if x.size(1) == self.pos_embed.size(1):
            x = x + self.pos_embed
        else:
            # Interpolate position embedding for variable-length inputs
            pe = F.interpolate(
                self.pos_embed.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
            x = x + pe
        return self.encoder(x)


# -----------------------------------------------------------------------------
# FKAN   [paper Section III-B-5]
# -----------------------------------------------------------------------------


class FastKANLayer(nn.Module):
    """FastKAN layer using radial basis function (RBF) bases (Eq. 15).

    Main path:   phi(x) = Sum_i w_i * exp(-(x - mu_i)^2 / (2*sigma_i^2))  followed by
                 a linear projection of the stacked RBF activations.
    Auxiliary path:  simple linear projection of the input.
    Both paths are summed and Layer-Normalised.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.num_basis = num_basis
        # RBF centres initialised on a uniform grid; learnable
        centers = torch.linspace(grid_range[0], grid_range[1], num_basis)
        self.centers = nn.Parameter(centers.unsqueeze(0).expand(in_features, -1).clone())
        # Log-widths (learnable); sigma = exp(log_width)
        self.log_widths = nn.Parameter(torch.zeros(in_features, num_basis))
        # Main path: (in_features x num_basis) -> out_features
        self.rbf_proj = nn.Linear(in_features * num_basis, out_features)
        # Auxiliary path: in_features -> out_features
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (..., in_features)
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)          # (N, F)
        # RBF activations: phi(x_j) for each input dimension j
        x_exp = x_flat.unsqueeze(-1)                       # (N, F, 1)
        c = self.centers.unsqueeze(0)                      # (1, F, M)
        sigma = self.log_widths.exp().unsqueeze(0)         # (1, F, M)
        rbf = torch.exp(-((x_exp - c) ** 2) / (2 * sigma ** 2))  # (N, F, M)
        main = self.rbf_proj(rbf.reshape(x_flat.size(0), -1))     # (N, out)
        aux = self.linear(x_flat)                          # (N, out)
        out = self.norm(main + aux)
        return out.reshape(*shape[:-1], out.size(-1))


class FKAN(nn.Module):
    """Fast Kernel Activation Network for blood pressure regression (Eq. 17).

    Stacks FastKANLayer modules with a final linear output layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        out_features: int = 2,
        num_layers: int = 2,
        num_basis: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_f = in_features
        for _ in range(num_layers - 1):
            layers.append(FastKANLayer(in_f, hidden_features, num_basis))
            layers.append(nn.Dropout(dropout))
            in_f = hidden_features
        layers.append(nn.Linear(in_f, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# ACFA (full model)
# -----------------------------------------------------------------------------


@register_model("acfa")
class ACFA(nn.Module):
    """Adaptive Cross-domain Fusion Architecture for cuffless BP estimation.

    Pipeline (Section III-C, Module Integration Strategy):

        PPG (B, L)  ->  DyCASNet  ->  stride-4 pool  ->  xLSTM  ->  Transformer
                    ->  global avg pool  ->  FKAN  ->  [SBP, DBP]

    Args:
        d_model:       Channel / embedding dimension used throughout (default 64).
        xlstm_layers:  Total number of xLSTM layers; even layers are sLSTM,
                       odd layers are mLSTM (default 4).
        xlstm_heads:   Attention heads inside mLSTM blocks (default 4).
        tr_layers:     Number of Transformer encoder layers (default 2).
        tr_nhead:      Transformer attention heads (default 4).
        fkan_hidden:   Hidden width of FastKAN layers (default 128).
        fkan_layers:   Number of FastKAN layers before the output head (default 2).
        num_basis:     RBF basis functions per input dimension in FastKANLayer (default 8).
        num_kernels:   Dynamic convolution kernel sets in DCB (default 4).
        reduction:     SE bottleneck reduction factor in CASB and DCB (default 4).
        pool_stride:   Temporal downsampling factor applied after DyCASNet
                       to reduce the sequence length before xLSTM / Transformer
                       (default 4 -> 1 000 -> 250 time steps).
        dropout:       Dropout rate in Transformer and FKAN (default 0.1).
        input_length:  Input PPG segment length in samples (default 1 000).
        out_features:  Output dimension -- 2 for [SBP, DBP] (default 2).
    """

    def __init__(
        self,
        d_model: int = 64,
        xlstm_layers: int = 4,
        xlstm_heads: int = 4,
        tr_layers: int = 2,
        tr_nhead: int = 4,
        fkan_hidden: int = 128,
        fkan_layers: int = 2,
        num_basis: int = 8,
        num_kernels: int = 4,
        reduction: int = 4,
        pool_stride: int = 4,
        dropout: float = 0.1,
        input_length: int = 1000,
        out_features: int = 2,
    ):
        super().__init__()
        # Derived sequence length after temporal pooling
        seq_len = math.ceil(input_length / pool_stride)

        self.dycasnet = DyCASNet(
            in_channels=1,
            d_model=d_model,
            num_kernels=num_kernels,
            reduction=reduction,
        )
        # Stride-4 average pooling to keep Transformer attention tractable
        self.temporal_pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.xlstm = xLSTMStack(d_model, num_layers=xlstm_layers, num_heads=xlstm_heads)
        self.transformer = TransformerBranch(
            d_model=d_model,
            nhead=tr_nhead,
            num_layers=tr_layers,
            seq_len=seq_len,
            dropout=dropout,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fkan = FKAN(
            in_features=d_model,
            hidden_features=fkan_hidden,
            out_features=out_features,
            num_layers=fkan_layers,
            num_basis=num_basis,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: PPG waveform tensor of shape (B, L) or (B, 1, L).

        Returns:
            Tensor of shape (B, 2) with [SBP, DBP] predictions in mmHg.
        """
        x = ensure_3d(x)                        # (B, 1, L)
        # 1. DyCASNet - dual-domain feature extraction
        x = self.dycasnet(x)                    # (B, d_model, L)
        # 2. Temporal pooling - reduce sequence length
        x = self.temporal_pool(x)              # (B, d_model, L/4)
        # 3. Reshape to sequence format for xLSTM / Transformer
        x = x.transpose(1, 2)                  # (B, L/4, d_model)
        # 4. xLSTM - local + long-range temporal modelling
        x = self.xlstm(x)                      # (B, L/4, d_model)
        # 5. Transformer - global context
        x = self.transformer(x)               # (B, L/4, d_model)
        # 6. Global average pooling + FKAN regression
        x = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        return self.fkan(x)                    # (B, out_features)
