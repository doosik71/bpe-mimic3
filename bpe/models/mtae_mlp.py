"""Multi-Task AutoEncoder with separate SBP/DBP MLP heads (MTAE_MLP).

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Already targets 1,000-sample (8 s @ 125 Hz) input -- no dimension changes
were needed for this project's MIMIC-III dataset.
"""

import torch
from torch import nn

from bpe.models.blocks import ConvBnAct1d, ensure_3d
from bpe.models.registry import register_model


SBP_HIDDEN_DIM: int = 16
DBP_HIDDEN_DIM: int = 16


class _Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnAct1d(1, 32, 7, stride=2),    # (B,  32, 500)
            ConvBnAct1d(32, 64, 7, stride=2),   # (B,  64, 250)
            ConvBnAct1d(64, 128, 5, stride=2),  # (B, 128, 125)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = self.pool(self.conv(x)).flatten(1)
        return torch.sigmoid(self.fc(x))


class _Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128)
        self.up = nn.Sequential(
            nn.Upsample(125),
            ConvBnAct1d(128, 64, 5),
            nn.Upsample(250),
            ConvBnAct1d(64, 32, 7),
            nn.Upsample(500),
            ConvBnAct1d(32, 16, 7),
            nn.Upsample(1000),
            nn.Conv1d(16, 1, 7, padding=3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.fc(z).unsqueeze(-1))  # (B, 1, 1000)


class _MLPHead(nn.Module):
    """2-layer MLP regression head producing a single scalar output."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


@register_model("mtae_mlp")
class MTAE_MLP(nn.Module):
    """Multi-Task AutoEncoder with separate SBP/DBP MLP heads.

    Same encoder/decoder as MTAE, but the single shared linear BP head is
    replaced with two independent MLP heads -- one for SBP, one for DBP --
    since SBP consistently shows higher error than DBP with a shared head.
    Their outputs are concatenated to preserve the (B, 2) output interface.

    The combined loss used during training is::

        loss = (1 - recon_weight) * bp_loss + recon_weight * recon_loss

    where both sub-losses use the same criterion passed by the trainer.

    Note: bpe/trainer.py does not yet call `compute_loss` when present --
    training this model today falls back to plain L1 loss on `forward()`'s
    output, ignoring the reconstruction term, until the trainer is updated.

    Args:
        latent_dim:     Bottleneck size.  Default: 16.
        sbp_hidden_dim: Hidden width of the SBP MLP head.  Default: SBP_HIDDEN_DIM.
        dbp_hidden_dim: Hidden width of the DBP MLP head.  Default: DBP_HIDDEN_DIM.
        head_dropout:   Dropout probability inside each MLP head.  Default: 0.1.
        recon_weight:   Reconstruction loss weight in [0, 1].  Default: 0.5.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        sbp_hidden_dim: int = SBP_HIDDEN_DIM,
        dbp_hidden_dim: int = DBP_HIDDEN_DIM,
        head_dropout: float = 0.1,
        recon_weight: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.recon_weight = recon_weight
        self.encoder = _Encoder(latent_dim)
        self.decoder = _Decoder(latent_dim)
        self.sbp_head = _MLPHead(latent_dim, sbp_hidden_dim, head_dropout)
        self.dbp_head = _MLPHead(latent_dim, dbp_hidden_dim, head_dropout)

    def _predict(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.sbp_head(z), self.dbp_head(z)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BP predictions, shape (B, 2) -- [SBP, DBP]."""
        return self._predict(self.encoder(x))

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-task loss; called by Trainer when this method exists.

        Returns:
            (loss, pred) where ``pred`` has shape (B, 2).
        """
        x3d = ensure_3d(x)
        z = self.encoder(x3d)
        pred = self._predict(z)
        recon = self.decoder(z)

        bp_loss = criterion(pred, y)
        recon_loss = criterion(recon, x3d)

        loss = (1.0 - self.recon_weight) * bp_loss + self.recon_weight * recon_loss
        return loss, pred
