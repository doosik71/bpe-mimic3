"""ResNet1D37: 37-layer variant of ResNet1D61 (~60% of its layer count).

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Fully length-agnostic (global/adaptive pooling throughout) -- no dimension
changes were needed for this project's 1,000-sample MIMIC-III input.

ResNet1D61: 4 stages x 2 BasicBlock1D = 8 blocks, 61 layers.
ResNet1D37: 4 stages x 1 BasicBlock1D = 4 blocks, 37 layers.
Channel widths and the stem are unchanged.
"""

from bpe.models.registry import register_model
from bpe.models.resnet1d61 import BasicBlock1D, ResNet1D61


@register_model("resnet1d37")
class ResNet1D37(ResNet1D61):
    """Halved-depth 1D ResNet -- 4 residual blocks across 4 stages."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_features=out_features,
            base_channels=base_channels,
            layers=(1, 1, 1, 1),
            block=BasicBlock1D,
            dropout=dropout,
        )
