"""Name -> constructor registry so `train-model.py --model <name>` can
select an architecture without the training/eval code depending on any
specific model class. Adding a new calibration-free architecture later is a
pure addition to `_CALIBRATION_FREE_MODELS`.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from bpe.models.calibration_free import CalibrationFreeCNN
from bpe.models.siamese import SiameseCalibrationModel

_CALIBRATION_FREE_MODELS: dict[str, Callable[..., nn.Module]] = {
    "cnn": CalibrationFreeCNN,
}

_CALIBRATION_BASED_MODELS: dict[str, Callable[..., nn.Module]] = {
    "siamese": SiameseCalibrationModel,
}


def build_calibration_free_model(name: str, **kwargs) -> nn.Module:
    try:
        constructor = _CALIBRATION_FREE_MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown calibration-free model {name!r}; available: {sorted(_CALIBRATION_FREE_MODELS)}"
        ) from None
    return constructor(**kwargs)


def build_calibration_based_model(name: str, **kwargs) -> nn.Module:
    try:
        constructor = _CALIBRATION_BASED_MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown calibration-based model {name!r}; available: {sorted(_CALIBRATION_BASED_MODELS)}"
        ) from None
    return constructor(**kwargs)


def list_calibration_free_models() -> list[str]:
    return sorted(_CALIBRATION_FREE_MODELS)


def list_calibration_based_models() -> list[str]:
    return sorted(_CALIBRATION_BASED_MODELS)
