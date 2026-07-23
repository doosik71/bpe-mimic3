"""Name -> constructor registry so `train-model.py --model <name>` can
select an architecture without the training/eval code depending on any
specific model class. Adding a new architecture later is a pure addition:
decorate its constructor with `@register_model(...)` and make sure the
module is imported below so the decorator runs.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_CALIBRATION_FREE_MODELS: dict[str, Callable[..., nn.Module]] = {}

_CALIBRATION_BASED_MODELS: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str, *, calibration_based: bool = False):
    """Decorator: register a model constructor by name.

    Lets a model module self-register on import instead of needing a manual
    registry entry here. `calibration_based=False` (the default) registers a
    calibration-free architecture (predicts BP from a single window);
    `calibration_based=True` registers a model that needs a patient-specific
    reference reading (e.g. SpectroSiamese). The two kinds live in separate
    dicts so `train-model.py` can pick the right build path.
    """

    def decorator(constructor: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        key = name.strip().lower().replace("-", "_")
        if not key:
            raise ValueError("model name must not be empty")
        registry = _CALIBRATION_BASED_MODELS if calibration_based else _CALIBRATION_FREE_MODELS
        if key in registry:
            raise ValueError(f"model already registered: {key}")
        registry[key] = constructor
        return constructor

    return decorator


# Importing these triggers their @register_model(...) decorators above, so
# they must come after `register_model` is defined. resnet1d61 comes first
# since resnet1d13/21/37 and st_resnet import BasicBlock1D from it. The
# resnet/mtae/etc. architectures are ported from temp/__legacy_vitaldb_models.
from bpe.models import spectro_cnn as _spectro_cnn  # noqa: E402,F401
from bpe.models import spectro_siamese as _spectro_siamese  # noqa: E402,F401
from bpe.models import resnet1d61 as _resnet1d61  # noqa: E402,F401
from bpe.models import acfa as _acfa  # noqa: E402,F401
from bpe.models import ae_lstm as _ae_lstm  # noqa: E402,F401
from bpe.models import bpnet_cf as _bpnet_cf  # noqa: E402,F401
from bpe.models import conv_reg as _conv_reg  # noqa: E402,F401
from bpe.models import mtae as _mtae  # noqa: E402,F401
from bpe.models import mtae_mlp as _mtae_mlp  # noqa: E402,F401
from bpe.models import pctn as _pctn  # noqa: E402,F401
from bpe.models import ppnet as _ppnet  # noqa: E402,F401
from bpe.models import resnet1d13 as _resnet1d13  # noqa: E402,F401
from bpe.models import resnet1d21 as _resnet1d21  # noqa: E402,F401
from bpe.models import resnet1d37 as _resnet1d37  # noqa: E402,F401
from bpe.models import st_resnet as _st_resnet  # noqa: E402,F401


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
