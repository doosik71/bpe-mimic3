# Import the registry first: loading it runs every @register_model decorator
# (including SpectroCNN and SpectroSiamese) via its side-effect imports. The
# class re-exports below then hit already-loaded modules, avoiding the
# circular import that would arise if the classes were imported first.
from bpe.models.registry import (
    build_calibration_based_model,
    build_calibration_free_model,
    list_calibration_based_models,
    list_calibration_free_models,
)
from bpe.models.spectro_cnn import SpectroCNN
from bpe.models.spectro_siamese import SpectroSiamese

# _backbone / _frontend are internal building blocks (see their leading
# underscore); models are selected through the registry, so the public API of
# this package is the build/list helpers plus the concrete model classes.
__all__ = [
    "SpectroCNN",
    "SpectroSiamese",
    "build_calibration_based_model",
    "build_calibration_free_model",
    "list_calibration_based_models",
    "list_calibration_free_models",
]
