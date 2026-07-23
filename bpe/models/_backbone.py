"""Shared AlexNet-inspired feature-extraction backbone (docs/method-spectrogram-cnn.md §2,
docs/development-plan.md §5.1): 5 conv layers + the first 2 of 3 FC layers,
producing an embedding vector. Both the calibration-free CNN and the
Siamese calibration-based model build on this same backbone -- the
calibration-free model adds a direct regression head, the Siamese model
runs two copies (weight-shared, since it's a single instance) and regresses
on their difference. See bpe/models/spectro_cnn.py and
bpe/models/spectro_siamese.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bpe.models._frontend import LogSpectrogram
from bpe.preprocess.pipeline import DEFAULT_TARGET_FS, DEFAULT_WINDOW_SEC

DEFAULT_INPUT_SAMPLES = int(round(DEFAULT_WINDOW_SEC * DEFAULT_TARGET_FS))
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_DROPOUT = 0.5


class _ConvBnReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class PPGFeatureBackbone(nn.Module):
    """Spectrogram -> embedding vector.

    Conv stack: conv1 -> pool, conv2 -> pool, conv3 -> conv4 (directly
    connected, no pooling between), conv5 -> pool -- matching method-spectrogram-cnn.md's
    "max pooling after the 1st/2nd/5th conv layer, 3rd and 4th directly
    connected" description. Batch norm after every conv, ReLU throughout.
    FC1/FC2 each preceded by dropout, matching AlexNet's placement (dropout
    before fc6/fc7, not before the final output layer).

    Channel counts, kernel sizes, and `embedding_dim` are not specified by
    method-spectrogram-cnn.md (only the AlexNet-inspired *structure* is) -- these are this
    project's own choice, sized down from AlexNet's original 224x224-image
    hyperparameters to fit the much smaller spectrogram input.
    """

    def __init__(
        self,
        fs: float = DEFAULT_TARGET_FS,
        input_samples: int = DEFAULT_INPUT_SAMPLES,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.frontend = LogSpectrogram(fs)

        self.conv1 = _ConvBnReLU(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = _ConvBnReLU(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = _ConvBnReLU(64, 128, kernel_size=3, padding=1)
        self.conv4 = _ConvBnReLU(128, 128, kernel_size=3, padding=1)
        self.conv5 = _ConvBnReLU(128, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2)

        flatten_dim = self._infer_flatten_dim(input_samples)

        self.fc1 = nn.Sequential(nn.Dropout(dropout), nn.Linear(
            flatten_dim, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(
            512, embedding_dim), nn.ReLU(inplace=True))
        self.embedding_dim = embedding_dim

    def _conv_stack(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(spec))
        x = self.pool2(self.conv2(x))
        x = self.conv4(self.conv3(x))
        x = self.pool5(self.conv5(x))
        return x

    def _infer_flatten_dim(self, input_samples: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_samples)
            spec = self.frontend(dummy)
            out = self._conv_stack(spec)
        return out.numel()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """`waveform`: `(batch, samples)` -> `(batch, embedding_dim)`."""
        spec = self.frontend(waveform)
        x = self._conv_stack(spec)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
