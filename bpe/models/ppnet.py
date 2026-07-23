"""PP-Net: CNN-LSTM (LRCN) framework for PPG-based blood pressure estimation.

Ported from the VitalDB-era experiments in temp/__legacy_vitaldb_models.
Already targets 1,000-sample (8 s @ 125 Hz) input -- no dimension changes
were needed for this project's MIMIC-III dataset.

Reference:
    Panwar M. et al., "PP-Net: A Deep Learning Framework for PPG-Based
    Blood Pressure and Heart Rate Estimation," IEEE Sensors Journal,
    vol. 20, no. 17, pp. 10000-10011, Sep. 2020.
    https://doi.org/10.1109/JSEN.2020.2990864

Architecture -- Long-term Recurrent Convolutional Network (LRCN):
    Input:       PPG waveform (B, 1000) -- 8 s @ 125 Hz
    Downsample:  AvgPool1d(4, stride=4) -> (B, 1, 250)
                 [mirrors the 4x down-sampling applied as pre-processing in the
                 paper; Section III-A and Table II show this beats inherent stride]
    CNN Block 1: Conv1d(1->20, kernel=9, same-pad) -> ReLU
                 -> MaxPool1d(4) -> Dropout(0.1)   -> (B, 20, 62)
    CNN Block 2: Conv1d(20->20, kernel=9, same-pad) -> ReLU
                 -> MaxPool1d(4) -> Dropout(0.1)   -> (B, 20, 15)
    Reshape:     (B, 20, 15) -> (B, 15, 20) -- time-first for LSTM
    LSTM 1:      input_size=20, hidden_size=64, tanh -> Dropout(0.1)
    LSTM 2:      input_size=64, hidden_size=128, tanh -> Dropout(0.1)
    FC:          Linear(128, 2) -- [SBP, DBP] in mmHg

Hyperparameters taken directly from the paper (Section III-B / Fig. 3):
    cnn_filters=20, kernel_size=9, pool_size=4, lstm_units=(64, 128), dropout=0.1

Adaptations from the paper:
    * Input: 1,000-sample PPG (8 s @ 125 Hz); model downsamples 4x internally
      to 250 samples, matching the paper's intended CNN input length.
    * Output: [SBP, DBP] (2-dimensional) instead of [DBP, SBP, HR]
      (3-dimensional) -- heart-rate estimation is out of scope for this project.
"""

import torch
from torch import nn

from bpe.models._blocks import ensure_3d
from bpe.models.registry import register_model


@register_model("ppnet")
class PPNet(nn.Module):
    """PP-Net: Long-term Recurrent Convolutional Network for cuffless BP estimation.

    A hybrid CNN-LSTM architecture that first extracts local spatial features
    with two 1-D convolutional blocks, then models temporal sequential
    dependencies with two stacked LSTM layers before regressing SBP and DBP.

    Sequence lengths for a 1000-sample input (default):
        1000  --AvgPool(4)-->  250
              --Conv+MaxPool(4)-->  62
              --Conv+MaxPool(4)-->  15   (LSTM time steps)

    Args:
        cnn_filters:   Number of output filters in each CNN layer.  Default: 20.
        kernel_size:   Conv1d kernel width (same-padding applied).  Default: 9.
        pool_size:     MaxPool kernel size and stride.  Default: 4.
        lstm_units:    Hidden-unit counts for the two LSTM layers.  Default: (64, 128).
        dropout:       Dropout probability applied after each pooling and LSTM
                       layer.  Default: 0.1 (paper Section III-B).
        out_features:  Output dimension -- 2 for [SBP, DBP].  Default: 2.
    """

    def __init__(
        self,
        cnn_filters: int = 20,
        kernel_size: int = 9,
        pool_size: int = 4,
        lstm_units: tuple[int, int] = (64, 128),
        dropout: float = 0.1,
        out_features: int = 2,
    ) -> None:
        super().__init__()

        # 4x down-sample: 1000 -> 250  (Section III-A: "scaling factor of 4")
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4)

        padding = (kernel_size - 1) // 2  # same-length convolution

        # CNN block 1 -- (B, 1, 250) -> (B, cnn_filters, 62)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, cnn_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(dropout),
        )

        # CNN block 2 -- (B, cnn_filters, 62) -> (B, cnn_filters, 15)
        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout(dropout),
        )

        lstm1_units, lstm2_units = lstm_units

        # LSTM layers -- paper: tanh activation (PyTorch LSTM default)
        self.lstm1 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm1_units,
            batch_first=True,
        )
        self.drop_lstm1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=lstm1_units,
            hidden_size=lstm2_units,
            batch_first=True,
        )
        self.drop_lstm2 = nn.Dropout(dropout)

        self.fc = nn.Linear(lstm2_units, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BP predictions of shape (B, 2) -- [SBP, DBP].

        Args:
            x: PPG waveform tensor of shape (B, L) or (B, 1, L).
        """
        x = ensure_3d(x)           # (B, 1, 1000)

        x = self.downsample(x)     # (B, 1, 250)

        x = self.cnn1(x)           # (B, 20, 62)
        x = self.cnn2(x)           # (B, 20, 15)

        x = x.transpose(1, 2)      # (B, 15, 20) -- time-first for LSTM

        x, _ = self.lstm1(x)       # (B, 15, 64)
        x = self.drop_lstm1(x)

        x, _ = self.lstm2(x)       # (B, 15, 128)
        x = self.drop_lstm2(x)

        x = x[:, -1, :]            # final time step: (B, 128)
        return self.fc(x)           # (B, 2)
