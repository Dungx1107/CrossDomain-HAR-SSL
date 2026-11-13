"""
===============================================================================
MODULE ENCODER CNN-LSTM ĐAN XEN (INTERLEAVED) CHO CẢM BIẾN HAR
===============================================================================
Đặt file tại: models/encoders/cnn_lstm_interleaved.py

Ý tưởng:
    Sau MỖI khối CNN là một lớp LSTM, rồi mới sang khối CNN kế tiếp:
        x -> [Conv-BN-ReLU-Pool] -> LSTM -> [Conv-BN-ReLU-Pool] -> LSTM -> ...

    Khác với `cnn_lstm` (4 khối CNN rồi 1 LSTM ở cuối), ở đây thông tin tuần tự
    được mô hình hóa ở nhiều độ phân giải thời gian (T = 64, 32, 16, 8).

Input : (B, in_channels, 128)
Output: (B, feature_dim, 8)    <- CÙNG dạng với `standard`

Luồng shape (mặc định):
    (B, 6, 128)
    block1 conv7 -> (B, 32, 64)  -> lstm1 (hidden 32)
    block2 conv5 -> (B, 64, 32)  -> lstm2 (hidden 64)
    block3 conv5 -> (B, 96, 16)  -> lstm3 (hidden 96)
    block4 conv3 -> (B, 128, 8)  -> lstm4 (hidden 128)

Lưu ý:
    - Mặc định LSTM MỘT CHIỀU (bidirectional=False) để không rò rỉ thông tin
      tương lai vào nhánh TC của TS-TCC. Chỉ bật bidirectional=True cho
      SwAV / CrossHAR.
    - Với bidirectional=True, hidden mỗi chiều = số kênh // 2 nên shape không đổi.
    - Ước lượng tham số (mặc định): ~79k (CNN) + ~248k (4 LSTM) ≈ 328k.
===============================================================================
"""

from typing import Sequence

import torch
import torch.nn as nn


class InterleavedCNNLSTMEncoder(nn.Module):
    # Cấu hình giống StandardSensorEncoder1D (kênh cuối = feature_dim)
    _KERNELS = (7, 5, 5, 3)
    _MID_CHANNELS = (32, 64, 96)

    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 128,
        lstm_at: Sequence[int] = (1, 2, 3, 4),
        bidirectional: bool = False,
        lstm_residual: bool = False,
    ):
        """
        Args:
            in_channels  : Số kênh cảm biến đầu vào.
            feature_dim  : Số kênh đặc trưng đầu ra (128).
            lstm_at      : Các khối (đánh số 1..4) có LSTM đi sau. Mặc định cả 4.
                           Dùng để ablation, vd (3, 4) hoặc (4,).
            bidirectional: False (mặc định) để an toàn với TC của TS-TCC.
            lstm_residual: True -> x = x + LSTM(x) (skip connection quanh LSTM).
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.bidirectional = bidirectional
        self.lstm_residual = lstm_residual
        self.lstm_at = tuple(sorted(set(lstm_at)))

        if any(i < 1 or i > 4 for i in self.lstm_at):
            raise ValueError("lstm_at chỉ được chứa các số trong 1..4")

        channels = list(self._MID_CHANNELS) + [feature_dim]
        if bidirectional and any(c % 2 != 0 for c in channels):
            raise ValueError("Số kênh mỗi khối phải chẵn khi bidirectional=True")

        prev = in_channels
        for i, (ch, k) in enumerate(zip(channels, self._KERNELS), start=1):
            self.add_module(
                f"block{i}",
                nn.Sequential(
                    nn.Conv1d(prev, ch, kernel_size=k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ),
            )
            if i in self.lstm_at:
                self.add_module(
                    f"lstm{i}",
                    nn.LSTM(
                        input_size=ch,
                        hidden_size=ch // 2 if bidirectional else ch,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=bidirectional,
                    ),
                )
            prev = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, 128)
        for i in range(1, 5):
            x = getattr(self, f"block{i}")(x)            # (B, C_i, T_i)
            if i in self.lstm_at:
                y, _ = getattr(self, f"lstm{i}")(x.transpose(1, 2))  # (B, T_i, C_i)
                y = y.transpose(1, 2)                    # (B, C_i, T_i)
                x = x + y if self.lstm_residual else y.contiguous()
        return x                                         # (B, feature_dim, 8)