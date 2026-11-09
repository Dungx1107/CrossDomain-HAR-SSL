"""
===============================================================================
MODULE ENCODER CNN-LSTM (STANDARD 1D-CNN + LSTM) CHO CẢM BIẾN HAR
===============================================================================
Đặt file tại: models/encoders/cnn_lstm.py

Ý tưởng:
    Giữ NGUYÊN 4 khối CNN của StandardSensorEncoder1D (kế thừa, nên tên tham số
    block1..block4 và state_dict của phần CNN y hệt `standard`), rồi thêm 1 lớp
    LSTM phía trên chuỗi đặc trưng. Nhờ vậy so sánh `standard` vs `cnn_lstm`
    cô lập được đúng một yếu tố: có/không có thành phần hồi quy.

Input : (B, in_channels, 128)
Output: (B, feature_dim, 8)    <- CÙNG dạng với `standard`, nên dùng được cho
                                   TS-TCC / hybrid (TC cần 3D), SwAV (GAP),
                                   CrossHAR và ClassifierHead hiện có.

Lưu ý quan trọng:
    - Mặc định LSTM MỘT CHIỀU (bidirectional=False). Nhánh TC của TS-TCC cắt
      chuỗi quá khứ z[:, :t+1] SAU khi encoder đã chạy trên toàn chuỗi; với
      BiLSTM, đặc trưng tại bước t đã chứa thông tin tương lai nên việc dự đoán
      z_{t+k} bị rò rỉ. Chỉ bật bidirectional=True cho SwAV/CrossHAR.
    - BiLSTM dùng hidden = feature_dim // 2 mỗi chiều để tổng vẫn là feature_dim.
===============================================================================
"""

import torch
import torch.nn as nn

from models.encoders.cnn1d import StandardSensorEncoder1D


class CNNLSTMEncoder(StandardSensorEncoder1D):
    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        lstm_dropout: float = 0.0,
    ):
        """
        Args:
            in_channels  : Số kênh cảm biến đầu vào.
            feature_dim  : Số kênh đặc trưng đầu ra (128).
            lstm_layers  : Số lớp LSTM (1 theo khuyến nghị "shallow LSTM" của Bock et al.).
            bidirectional: False (mặc định) để an toàn với TC của TS-TCC.
            lstm_dropout : Dropout giữa các lớp LSTM (chỉ có tác dụng khi lstm_layers > 1).
        """
        super().__init__(in_channels=in_channels, feature_dim=feature_dim)

        if bidirectional and feature_dim % 2 != 0:
            raise ValueError("feature_dim phải chẵn khi bidirectional=True")

        self.bidirectional = bidirectional
        hidden_size = feature_dim // 2 if bidirectional else feature_dim

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, 128)
        feat = super().forward(x)                   # (B, 128, 8)  (4 khối CNN của standard)
        out, _ = self.lstm(feat.transpose(1, 2))    # (B, 8, 128)
        return out.transpose(1, 2).contiguous()     # (B, 128, 8)