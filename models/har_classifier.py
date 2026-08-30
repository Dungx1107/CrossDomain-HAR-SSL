"""
===============================================================================
    Mô hình nhận diện hoạt động con người (HAR) tổng quát.
    Ghép 1 Encoder (Backbone) với 1 ClassifierHead.
    Dùng chung cho:
      - Baseline Supervised Training
      - SSL Linear Probing
      - SSL Full Fine-Tuning
      - Cross-Domain Transfer Evaluation
    Nhận Tensor cảm biến thô (B, Channels, Window) -> Cho ra Dự đoán (B, Num_Classes)

===============================================================================
"""

import torch
import torch.nn as nn
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead


class HARClassifier(nn.Module):
    def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = 5,
            feature_dim: int = 128,
            encoder: nn.Module = None,
            classifier: nn.Module = None
    ):
        """
        Khởi tạo mô hình HAR phân loại đa lớp.

        Args:
            in_channels (int): Số kênh cảm biến đầu vào (mặc định 6: Acc + Gyro).
            num_classes (int): Số lượng nhãn hoạt động (mặc định 5 cho Cross-Domain).
            feature_dim (int): Số chiều véc-tơ đặc trưng từ Encoder (mặc định 128).
            encoder (nn.Module, optional): Backbone tùy chỉnh nếu muốn inject từ ngoài vào.
            classifier (nn.Module, optional): Head tùy chỉnh nếu muốn inject từ ngoài vào.
        """
        super().__init__()

        # 1. Khung xương trích xuất đặc trưng (Backbone)
        self.encoder = encoder if encoder is not None else StandardSensorEncoder1D(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        # 2. Đầu phân loại nhiệm vụ (Classifier Head)
        self.classifier = classifier if classifier is not None else ClassifierHead(
            feature_dim=feature_dim,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng dữ liệu:
            x (B, in_channels, seq_len)
            -> Encoder -> features (B, feature_dim)
            -> ClassifierHead -> logits (B, num_classes)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits