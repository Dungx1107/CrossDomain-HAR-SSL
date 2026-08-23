"""
===============================================================================
MÔ HÌNH PHÂN LOẠI HOÀN CHỈNH (SUPERVISED HAR MODEL)
===============================================================================
Mục đích:
    Kết hợp giữa Encoder (Trích xuất đặc trưng) và ClassifierHead (Phân loại).
    Nhận Tensor cảm biến thô (B, Channels, Window) -> Cho ra Dự đoán (B, Num_Classes)
===============================================================================
"""

import torch
import torch.nn as nn
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead


class SupervisedHARModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 6,
        feature_dim: int = 128,
        encoder: nn.Module = None,
        classifier: nn.Module = None
    ):
        """
        Khởi tạo mô hình Supervised HAR.
        Nếu không truyền encoder/classifier thì tự động tạo theo in_channels và num_classes.
        """
        super().__init__()
        self.encoder = encoder if encoder is not None else StandardSensorEncoder1D(
            in_channels=in_channels,
            feature_dim=feature_dim
        )
        self.classifier = classifier if classifier is not None else ClassifierHead(
            feature_dim=feature_dim,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng dữ liệu:
            x (B, Channels, Window) -> Encoder -> features (B, 128) -> Classifier -> logits (B, Num_Classes)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
