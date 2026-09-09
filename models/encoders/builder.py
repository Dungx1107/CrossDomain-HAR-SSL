"""
FACTORY BUILDER KHỞI TẠO ENCODER/BACKBONE CHO TS-TCC
"""

import torch.nn as nn
from models.encoders.cnn_transformer import CNNTransformerEncoder
from models.encoders.vit_1d import ViT1DEncoder
from models.encoders.cnn1d import StandardSensorEncoder1D


def build_encoder(
        backbone_type: str = "standard",
        in_channels: int = 6) -> nn.Module:
    """
    Args:
        backbone_type: 'standard' | 'vit_1d' | 'cnn_transformer'
        in_channels: Số kênh cảm biến (mặc định là 6)
    """
    if backbone_type == "standard":
        return StandardSensorEncoder1D(
            in_channels=in_channels,
            feature_dim=128
        )
    elif backbone_type == "cnn_transformer":
        return CNNTransformerEncoder(
            in_channels=in_channels,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1
        )

    elif backbone_type == "vit_1d":
        return ViT1DEncoder(
            in_channels=in_channels,
            patch_size=16,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1
        )

    else:
        raise ValueError(
            f"❌ Không hỗ trợ backbone_type='{backbone_type}'. "
            f"Lựa chọn hợp lệ: ['standard', 'cnn_transformer', 'vit_1d']"
        )

