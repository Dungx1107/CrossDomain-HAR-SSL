"""
FACTORY BUILDER KHỞI TẠO ENCODER/BACKBONE CHO TS-TCC
"""
import torch
import torch.nn as nn
from models.encoders.cnn_transformer import CNNTransformerEncoder
from models.encoders.vit_1d import ViT1DEncoder
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.encoders.tstcc_encoder import TSTCCEncoder
from models.encoders.cnn_lstm import CNNLSTMEncoder
from models.encoders.cnn_lstm_interleaved import InterleavedCNNLSTMEncoder


def build_encoder(
        backbone_type: str = "standard",
        in_channels: int = 6) -> nn.Module:
    """
    Args:
        backbone_type: 'standard' | 'tstcc' | 'vit_1d' | 'cnn_transformer' | 'cnn_lstm' | 'cnn_lstm_il'
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
    elif backbone_type == "tstcc":
        return TSTCCEncoder(
            in_channels=in_channels,
            feature_dim=128,
            dropout=0.35
        )
    elif backbone_type == "cnn_lstm":
        # Standard 1D-CNN + 1 lớp LSTM một chiều (an toàn cho TC của TS-TCC)
        return CNNLSTMEncoder(
            in_channels=in_channels,
            feature_dim=128,
            lstm_layers=1,
            bidirectional=False
        )

    elif backbone_type == "cnn_lstm_il":
        # CNN-LSTM đan xen: sau mỗi khối CNN là một LSTM một chiều
        return InterleavedCNNLSTMEncoder(
            in_channels=in_channels,
            feature_dim=128,
            lstm_at=(1, 2, 3, 4),
            bidirectional=False,
            lstm_residual=False
        )

    else:
        raise ValueError(
            f"❌ Không hỗ trợ backbone_type='{backbone_type}'. "
            f"Lựa chọn hợp lệ: ['standard', 'tstcc', 'cnn_transformer', 'vit_1d', 'cnn_lstm', 'cnn_lstm_il']"
        )

if __name__ =="__main__":
    for b in ["standard", "tstcc", "cnn_transformer", "vit_1d", "cnn_lstm", "cnn_lstm_il"]:
        enc = build_encoder(b, in_channels=6)
        n_params = sum(p.numel() for p in enc.parameters())
        with torch.no_grad():
            out = enc(torch.randn(2, 6, 128))
        print(f"Backbone: {b:<16} | Output Shape: {list(out.shape)} | Params: {n_params:,}")

    enc = build_encoder("vit_1d", in_channels=6)
    enc.eval()
    with torch.no_grad():
        out1 = enc(torch.randn(2, 6, 128))  # L = 128
        out2 = enc(torch.randn(2, 6, 256))  # L = 256 (gấp đôi)
    print("Output với L=128:", list(out1.shape))
    print("Output với L=256:", list(out2.shape))