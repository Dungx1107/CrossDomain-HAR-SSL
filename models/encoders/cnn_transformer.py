"""
MÔ HÌNH: CNN-TRANSFORMER HYBRID ENCODER CHO CHUỖI THỜI GIAN HAR
Input : (B, 6, 128)
Output: (B, 128, 32)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Mã hóa vị trí hình sin cho chuỗi thời gian"""

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class CNNTransformerEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 6,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.1
    ):
        super().__init__()

        # 1. 1D-CNN Stem: Bắt sóng vi mô và nén bước thời gian 128 -> 32
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (B, 64, 64)

            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # (B, 128, 32)
        )

        # 2. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=64)
        self.dropout = nn.Dropout(p=dropout)

        # 3. Transformer Encoder (Multi-Head Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 6, 128)
        feat = self.cnn_stem(x)  # (B, 128, 32)
        feat = feat.transpose(1, 2)  # (B, 32, 128)
        feat = self.pos_encoder(feat)
        feat = self.dropout(feat)
        out = self.transformer_encoder(feat)  # (B, 32, 128)
        out = self.norm(out)
        return out.transpose(1, 2)  # Trả về (B, 128, 32)