"""
MÔ HÌNH: 1D-VISION TRANSFORMER (ViT-1D) CHO CHUỖI THỜI GIAN HAR
Input : (B, 6, 128)
Output: (B, 128, 8)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ViT1DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        patch_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # Linear Patch Projection qua Conv1d không chồng lấn (stride = patch_size)
        self.patch_embed = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=32)
        self.dropout = nn.Dropout(p=dropout)

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
        tokens = self.patch_embed(x)     # (B, 128, 8)
        tokens = tokens.transpose(1, 2)  # (B, 8, 128)
        tokens = self.pos_encoder(tokens)
        tokens = self.dropout(tokens)
        out = self.transformer_encoder(tokens)  # (B, 8, 128)
        out = self.norm(out)
        return out.transpose(1, 2)       # Trả về (B, 128, 8)