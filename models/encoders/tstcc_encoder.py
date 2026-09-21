"""
===============================================================================
MODULE: TS-TCC OFFICIAL 1D-CNN BACKBONE ENCODER (EXACT 100% REPRODUCTION)
===============================================================================
Mã nguồn đối chiếu trực tiếp từ class `base_Model` trong repository chính thức:
    emadeldeen24/TS-TCC (IJCAI 2021)

Chi tiết cấu hình và luồng tensor chuẩn cho input (B, 6, 128):
    1. Input: (B, 6, 128)
    2. Block 1:
       - Conv1d: in=6, out=32, k=kernel_size, stride=1, padding=kernel_size//2, bias=False
         -> L_out = floor((128 + 2*4 - 8)/1) + 1 = 129
       - BatchNorm1d(32) -> ReLU()
       - MaxPool1d(k=2, stride=2, padding=1)
         -> L_out = floor((129 + 2*1 - 2)/2) + 1 = 65
       - Dropout(dropout)
       -> Output Block 1: (B, 32, 65)
    3. Block 2:
       - Conv1d: in=32, out=64, k=8, stride=1, padding=4, bias=False
         -> L_out = floor((65 + 2*4 - 8)/1) + 1 = 66
       - BatchNorm1d(64) -> ReLU()
       - MaxPool1d(k=2, stride=2, padding=1)
         -> L_out = floor((66 + 2*1 - 2)/2) + 1 = 34
       -> Output Block 2: (B, 64, 34)
    4. Block 3:
       - Conv1d: in=64, out=feature_dim, k=8, stride=1, padding=4, bias=False
         -> L_out = floor((34 + 2*4 - 8)/1) + 1 = 35
       - BatchNorm1d(feature_dim) -> ReLU()
       - MaxPool1d(k=2, stride=2, padding=1)
         -> L_out = floor((35 + 2*1 - 2)/2) + 1 = 18
       -> Output Block 3: (B, 128, 18)
===============================================================================
"""

import torch
import torch.nn as nn


class TSTCCEncoder(nn.Module):
    """
    Backbone 1D-CNN trích xuất đặc trưng khớp 100% với kiến trúc `base_Model` của TS-TCC.
    Lược bỏ tầng self.logits để dùng làm Encoder độc lập cho Pretrain SSL & Transfer Learning.
    """

    def __init__(
        self,
        in_channels: int = 6,
        feature_dim: int = 128,
        kernel_size: int = 8,
        stride: int = 1,
        dropout: float = 0.35
    ):
        """
        Args:
            in_channels (int): Số kênh cảm biến đầu vào (mặc định 6).
            feature_dim (int): Số kênh đặc trưng đầu ra (mặc định 128).
            kernel_size (int): Kích thước kernel của Block 1 lấy từ config (mặc định 8).
            stride (int): Stride của Conv (mặc định 1).
            dropout (float): Tỷ lệ dropout ở Block 1 (mặc định 0.35).
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # BLOCK 1: Conv(k=kernel_size, p=kernel_size//2) -> BN -> ReLU -> MaxPool -> Dropout
        # ---------------------------------------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        # ---------------------------------------------------------------------
        # BLOCK 2: Conv(k=8, p=4) -> BN -> ReLU -> MaxPool
        # ---------------------------------------------------------------------
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        # ---------------------------------------------------------------------
        # BLOCK 3: Conv(k=8, p=4) -> BN -> ReLU -> MaxPool
        # ---------------------------------------------------------------------
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=64,
                out_channels=feature_dim,
                kernel_size=8,
                stride=1,
                bias=False,
                padding=4
            ),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor tín hiệu (B, in_channels, 128)
        Returns:
            torch.Tensor: Feature map tensor shape (B, 128, 18)
        """
        x = self.conv_block1(x)  # (B, 32, 65)
        x = self.conv_block2(x)  # (B, 64, 34)
        x = self.conv_block3(x)  # (B, 128, 18)
        return x