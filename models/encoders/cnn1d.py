"""
===============================================================================
MODULE ENCODER 1D-CNN (STANDARD BACKBONE) CHO CẢM BIẾN HAR
===============================================================================
Mục đích:
    Trích xuất véc-tơ đặc trưng (Feature Vector) 128 chiều từ tín hiệu cảm biến thô.
    Nhận vào Tensor shape: (Batch_Size, Channels, Window_Length) -> vd: (64, 6, 128)
    Trả về Tensor shape:   (Batch_Size, Feature_Dim)           -> vd: (64, 128)
===============================================================================
"""

import torch
import torch.nn as nn


class StandardSensorEncoder1D(nn.Module):
    """
    Khung xương Encoder 1D-CNN chuẩn hóa gồm 4 khối Tích chập (Convolution Blocks)
    """

    def __init__(self, in_channels: int = 6, feature_dim: int = 128):
        """
        Khởi tạo các tầng mạng

        Tham số:
            in_channels (int): Số kênh cảm biến đầu vào (default: 6 cho MotionSense)
            feature_dim (int): Kích thước Vector đặc trưng đầu ra (default: 128)
        """
        super().__init__()
        self.feature_dim = feature_dim

        # ---------------------------------------------------------------------
        # KHỐI 1: Trích xuất đặc trưng tần số thấp (Low-level local patterns)
        # Input: (B, in_channels, 128) -> Output: (B, 32, 64)
        # ---------------------------------------------------------------------
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Giảm độ dài chuỗi 128 -> 64
        )

        # ---------------------------------------------------------------------
        # KHỐI 2: Trích xuất đặc trưng cấp trung
        # Input: (B, 32, 64) -> Output: (B, 64, 32)
        # ---------------------------------------------------------------------
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Giảm độ dài chuỗi 64 -> 32
        )

        # ---------------------------------------------------------------------
        # KHỐI 3: Trích xuất đặc trưng cấp cao
        # Input: (B, 64, 32) -> Output: (B, 128, 16)
        # ---------------------------------------------------------------------
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Giảm độ dài chuỗi 32 -> 16
        )

        # ---------------------------------------------------------------------
        # KHỐI 4: Trích xuất đặc trưng trừu tượng toàn cục
        # Input: (B, 128, 16) -> Output: (B, 256, 8)
        # ---------------------------------------------------------------------
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Giảm độ dài chuỗi 16 -> 8
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng tính toán lan truyền tiến (Forward Pass)
        """
        # x shape ban đầu: (B, in_channels, 128)
        x = self.block1(x)  # -> (B, 32, 64)
        x = self.block2(x)  # -> (B, 64, 32)
        x = self.block3(x)  # -> (B, 96, 16)
        x = self.block4(x)  # -> (B, 128, 8)
        return x

