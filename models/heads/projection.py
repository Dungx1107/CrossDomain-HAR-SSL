"""
===============================================================================
MỤC ĐÍCH:
    Tầng chiếu phi tuyến (Non-linear Projection Head) dùng trong học tự giám sát (SSL).
    Nhiệm vụ: Biến đổi vector đặc trưng 128 chiều từ Encoder sang không gian đối chiếu (thường là 64 chiều)
    và chuẩn hóa L2 trước khi tính hàm mất mát Contrastive Loss.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        projection_dim: int = 64,
        hidden_dim: int = None
    ):
        """
        Args:
            feature_dim (int): Chiều đầu ra của Encoder (mặc định: 128).
            projection_dim (int): Chiều vector sau khi chiếu để tính Loss (mặc định: 64).
            hidden_dim (int, optional): Chiều của lớp ẩn trung gian (mặc định bằng feature_dim).
        """
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else feature_dim

        # MLP 2 lớp với hàm kích hoạt phi tuyến ReLU
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Vector đặc trưng shape (Batch_Size, feature_dim).
            normalize (bool): Có thực hiện L2 Normalization hay không (chuẩn trong SimCLR/TS-TCC).
        Returns:
            torch.Tensor: Vector chiếu shape (Batch_Size, projection_dim).
        """
        z = self.net(x)
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z