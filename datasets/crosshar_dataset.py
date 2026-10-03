"""
===============================================================================
MODULE: CROSSHAR PHYSICALLY-INFORMED AUGMENTATION & DATASET WRAPPER
===============================================================================
Triển khai đúng 100% đặc tả bài báo CrossHAR:
1. Physical3DAugmentation: Ma trận hoán vị 3D (6 góc xoay trực giao) áp dụng
   đồng bộ cho cả 3 trục Acc và 3 trục Gyro.
2. ChannelInstanceNorm: Chuẩn hóa Instance Normalization trên từng kênh riêng biệt.
3. Sinh 2 view cho Contrastive Regularization:
   - View x^-: Biến đổi biên độ với nhiễu Gauss N(2, 1.1)
   - View x^+: Cắt đoạn (permutation) và thêm nhiễu ngẫu nhiên
4. CrossHARPretrainDataset: Đóng gói trả về bộ 3 tensor cho từng mẫu:
   (x_raw_normalized, x_neg, x_pos) phục vụ cả 2 nhánh MSM và Contrastive.
===============================================================================
"""

import sys
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Physical3DAugmentation:
    """
    Tăng cường dữ liệu vật lý dựa trên 6 ma trận hoán vị trực giao (3D Permutations).
    Áp dụng đồng bộ cho cụm Gia tốc kế (kênh 0,1,2) và Con quay hồi chuyển (kênh 3,4,5).
    """
    def __init__(self):
        # 6 hoán vị trục tương ứng với 6 hướng trực giao không gian 3D
        self.permutations = [
            [0, 1, 2],  # (x, y, z) - gốc
            [0, 2, 1],  # (x, z, y)
            [1, 0, 2],  # (y, x, z)
            [1, 2, 0],  # (y, z, x)
            [2, 0, 1],  # (z, x, y)
            [2, 1, 0],  # (z, y, x)
        ]

    def apply_permutation(self, x: torch.Tensor, perm_idx: int) -> torch.Tensor:
        """
        Hoán đổi trục trên tensor tín hiệu x có shape (6, L).
        """
        perm = self.permutations[perm_idx]
        x_aug = x.clone()

        # Hoán vị 3 trục gia tốc kế (kênh 0, 1, 2)
        x_aug[0:3, :] = x[perm, :]

        # Hoán vị 3 trục con quay hồi chuyển (kênh 3, 4, 5) theo cùng một thứ tự
        gyro_perm = [p + 3 for p in perm]
        x_aug[3:6, :] = x[gyro_perm, :]

        return x_aug

    def expand_dataset_6x(self, X: torch.Tensor) -> torch.Tensor:
        """
        Nhân 6 lần dung lượng dữ liệu bằng cách áp dụng toàn bộ 6 hoán vị cho mỗi mẫu.
        Input: X shape (N, 6, L)
        Output: X_expanded shape (6 * N, 6, L)
        """
        augmented_list = []
        for p_idx in range(6):
            perm = self.permutations[p_idx]
            x_perm = X.clone()
            x_perm[:, 0:3, :] = X[:, perm, :]
            gyro_perm = [p + 3 for p in perm]
            x_perm[:, 3:6, :] = X[:, gyro_perm, :]
            augmented_list.append(x_perm)

        return torch.cat(augmented_list, dim=0)


def channel_instance_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Chuẩn hóa Instance Normalization trên từng kênh (Channel-wise).
    x: tensor có shape (C, L) hoặc (B, C, L)
    """
    if x.dim() == 2:
        # Shape: (C, L)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + eps)
    elif x.dim() == 3:
        # Shape: (B, C, L)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + eps)
    else:
        raise ValueError(f"Tensor phải có 2 hoặc 3 chiều, nhận được: {x.dim()}")


class ContrastiveViewGenerator:
    """
    Sinh cặp view tích cực (Positive Pair) cho Nhánh 2 (Contrastive Regularization):
    - View 1 (x^-): Scale biên độ với phân phối Gauss N(2, 1.1)
    - View 2 (x^+): Cắt chia chuỗi thành các phân đoạn, xáo trộn (permutation) + Gaussian jitter
    """
    def __init__(self, num_segments: int = 4):
        self.num_segments = num_segments

    def generate_views(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: x shape (C=6, L=128)
        Output: (x_neg, x_pos) đều có shape (C, L)
        """
        c, l = x.shape

        # 1. View 1 (x^-): Biến đổi biên độ với nhiễu Gauss N(loc=2.0, scale=1.1)
        scale_factor = torch.randn(1).item() * 1.1 + 2.0
        x_neg = x * scale_factor

        # 2. View 2 (x^+): Permute segments + Thêm nhiễu ngẫu nhiên
        seg_len = l // self.num_segments
        # Cắt thành các khối và xáo trộn thứ tự các khối
        segments = [x[:, i * seg_len:(i + 1) * seg_len] for i in range(self.num_segments)]
        perm_indices = torch.randperm(self.num_segments).tolist()
        x_shuffled = torch.cat([segments[idx] for idx in perm_indices], dim=1)

        # Thêm nhiễu nhỏ
        jitter = torch.randn_like(x_shuffled) * 0.05
        x_pos = x_shuffled + jitter

        # Chuẩn hóa lại cả 2 view qua Instance Norm để giữ thang đo ổn định
        x_neg = channel_instance_norm(x_neg)
        x_pos = channel_instance_norm(x_pos)

        return x_neg, x_pos


class CrossHARPretrainDataset(Dataset):
    """
    PyTorch Dataset cho bước Pretrain phân cấp của CrossHAR.
    Tự động chuẩn hóa Instance Norm và sinh view tương phản theo từng mẫu.
    """
    def __init__(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.float()

        if X.ndim == 3 and X.shape[1] == 128 and X.shape[2] == 6:
            X = X.permute(0, 2, 1)

        self.samples = X
        self.view_gen = ContrastiveViewGenerator()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_raw: Tín hiệu gốc đã chuẩn hóa qua Instance Norm (đưa vào Nhánh 1: MSM)
            x_neg: View 1 (đưa vào Nhánh 2: Contrastive)
            x_pos: View 2 (đưa vào Nhánh 2: Contrastive)
        """
        x_raw = channel_instance_norm(self.samples[idx])
        x_neg, x_pos = self.view_gen.generate_views(x_raw)
        return x_raw, x_neg, x_pos