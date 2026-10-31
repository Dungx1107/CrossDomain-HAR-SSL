"""
===============================================================================
MODULE: DATA I/O (ĐỌC FILE .pt ĐÃ TIỀN XỬ LÝ)
===============================================================================
Đọc các split (train/val/test) dạng .pt có khóa "samples" và "labels",
lọc về các lớp chung và chuẩn hóa shape về (N, C, T).
===============================================================================
"""

from pathlib import Path
from typing import Tuple, Union

import torch


def load_data(path: Union[str, Path], num_classes: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        path: Đường dẫn file .pt (dict gồm "samples" và "labels").
        num_classes: Chỉ giữ các mẫu có nhãn trong [0, num_classes).
    Returns:
        samples (float32, shape (N, C, T)), labels (long, shape (N,)).
    """
    d = torch.load(path, map_location="cpu", weights_only=True)

    labels = d["labels"].reshape(-1)
    mask = (labels >= 0) & (labels < num_classes)
    samples, labels = d["samples"][mask].float(), labels[mask].long()

    # (N, 128, 6) -> (N, 6, 128): đưa kênh cảm biến lên trước thời gian
    if samples.ndim == 3 and samples.shape[1] == 128 and samples.shape[2] == 6:
        samples = samples.permute(0, 2, 1).contiguous()

    return samples, labels