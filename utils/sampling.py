"""
===============================================================================
MODULE FEW-LABEL SAMPLING (LẤY MẪU PHÂN TẦNG VỚI 5 RANDOM SEEDS)
===============================================================================
Mục đích:
    - Trích xuất tập con có nhãn (1%, 5%, 10%) từ tập huấn luyện.
    - Đảm bảo tỷ lệ các lớp không bị lệch (Stratified Sampling).
    - Cố định danh sách seed để kết quả có thể tái lập (Reproducible).
===============================================================================
"""

import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

# Danh sách 5 Random Seeds chuẩn cho toàn bộ nghiên cứu
SEEDS = [42, 1337, 2024, 7, 99]
RATIO = 0.1  # 0.05, 0.01


def get_stratified_few_label_indices(labels, ratio=RATIO, seed=SEEDS[0]):
    """
    Trích xuất chỉ số (indices) của một tỷ lệ dữ liệu nhất định bảo toàn phân phối nhãn.
    Args:
        labels (np.ndarray hoặc list): Mảng chứa nhãn số nguyên của toàn bộ tập Train.
        ratio (float): Tỷ lệ lấy mẫu (ví dụ: 0.01, 0.05, 0.10).
        seed (int): Random seed.
    Returns:
        np.ndarray: Mảng chứa các index được chọn.
    """
    labels = np.array(labels)
    n_samples = len(labels)
    n_select = max(int(n_samples * ratio), len(np.unique(labels)))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_select, random_state=seed)

    # Tạo mảng giả X để phân tách theo y (labels)
    dummy_X = np.zeros(n_samples)
    selected_indices, _ = next(sss.split(dummy_X, labels))

    return selected_indices


def create_few_label_subset(dataset, ratio=RATIO, seed=SEEDS[0]):
    """
    Bọc PyTorch Dataset thành một Subset chỉ chứa tỷ lệ phần trăm dữ liệu đã chọn.
    """
    # Trích xuất toàn bộ nhãn từ Dataset
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
        # cú pháp gọi hàm __getitem__(i) được định nghĩa trong class Dataset.

    selected_idx = get_stratified_few_label_indices(labels, ratio=ratio, seed=seed)
    return Subset(dataset, selected_idx), selected_idx


# hàm lấy mẫu phân tầng trên uci
# Hàm StratifiedShuffleSplit thuộc thư viện scikit-learn. Thuật toán này không nhận PyTorch Tensor, nó bắt buộc đầu vào phải là mảng NumPy (np.ndarray) thì mới tính toán chia phân tầng được.
def sample_subset_by_ratio(
        x_all,
        y_all,
        fraction,
        seed=42):
    if fraction >= 1:
        return x_all.clone(), y_all.clone()
    num_classes = len(torch.unique(y_all))
    n_samples = max(num_classes, int(len(x_all) * fraction))

    # xử lí nếu đầu vào là tensor (phải chuyển sang numpy vì hàm lấy mẫu phân tầng chỉ hiểu numpy)
    y_np = y_all.detach().cpu().numpy() if isinstance(y_all, torch.Tensor) else y_all
    x_np = x_all.detach().cpu().numpy() if isinstance(x_all, torch.Tensor) else x_all

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_samples,
        random_state=seed
    )
    train_idx, _ = next(sss.split(x_np, y_np))
    return x_all[train_idx], y_all[train_idx]
