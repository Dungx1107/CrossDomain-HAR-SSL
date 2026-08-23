"""
MỤC ĐÍCH:
    - Bọc dữ liệu chuỗi thời gian thô (Unlabeled Time-Series) thành PyTorch Dataset.
    - Cung cấp luồng dữ liệu (Data Pipeline) cho quá trình Self-Supervised Pre-training (TS-TCC).
    - Tự động chuẩn hóa chiều không gian Tensor về chuẩn Conv1D: (N, Channels, Time_Steps).
    - Cấp phát đồng thời 2 views biến đổi (Weak View và Strong View) cho mỗi mẫu dữ liệu.
"""

import numpy as np
from torch.utils.data import Dataset
from utils.augmentations import TS_TCC_Augmentation


class SSLTimeSeriesDataset(Dataset):
    """
    Dataset bọc dữ liệu chuỗi thời gian phục vụ tiền huấn luyện Tự giám sát (Self-Supervised Learning).
    """

    def __init__(self, data, augmentor=None):
        """
        Khởi tạo và tiền xử lý chiều ma trận.
        """
        super(SSLTimeSeriesDataset, self).__init__()

        # Kiểm tra nếu shape là (N, T, C) với T > C (vd: 128 > 6) thì hoán vị trục sang (N, C, T)
        if data.shape[1] > data.shape[2]:
            data = np.transpose(data, (0, 2, 1))

        self.data = data
        # Gán bộ biến đổi Weak/Strong Augmentation
        self.augmentor = augmentor if augmentor is not None else TS_TCC_Augmentation()

    def __len__(self):
        """Trả về tổng số mẫu cửa sổ nạp vào."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Truy xuất mẫu tại idx và trả về bộ đôi (x_weak, x_strong) dạng PyTorch Tensor (C, T).
        """
        x = self.data[idx]
        x_weak, x_strong = self.augmentor(x)
        return x_weak, x_strong
