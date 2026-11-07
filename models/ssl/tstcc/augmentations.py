"""
===============================================================================
MODULE: TIME-SERIES DATA AUGMENTATION CHO TS-TCC
===============================================================================
Mã nguồn đối chiếu trực tiếp từ file `augmentations.py` của repository chính thức:
    emadeldeen24/TS-TCC (IJCAI 2021)
===============================================================================
"""

import numpy as np
import torch


def jitter(x: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    Thêm nhiễu trắng Gaussian vào tín hiệu theo chuẩn TS-TCC gốc (sigma=0.8).
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise


def scaling(x: np.ndarray, sigma: float = 1.1) -> np.ndarray:
    """
    Co giãn biên độ tín hiệu với loc=2.0 và sigma=1.1 theo chuẩn TS-TCC gốc.
    Nhận tensor (C, T) hoặc (B, C, T).
    """
    is_2d = (x.ndim == 2)
    if is_2d:
        x = x[np.newaxis, ...]  # (1, C, T)

    B, C, T = x.shape
    factor = np.random.normal(loc=2.0, scale=sigma, size=(B, T))

    # factor shape (B, 1, T) nhân dọc theo channels C
    scaled_x = x * factor[:, np.newaxis, :]

    if is_2d:
        scaled_x = scaled_x.squeeze(0)

    return scaled_x

def permutation(x: np.ndarray, max_segments: int = 5, seg_mode: str = "random") -> np.ndarray:
    """
    Hoán vị trật tự các đoạn thời gian.
    Áp dụng đồng bộ cho TẤT CẢ các kênh cảm biến.
    Giả định x có layout (C, T) hoặc (B, C, T).
    """
    is_2d = (x.ndim == 2)
    if is_2d:
        x = x[np.newaxis, ...]  # (1, C, T)

    B, C, T = x.shape
    orig_steps = np.arange(T)
    num_segs = np.random.randint(1, max_segments + 1, size=B)
    ret = np.zeros_like(x)

    for i in range(B):
        if num_segs[i] > 1:
            if seg_mode == "random":
                n_splits = min(num_segs[i] - 1, T - 2)
                if n_splits > 0:
                    split_points = np.random.choice(T - 2, n_splits, replace=False)
                    split_points.sort()
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = [orig_steps]
            else:
                splits = np.array_split(orig_steps, num_segs[i])

            # Xáo trộn thứ tự các đoạn
            perm_order = np.random.permutation(len(splits))
            shuffled_splits = [splits[k] for k in perm_order]
            warp = np.concatenate(shuffled_splits).ravel()

            # Dùng np.take để tránh advanced-indexing đảo trục
            ret[i] = np.take(x[i], warp, axis=-1)
        else:
            ret[i] = x[i]

    if is_2d:
        ret = ret.squeeze(0)

    return ret

class TS_TCC_Augmentation:
    def __init__(
        self,
        jitter_scale_ratio: float = 1.1,
        jitter_ratio: float = 0.8,
        max_seg: int = 5
    ):
        self.jitter_scale_ratio = jitter_scale_ratio
        self.jitter_ratio = jitter_ratio
        self.max_seg = max_seg

    def weak_transform(self, x: np.ndarray) -> np.ndarray:
        return scaling(x, sigma=self.jitter_scale_ratio)

    def strong_transform(self, x: np.ndarray) -> np.ndarray:
        x_perm = permutation(x, max_segments=self.max_seg)
        x_strong = jitter(x_perm, sigma=self.jitter_ratio)
        return x_strong

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x_numpy = x.detach().cpu().numpy()
        else:
            x_numpy = np.array(x)

        # Đảm bảo nếu đưa vào (T, C) = (128, 6) thì đảo thành (6, 128)
        if x_numpy.ndim == 2 and x_numpy.shape[0] == 128 and x_numpy.shape[1] == 6:
            x_numpy = x_numpy.T

        x_w = self.weak_transform(x_numpy.copy())
        x_s = self.strong_transform(x_numpy.copy())

        return (
            torch.from_numpy(x_w).float(),
            torch.from_numpy(x_s).float()
        )