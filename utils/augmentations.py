"""
MỤC ĐÍCH: Module cung cấp các thuật toán biến đổi tín hiệu chuỗi thời gian (Time-Series
    Data Augmentation) và bọc dữ liệu (Dataset Wrapper) phục vụ cho quá trình
    huấn luyện Tự giám sát (Self-Supervised Learning - SSL) theo chuẩn TS-TCC.
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline


def jitter(x, sigma=0.05):
    """Thêm nhiễu trắng Gaussian vào tín hiệu."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise


def scaling(x, sigma=0.1):
    """Nhân toàn bộ kênh tín hiệu với hệ số co giãn biên độ ngẫu nhiên."""
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
    return x * factor


def time_warp(x, sigma=0.2, num_knots=4):
    """Uốn cong trục thời gian cục bộ bằng Cubic Spline."""
    C, T = x.shape
    time_orig = np.arange(T)
    knot_positions = np.linspace(0, T - 1, num=num_knots + 2)
    random_shifts = np.random.normal(loc=1.0, scale=sigma, size=(num_knots + 2,))
    warped_knot_positions = knot_positions * random_shifts

    warped_knot_positions = (warped_knot_positions - warped_knot_positions[0]) / (
        warped_knot_positions[-1] - warped_knot_positions[0]
    ) * (T - 1)

    spline = CubicSpline(knot_positions, warped_knot_positions)
    warped_time = np.clip(spline(time_orig), 0, T - 1)

    x_warped = np.zeros_like(x)
    for c in range(C):
        x_warped[c] = np.interp(time_orig, warped_time, x[c])
    return x_warped


def permutation(x, max_segments=4):
    """Chia tín hiệu thành N đoạn con và xáo trộn ngẫu nhiên."""
    C, T = x.shape
    seg_len = T // max_segments
    segments = [
        x[:, i * seg_len : (i + 1) * seg_len if i < max_segments - 1 else T]
        for i in range(max_segments)
    ]
    np.random.shuffle(segments)
    return np.concatenate(segments, axis=1)

# =============================================================================
# BỘ PHỐI HỢP WEAK & STRONG AUGMENTATION
# =============================================================================
class TS_TCC_Augmentation:
    """
    Bộ tạo 2 views (Weak view và Strong view) theo chuẩn TS-TCC.
    """
    def __init__(self, jitter_sigma=0.05, scale_sigma=0.1, warp_sigma=0.2, n_perm=4):
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.warp_sigma = warp_sigma
        self.n_perm = n_perm

    def weak_transform(self, x):
        """View yếu: Kết hợp Scaling và Jittering nhẹ."""
        x_aug = scaling(x, sigma=self.scale_sigma)
        x_aug = jitter(x_aug, sigma=self.jitter_sigma)
        return x_aug

    def strong_transform(self, x):
        """View mạnh: Kết hợp Permutation và Time-Warping."""
        x_aug = permutation(x, max_segments=self.n_perm)
        x_aug = time_warp(x_aug, sigma=self.warp_sigma)
        return x_aug

    def __call__(self, x):
        """
        Nhận vào mảng numpy (C, T) và trả về 2 views dạng PyTorch Tensor.
        """
        x_w = self.weak_transform(x.copy())
        x_s = self.strong_transform(x.copy())

        return (
            torch.tensor(x_w, dtype=torch.float32),
            torch.tensor(x_s, dtype=torch.float32)
        )