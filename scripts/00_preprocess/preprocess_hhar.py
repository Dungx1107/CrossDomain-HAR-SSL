"""
===============================================================================
TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: HHAR (PHONE, WATCH & COMBINED)
1. Đọc và lọc nhãn 5 lớp chung (bỏ 'bike' và 'null').
2. Nhóm theo (User, Device, gt) để bảo toàn tính thuần nhất của nhãn.
3. Chia đoạn theo gaps > 1.0s, áp bộ lọc Low-pass 10Hz và nội suy tuyến tính về 50Hz.
4. Trượt cửa sổ 128 mẫu (stride 64) với MIN_SAMPLES = 256.
5. Đóng gói riêng hhar_phone, hhar_watch và bản gộp hhar_combined cho SSL.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.hhar_config import HHARConfig

RAW_DATA_DIR = HHARConfig.RAW_DATA_DIR
LABEL_MAP = HHARConfig.LABEL_MAPPING
TRAIN_SUBJECTS = HHARConfig.TRAIN_SUBJECTS
VAL_SUBJECTS = HHARConfig.VAL_SUBJECTS
TEST_SUBJECTS = HHARConfig.TEST_SUBJECTS
WINDOW_SIZE = HHARConfig.WINDOW_SIZE
STRIDE = HHARConfig.STRIDE
TARGET_FS = HHARConfig.TARGET_FS
GAP_THRESHOLD_S = HHARConfig.GAP_THRESHOLD_S
LOWPASS_CUTOFF = HHARConfig.LOWPASS_CUTOFF
FILTER_ORDER = HHARConfig.FILTER_ORDER
MIN_SAMPLES = HHARConfig.MIN_SAMPLES
IN_CHANNELS = HHARConfig.IN_CHANNELS

USECOLS = ["Creation_Time", "x", "y", "z", "User", "Device", "gt"]


def split_at_gaps(times_s: np.ndarray, gap_threshold_s: float):
    if len(times_s) == 0:
        return []
    gaps = np.flatnonzero(np.diff(times_s) > gap_threshold_s) + 1
    starts = np.r_[0, gaps]
    ends = np.r_[gaps, len(times_s)]
    return [slice(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def effective_fs(times_s: np.ndarray) -> float:
    duration = float(times_s[-1] - times_s[0])
    if len(times_s) < 2 or duration <= 0:
        return 0.0
    return float((len(times_s) - 1) / duration)


def lowpass_if_needed(values: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    if fs <= 20.0 or len(values) <= (order * 3 + 1):
        return values
    nyq = 0.5 * fs
    cutoff_eff = min(cutoff, np.nextafter(nyq, 0.0))
    if cutoff_eff <= 0:
        return values
    b, a = butter(order, cutoff_eff / nyq, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(values) <= padlen:
        return values
    return filtfilt(b, a, values, axis=0)


def resample_recording(df: pd.DataFrame, target_fs: float) -> np.ndarray:
    df = df.sort_values("Creation_Time").drop_duplicates("Creation_Time")
    if len(df) < 2:
        return np.empty((0, 3), dtype=np.float32)

    times_s = df["Creation_Time"].to_numpy(dtype=np.float64) / 1e9
    values = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    signal_segments = []
    for segment in split_at_gaps(times_s, GAP_THRESHOLD_S):
        seg_t = times_s[segment]
        seg_values = values[segment]
        if len(seg_t) < 2:
            continue

        seg_t = seg_t - seg_t[0]
        duration = float(seg_t[-1] - seg_t[0])
        if duration <= 0:
            continue

        target_t = np.arange(0.0, duration, 1.0 / target_fs, dtype=np.float64)
        if len(target_t) == 0:
            continue

        fs = effective_fs(seg_t)
        filtered = lowpass_if_needed(seg_values, fs=fs, cutoff=LOWPASS_CUTOFF, order=FILTER_ORDER)
        resampled = np.column_stack([
            np.interp(target_t, seg_t, filtered[:, axis_idx])
            for axis_idx in range(filtered.shape[1])
        ])
        signal_segments.append(resampled.astype(np.float32, copy=False))

    if not signal_segments:
        return np.empty((0, 3), dtype=np.float32)

    return np.vstack(signal_segments)


def process_sensor_pair(acc_file: str, gyro_file: str) -> tuple:
    acc_path = os.path.join(RAW_DATA_DIR, acc_file)
    gyro_path = os.path.join(RAW_DATA_DIR, gyro_file)

    if not os.path.exists(acc_path) or not os.path.exists(gyro_path):
        print(f"⚠️ Không tìm thấy file: {acc_path} hoặc {gyro_path}")
        return [], [], []

    print(f"\n⏳ Đang xử lý: {acc_file} + {gyro_file}")
    acc_df = pd.read_csv(acc_path, usecols=USECOLS)
    gyro_df = pd.read_csv(gyro_path, usecols=USECOLS)

    valid_labels = set(LABEL_MAP.keys())
    acc_df = acc_df[acc_df["gt"].isin(valid_labels)]
    gyro_df = gyro_df[gyro_df["gt"].isin(valid_labels)]

    acc_groups = acc_df.groupby(["User", "Device", "gt"])
    gyro_groups = gyro_df.groupby(["User", "Device", "gt"])

    common_keys = sorted(set(acc_groups.groups.keys()) & set(gyro_groups.groups.keys()))
    print(f"   -> Gom được {len(common_keys)} phiên (User, Device, gt) hợp lệ")

    all_windows, all_labels, all_subjects = [], [], []

    for user, device, gt in common_keys:
        acc_group = acc_groups.get_group((user, device, gt))
        gyro_group = gyro_groups.get_group((user, device, gt))

        overlap_start = max(acc_group["Creation_Time"].min(), gyro_group["Creation_Time"].min())
        overlap_end = min(acc_group["Creation_Time"].max(), gyro_group["Creation_Time"].max())
        if overlap_end <= overlap_start:
            continue

        acc_group = acc_group[(acc_group["Creation_Time"] >= overlap_start) & (acc_group["Creation_Time"] <= overlap_end)]
        gyro_group = gyro_group[(gyro_group["Creation_Time"] >= overlap_start) & (gyro_group["Creation_Time"] <= overlap_end)]

        acc_signal = resample_recording(acc_group, TARGET_FS)
        gyro_signal = resample_recording(gyro_group, TARGET_FS)

        # Lọc theo MIN_SAMPLES = 256
        if len(acc_signal) < MIN_SAMPLES or len(gyro_signal) < MIN_SAMPLES:
            continue

        min_len = min(len(acc_signal), len(gyro_signal))
        acc_signal = acc_signal[:min_len]
        gyro_signal = gyro_signal[:min_len]

        sensor_data = np.concatenate([acc_signal, gyro_signal], axis=1)  # (T, 6)
        label_id = LABEL_MAP[gt]

        for start in range(0, min_len - WINDOW_SIZE + 1, STRIDE):
            end = start + WINDOW_SIZE
            window = sensor_data[start:end].T  # Shape: (6, 128)
            all_windows.append(window)
            all_labels.append(label_id)
            all_subjects.append(ord(user) - ord('a'))

    return all_windows, all_labels, all_subjects


def build_subset(X_all: torch.Tensor, y_all: torch.Tensor, subs_all: np.ndarray, subject_list: list) -> dict:
    mask = np.isin(subs_all, subject_list)
    n = int(mask.sum())

    if n == 0:
        return {
            "samples": torch.empty((0, IN_CHANNELS, WINDOW_SIZE), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.long),
            "subjects": torch.empty((0,), dtype=torch.long),
        }

    return {
        "samples": X_all[mask],
        "labels": y_all[mask],
        "subjects": torch.tensor(subs_all[mask], dtype=torch.long),
    }


def process_device_type(acc_name: str, gyro_name: str, output_dir: Path, device_name: str) -> tuple:
    print("\n" + "=" * 80)
    print(f"🚀 XỬ LÝ DỮ LIỆU: {device_name.upper()}")
    print(f"📁 Thư mục lưu: {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    windows, labels, subjects = process_sensor_pair(acc_name, gyro_name)

    if not windows:
        print(f"⚠️ Không trích xuất được dữ liệu cho {device_name}")
        return [], [], []

    X_all = torch.tensor(np.array(windows), dtype=torch.float32)  # (N, 6, 128)
    y_all = torch.tensor(np.array(labels), dtype=torch.long)      # (N,)
    subs_all = np.array(subjects, dtype=np.int64)

    train_data = build_subset(X_all, y_all, subs_all, TRAIN_SUBJECTS)
    val_data = build_subset(X_all, y_all, subs_all, VAL_SUBJECTS)
    test_data = build_subset(X_all, y_all, subs_all, TEST_SUBJECTS)

    all_data = {
        "samples": X_all,
        "labels": y_all,
        "subjects": torch.tensor(subs_all, dtype=torch.long),
    }

    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data, output_dir / "val.pt")
    torch.save(test_data, output_dir / "test.pt")
    torch.save(all_data, output_dir / "dataset_all.pt")

    print(f"\n📊 KẾT QUẢ ĐÓNG GÓI CHO {device_name.upper()}:")
    print(f"   -> Train : {train_data['samples'].shape} | Nhãn: {torch.bincount(train_data['labels']).tolist()}")
    print(f"   -> Val   : {val_data['samples'].shape} | Nhãn: {torch.bincount(val_data['labels']).tolist()}")
    print(f"   -> Test  : {test_data['samples'].shape} | Nhãn: {torch.bincount(test_data['labels']).tolist()}")
    print(f"   -> All   : {all_data['samples'].shape}")

    return windows, labels, subjects


def main():
    phone_windows, phone_labels, phone_subjects = process_device_type(
        acc_name="Phones_accelerometer.csv",
        gyro_name="Phones_gyroscope.csv",
        output_dir=HHARConfig.PROCESSED_DIR_PHONE,
        device_name="HHAR_PHONE"
    )

    watch_windows, watch_labels, watch_subjects = process_device_type(
        acc_name="Watch_accelerometer.csv",
        gyro_name="Watch_gyroscope.csv",
        output_dir=HHARConfig.PROCESSED_DIR_WATCH,
        device_name="HHAR_WATCH"
    )

    # Đóng gói bộ gộp HHAR Combined cho SSL Pretraining
    if phone_windows and watch_windows:
        combined_dir = HHARConfig.PROCESSED_DIR_COMBINED
        os.makedirs(combined_dir, exist_ok=True)

        X_combined = torch.cat([
            torch.tensor(np.array(phone_windows), dtype=torch.float32),
            torch.tensor(np.array(watch_windows), dtype=torch.float32),
        ], dim=0)

        y_combined = torch.cat([
            torch.tensor(np.array(phone_labels), dtype=torch.long),
            torch.tensor(np.array(watch_labels), dtype=torch.long),
        ], dim=0)

        subs_combined = np.concatenate([
            np.array(phone_subjects, dtype=np.int64),
            np.array(watch_subjects, dtype=np.int64),
        ])

        torch.save({
            "samples": X_combined,
            "labels": y_combined,
            "subjects": torch.tensor(subs_combined, dtype=torch.long),
        }, combined_dir / "dataset_all.pt")

        print("\n" + "=" * 80)
        print(f"🌐 ĐÃ TẠO TẬP PRETRAIN KẾT HỢP (COMBINED): {combined_dir / 'dataset_all.pt'}")
        print(f"   -> Tổng số mẫu: {X_combined.shape} (N, 6, 128)")
        print(f"   -> Phân phối nhãn: {torch.bincount(y_combined).tolist()}")
        print("=" * 80)

    print("\n🎉 HOÀN THÀNH TIỀN XỬ LÝ TOÀN BỘ HHAR!")


if __name__ == "__main__":
    main()