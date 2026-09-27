"""
===============================================================================
KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ĐÃ QUA TIỀN XỬ LÝ (UCI-HAR, MOTIONSENSE, HHAR)
===============================================================================
Mục đích:
1. Kiểm tra cấu trúc (shape), nhãn, subject, nan/inf của từng bộ dữ liệu.
2. Kiểm tra độ lớn gia tốc tĩnh (Static Gravity Check) để đảm bảo có trọng lực.
3. So sánh tương quan thống kê giữa Phone và Watch.

Cách dùng:
    python utils/check_data_quality.py
===============================================================================
"""

import os
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.hhar_config import HHARConfig
from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig

# Danh mục các dataset cần kiểm tra tự động
DATASETS = {
    "UCI-HAR": {
        "path": UCIHARConfig.DATA_ALL_PATH,
        "class_names": UCIHARConfig.CLASS_NAMES,
        "expected_subjects": 30,
        "static_labels": [3, 4],  # Sitting, Standing
    },
    "MotionSense": {
        "path": MotionSenseConfig.DATA_ALL_PATH,
        "class_names": MotionSenseConfig.CLASS_NAMES,
        "expected_subjects": 24,
        "static_labels": [3, 4],  # Sitting, Standing
    },
    "HHAR-Phone": {
        "path": HHARConfig.PROCESSED_DIR_PHONE / "dataset_all.pt",
        "class_names": HHARConfig.CLASS_NAMES,
        "expected_subjects": 9,
        "static_labels": [3, 4],  # Sitting, Standing
    },
    "HHAR-Watch": {
        "path": HHARConfig.PROCESSED_DIR_WATCH / "dataset_all.pt",
        "class_names": HHARConfig.CLASS_NAMES,
        "expected_subjects": 9,
        "static_labels": [3, 4],  # Sitting, Standing
    },
    "HHAR-Combined": {
        "path": HHARConfig.PROCESSED_DIR_COMBINED / "dataset_all.pt",
        "class_names": HHARConfig.CLASS_NAMES,
        "expected_subjects": 9,
        "static_labels": [3, 4],  # Sitting, Standing
    },
}


def check_shape(data, name):
    print(f"\n📐 SHAPE CHECK: {name}")
    print("-" * 55)

    samples = data["samples"]
    labels = data["labels"]
    subjects = data["subjects"]

    n_samples, n_channels, n_timesteps = samples.shape

    print(f"  ✅ Số mẫu        : {n_samples:,}")
    print(f"  ✅ Số kênh       : {n_channels}")
    print(f"  ✅ Số timesteps  : {n_timesteps}")
    print(f"  ✅ Tensor shape  : {samples.shape}")

    errors = []
    if n_channels != 6:
        errors.append(f"❌ Số kênh phải là 6, hiện tại là {n_channels}")
    if n_timesteps != 128:
        errors.append(f"❌ Số timesteps phải là 128, hiện tại là {n_timesteps}")
    if labels.shape[0] != n_samples:
        errors.append(f"❌ Số labels ({labels.shape[0]}) != samples ({n_samples})")
    if subjects.shape[0] != n_samples:
        errors.append(f"❌ Số subjects ({subjects.shape[0]}) != samples ({n_samples})")

    if errors:
        for err in errors:
            print(f"  {err}")
    else:
        print("  ✅ SHAPE OK!")

    return n_samples, n_channels, n_timesteps


def check_labels(data, name, class_names):
    print(f"\n🏷️ LABEL CHECK: {name}")
    print("-" * 55)

    labels = data["labels"].numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"  Tổng số mẫu: {total:,} | Số lớp: {len(unique_labels)}")
    for label_id in sorted(unique_labels):
        count = counts[unique_labels == label_id][0]
        pct = count / total * 100
        cls_name = (
            class_names[label_id]
            if 0 <= label_id < len(class_names)
            else f"Unknown_{label_id}"
        )
        bar = "█" * int(pct // 2)
        print(f"  {label_id:2d} [{cls_name:18s}] {count:7,} ({pct:5.1f}%) {bar}")

    if len(unique_labels) != len(class_names):
        print(f"  ⚠️ Cảnh báo: Cấu hình có {len(class_names)} lớp, dữ liệu có {len(unique_labels)} lớp")
    else:
        print(f"  ✅ Đầy đủ {len(class_names)} lớp!")

    return unique_labels, counts


def check_subjects(data, name, expected_count):
    print(f"\n👤 SUBJECT CHECK: {name}")
    print("-" * 55)

    subjects = data["subjects"].numpy()
    unique_subjects = np.unique(subjects)
    print(f"  Số subject: {len(unique_subjects)}/{expected_count}")
    print(f"  Danh sách ID: {sorted(unique_subjects)}")

    subject_counts = Counter(subjects)
    min_c = min(subject_counts.values())
    max_c = max(subject_counts.values())
    avg_c = sum(subject_counts.values()) / len(subject_counts)
    print(f"  Số mẫu/subject: min={min_c}, max={max_c}, avg={avg_c:.1f}")

    return unique_subjects


def check_nan_inf(data, name):
    print(f"\n🔍 NAN/INF CHECK: {name}")
    print("-" * 55)

    samples = data["samples"]
    n_nan = torch.isnan(samples).sum().item()
    n_inf = torch.isinf(samples).sum().item()

    if n_nan > 0 or n_inf > 0:
        print(f"  ❌ Phát hiện lỗi: {n_nan:,} giá trị NaN, {n_inf:,} giá trị Inf!")
        return False

    print("  ✅ Dữ liệu hoàn toàn sạch (0 NaN, 0 Inf)")
    return True


def check_statistics_and_gravity(data, name, static_labels):
    print(f"\n📊 STATISTICS & GRAVITY CHECK: {name}")
    print("-" * 55)

    samples = data["samples"]
    labels = data["labels"]

    stats = []
    ch_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for c in range(samples.shape[1]):
        c_data = samples[:, c, :].numpy().flatten()
        stats.append({
            "Channel": f"{c} ({ch_names[c]})",
            "Mean": np.mean(c_data),
            "Std": np.std(c_data),
            "Min": np.min(c_data),
            "Max": np.max(c_data),
        })

    print(pd.DataFrame(stats).round(4).to_string(index=False))

    # Kiểm tra độ lớn gia tốc ở trạng thái tĩnh (Sitting / Standing)
    mask = torch.isin(labels, torch.tensor(static_labels))
    if mask.sum() > 0:
        static_acc = samples[mask, 0:3, :]  # 3 kênh gia tốc
        mag = torch.sqrt((static_acc ** 2).sum(dim=1))
        mean_mag = mag.mean().item()
        print(f"\n  👉 Độ lớn gia tốc ở tư thế tĩnh (Sitting/Standing): {mean_mag:.4f}")
        if 0.8 <= mean_mag <= 1.2 or 8.0 <= mean_mag <= 11.5:
            print("  ✅ HỢP LỆ: Tín hiệu chứa thành phần trọng lực (~1.0g hoặc ~9.8 m/s²)")
        else:
            print("  ⚠️ CẢNH BÁO: Magnitude lệch xa trọng lực chuẩn. Cần kiểm tra lại đơn vị!")


def main():
    print("=" * 80)
    print("🔍 BẮT ĐẦU KIỂM TRA CHẤT LƯỢNG DỮ LIỆU TỔNG THỂ")
    print("=" * 80)

    checked = 0
    for name, cfg in DATASETS.items():
        path = Path(cfg["path"])
        if not path.exists():
            continue

        checked += 1
        print(f"\n{'=' * 80}\n📂 KIỂM TRA DATASET: {name}\n📁 File: {path}\n{'=' * 80}")
        data = torch.load(path, map_location="cpu", weights_only=True)

        check_shape(data, name)
        check_labels(data, name, cfg["class_names"])
        check_subjects(data, name, cfg["expected_subjects"])
        check_nan_inf(data, name)
        check_statistics_and_gravity(data, name, cfg["static_labels"])

    if checked == 0:
        print("❌ Không tìm thấy file dữ liệu nào. Hãy chạy script tiền xử lý trước!")
    else:
        print("\n" + "=" * 80)
        print("🎉 HOÀN THÀNH TẤT CẢ CÁC BƯỚC KIỂM TRA CHẤT LƯỢNG!")
        print("=" * 80)


if __name__ == "__main__":
    main()