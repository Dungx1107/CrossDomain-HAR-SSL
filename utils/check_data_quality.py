"""
===============================================================================
KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ĐÃ QUA TIỀN XỬ LÝ
===============================================================================
Mục đích:
1. Kiểm tra cấu trúc, phân phối, và tính hợp lệ của dữ liệu
2. Phát hiện lỗi tiền xử lý (missing values, sai shape, label imbalance,...)
3. So sánh thống kê giữa 2 datasets (UCI-HAR vs MotionSense)

Cách dùng:
    python utils/check_data_quality.py
===============================================================================
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# Thêm project root vào path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== CẤU HÌNH =====
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATASETS = {
    "UCI-HAR": os.path.join(DATA_DIR, "uci_har", "dataset_all.pt"),
    "MotionSense": os.path.join(DATA_DIR, "motionsense", "dataset_all.pt")
}

# ===== LABEL NAMES =====
UCI_LABELS = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

MOTION_LABELS = {
    0: "Downstairs",
    1: "Upstairs",
    2: "Walking",
    3: "Sitting",
    4: "Standing",
    5: "Jogging"
}


# ============================================================
# HÀM KIỂM TRA CỐT LÕI
# ============================================================

def check_shape(data, name):
    """Kiểm tra cấu trúc dữ liệu"""
    print(f"\n📐 SHAPE CHECK: {name}")
    print("-" * 50)

    samples = data['samples']
    labels = data['labels']
    subjects = data['subjects']

    n_samples, n_channels, n_timesteps = samples.shape

    print(f"  ✅ Số mẫu        : {n_samples:,}")
    print(f"  ✅ Số kênh       : {n_channels}")
    print(f"  ✅ Số timesteps  : {n_timesteps}")
    print(f"  ✅ Expected shape: (N, 6, 128) → {samples.shape}")
    print(f"  ✅ Labels shape  : {labels.shape}")
    print(f"  ✅ Subjects shape: {subjects.shape}")

    # Kiểm tra lỗi shape
    errors = []
    if n_channels != 6:
        errors.append(f"❌ Số kênh phải là 6, hiện tại là {n_channels}")
    if n_timesteps != 128:
        errors.append(f"❌ Số timesteps phải là 128, hiện tại là {n_timesteps}")
    if labels.shape[0] != n_samples:
        errors.append(f"❌ Số labels ({labels.shape[0]}) != số samples ({n_samples})")
    if subjects.shape[0] != n_samples:
        errors.append(f"❌ Số subjects ({subjects.shape[0]}) != số samples ({n_samples})")

    if errors:
        for err in errors:
            print(f"  {err}")
    else:
        print("  ✅ SHAPE OK!")

    return n_samples, n_channels, n_timesteps


# ============================================================

def check_labels(data, name, label_names):
    """Kiểm tra phân phối nhãn"""
    print(f"\n🏷️ LABEL CHECK: {name}")
    print("-" * 50)

    labels = data['labels'].numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)

    total = len(labels)
    print(f"  Tổng số mẫu: {total:,}")
    print(f"  Số lớp: {len(unique_labels)}")
    print()

    # Hiển thị phân phối từng lớp
    for label_id in sorted(unique_labels):
        count = counts[unique_labels == label_id][0]
        pct = count / total * 100
        name = label_names.get(int(label_id), f"Unknown_{label_id}")
        bar = "█" * int(pct // 2)
        print(f"  {label_id:2d} [{name:15s}] {count:7,} ({pct:5.1f}%) {bar}")

    # Kiểm tra số lớp
    if len(unique_labels) != 6:
        print(f"\n  ⚠️ WARNING: Kỳ vọng 6 lớp, tìm thấy {len(unique_labels)} lớp")
    else:
        print(f"\n  ✅ Có đủ 6 lớp!")

    # Kiểm tra cân bằng
    min_pct, max_pct = counts.min() / total * 100, counts.max() / total * 100
    balance_ratio = min_pct / max_pct

    if balance_ratio < 0.6:
        print(f"  ⚠️ Dữ liệu không cân bằng: min={min_pct:.1f}%, max={max_pct:.1f}%")
        print(f"     (Tỷ lệ: {balance_ratio:.2f} - lý tưởng là > 0.7)")
    else:
        print(f"  ✅ Dữ liệu tương đối cân bằng: min={min_pct:.1f}%, max={max_pct:.1f}%")

    return unique_labels, counts


# ============================================================

def check_subjects(data, name):
    """Kiểm tra phân phối Subject"""
    print(f"\n👤 SUBJECT CHECK: {name}")
    print("-" * 50)

    subjects = data['subjects'].numpy()
    unique_subjects = np.unique(subjects)

    print(f"  Số subject unique: {len(unique_subjects)}")
    print(f"  Subject IDs: {sorted(unique_subjects)}")

    # Kiểm tra Subject IDs
    if name == "UCI-HAR":
        expected = set(range(1, 31))
        actual = set(unique_subjects)
        missing = sorted(expected - actual)
        if missing:
            print(f"  ⚠️ Thiếu subject IDs: {missing}")
        else:
            print("  ✅ Có đủ 30 subjects (1-30)")

    elif name == "MotionSense":
        expected = set(range(1, 25))
        actual = set(unique_subjects)
        missing = sorted(expected - actual)
        if missing:
            print(f"  ⚠️ Thiếu subject IDs: {missing}")
        else:
            print("  ✅ Có đủ 24 subjects (1-24)")

    # Phân phối số mẫu mỗi subject
    subject_counts = Counter(subjects)
    min_count = min(subject_counts.values())
    max_count = max(subject_counts.values())
    avg_count = sum(subject_counts.values()) / len(subject_counts)

    print(f"  Số mẫu/subject: min={min_count}, max={max_count}, avg={avg_count:.1f}")

    if max_count / min_count > 5:
        print(
            f"  ⚠️ WARNING: Subject {max(subject_counts, key=subject_counts.get)} có {max_count} mẫu, trong khi subject {min(subject_counts, key=subject_counts.get)} chỉ có {min_count} mẫu (chênh lệch lớn)")

    return unique_subjects


# ============================================================

def check_nan_inf(data, name):
    """Kiểm tra giá trị NaN và Inf"""
    print(f"\n🔍 NAN/INF CHECK: {name}")
    print("-" * 50)

    samples = data['samples']

    n_nan = torch.isnan(samples).sum().item()
    n_inf = torch.isinf(samples).sum().item()

    if n_nan > 0:
        print(f"  ❌ Có {n_nan:,} giá trị NaN!")
    else:
        print(f"  ✅ Không có NaN")

    if n_inf > 0:
        print(f"  ❌ Có {n_inf:,} giá trị Inf!")
    else:
        print(f"  ✅ Không có Inf")

    return n_nan == 0 and n_inf == 0


# ============================================================

def check_statistics(data, name):
    """Kiểm tra thống kê cơ bản"""
    print(f"\n📊 STATISTICS CHECK: {name}")
    print("-" * 50)

    samples = data['samples']

    # Tính thống kê cho từng kênh
    stats = []
    for c in range(samples.shape[1]):
        channel_data = samples[:, c, :].numpy().flatten()
        stats.append({
            'channel': c,
            'mean': np.mean(channel_data),
            'std': np.std(channel_data),
            'min': np.min(channel_data),
            'max': np.max(channel_data),
            'p1': np.percentile(channel_data, 1),
            'p99': np.percentile(channel_data, 99),
        })

    df_stats = pd.DataFrame(stats)

    print("  Thống kê theo kênh:")
    print(df_stats.round(4).to_string(index=False))

    # Kiểm tra scale
    global_mean = samples.mean().item()
    global_std = samples.std().item()

    print(f"\n  Global mean: {global_mean:.4f}")
    print(f"  Global std : {global_std:.4f}")

    if abs(global_mean) > 10:
        print("  ⚠️ WARNING: Mean lớn (>10) - có thể chưa được chuẩn hóa tốt")

    if global_std > 10:
        print("  ⚠️ WARNING: Std lớn (>10) - có thể có outlier hoặc chưa chuẩn hóa")

    return df_stats


# ============================================================

def compare_datasets(dataset1, dataset2, name1, name2):
    """So sánh 2 datasets với nhau"""
    print(f"\n🔄 COMPARISON: {name1} vs {name2}")
    print("=" * 60)

    d1_samples = dataset1['samples'].numpy()
    d2_samples = dataset2['samples'].numpy()

    # Mean, std của từng dataset
    print(f"\n📈 Global statistics:")
    print(f"  {name1}: mean={d1_samples.mean():.4f}, std={d1_samples.std():.4f}")
    print(f"  {name2}: mean={d2_samples.mean():.4f}, std={d2_samples.std():.4f}")

    # So sánh từng kênh
    print(f"\n📊 Channel-wise comparison:")
    for c in range(6):
        d1_c = d1_samples[:, c, :].flatten()
        d2_c = d2_samples[:, c, :].flatten()

        print(f"  Channel {c:2d}:")
        print(f"    {name1:12s}: mean={d1_c.mean():.4f}, std={d1_c.std():.4f}")
        print(f"    {name2:12s}: mean={d2_c.mean():.4f}, std={d2_c.std():.4f}")

        # Kiểm tra sự khác biệt
        mean_diff = abs(d1_c.mean() - d2_c.mean()) / max(abs(d1_c.mean()), abs(d2_c.mean()))
        std_diff = abs(d1_c.std() - d2_c.std()) / max(d1_c.std(), d2_c.std())

        if mean_diff > 0.5 or std_diff > 0.5:
            print(f"    ⚠️ Chênh lệch lớn giữa 2 datasets (mean diff: {mean_diff:.2f}, std diff: {std_diff:.2f})")


# ============================================================

def main():
    print("=" * 80)
    print("🔍 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU SAU TIỀN XỬ LÝ")
    print("=" * 80)

    all_results = {}

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy file: {path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"📂 ĐANG KIỂM TRA: {name}")
        print(f"📁 Path: {path}")
        print("=" * 80)

        # Load data
        data = torch.load(path, map_location='cpu')

        # Chọn label names
        label_names = UCI_LABELS if name == "UCI-HAR" else MOTION_LABELS

        # Run checks
        n_samples, n_channels, n_timesteps = check_shape(data, name)
        unique_labels, counts = check_labels(data, name, label_names)
        unique_subjects = check_subjects(data, name)
        is_clean = check_nan_inf(data, name)
        df_stats = check_statistics(data, name)

        all_results[name] = {
            'n_samples': n_samples,
            'n_channels': n_channels,
            'n_timesteps': n_timesteps,
            'n_labels': len(unique_labels),
            'n_subjects': len(unique_subjects),
            'is_clean': is_clean,
            'stats': df_stats
        }

    # So sánh 2 datasets nếu cả 2 đều có
    if len(all_results) == 2:
        names = list(all_results.keys())
        data1 = torch.load(DATASETS[names[0]], map_location='cpu')
        data2 = torch.load(DATASETS[names[1]], map_location='cpu')
        compare_datasets(data1, data2, names[0], names[1])

    # Tóm tắt
    print("\n" + "=" * 80)
    print("📝 TÓM TẮT KẾT QUẢ")
    print("=" * 80)

    for name, result in all_results.items():
        status = "✅ OK" if result['is_clean'] else "❌ CÓ VẤN ĐỀ"
        print(f"\n{name}:")
        print(f"  Số mẫu    : {result['n_samples']:,}")
        print(f"  Số kênh   : {result['n_channels']}")
        print(f"  Timesteps : {result['n_timesteps']}")
        print(f"  Số lớp    : {result['n_labels']}")
        print(f"  Số subject: {result['n_subjects']}")
        print(f"  Status    : {status}")

    print("\n✅ KIỂM TRA HOÀN TẤT!")


if __name__ == "__main__":
    main()