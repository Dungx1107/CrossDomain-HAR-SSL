"""
===============================================================================
TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: MOTIONSENSE
1. Đọc dữ liệu CSV từ 'data/raw/motion_sense/'.
2. Trích xuất 6 kênh chuẩn: [userAcc_x, y, z, rotationRate_x, y, z].
3. Cắt cửa sổ trượt (window=128, stride=64, 50Hz).
4. Chia theo Subject IDs: Train (1-14), Val (15-18), Test (19-24).
5. Lưu ra các file .pt chuẩn hóa có kèm mảng 'subjects'.
===============================================================================
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig

RAW_DATA_DIR = MotionSenseConfig.RAW_DATA_DIR
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "motionsense")

# 6 kênh chuẩn hoá: 3 gia tốc người dùng + 3 con quay hồi chuyển
FEATURE_COLS = [
    'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
    'rotationRate.x', 'rotationRate.y', 'rotationRate.z'
]

LABEL_MAP = {
    'dws': 0,  # Downstairs
    'ups': 1,  # Upstairs
    'wlk': 2,  # Walking
    'sit': 3,  # Sitting
    'std': 4   # Standing
}


def process_subject_subset(subjects_list, subset_name: str = "train") -> dict:
    print(f"\n⏳ Đang xử lý tập: {subset_name.upper()} (Người dùng: {subjects_list})...")
    windows = []
    labels = []
    subjects = []

    folder_paths = glob.glob(os.path.join(RAW_DATA_DIR, "*_*"))

    for folder in folder_paths:
        if not os.path.isdir(folder):
            continue
        folder_name = os.path.basename(folder)
        act_code = folder_name.split('_')[0]

        # Chỉ lấy 5 lớp hành vi chung, bỏ qua 'jog' nếu có
        if act_code not in LABEL_MAP:
            continue

        label = LABEL_MAP[act_code]

        for sub_id in subjects_list:
            file_path = os.path.join(folder, f"sub_{sub_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                sensor_data = df[FEATURE_COLS].values
                num_samples = len(sensor_data)
                window_size = 128
                stride = 64

                for start in range(0, num_samples - window_size + 1, stride):
                    end = start + window_size
                    window = sensor_data[start:end].T  # Shape: (6, 128)
                    windows.append(window)
                    labels.append(label)
                    subjects.append(sub_id)

    samples_tensor = torch.tensor(np.array(windows), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
    subjects_tensor = torch.tensor(np.array(subjects), dtype=torch.long)

    print(f"   -> Mẫu thu được: {len(labels_tensor)} | Shape: {samples_tensor.shape}")
    print(f"   -> Phân phối nhãn: {torch.bincount(labels_tensor).tolist()}")

    return {
        "samples": samples_tensor,
        "labels": labels_tensor,
        "subjects": subjects_tensor
    }


def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: MOTIONSENSE")
    print(f"📂 Thư mục nguồn thô : {RAW_DATA_DIR}")
    print(f"💾 Thư mục lưu đầu ra : {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = process_subject_subset(list(range(1, 15)), "train")
    val_data = process_subject_subset(list(range(15, 19)), "val")
    test_data = process_subject_subset(list(range(19, 25)), "test")

    # Lưu 3 tập riêng biệt
    torch.save(train_data, os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "test.pt"))

    # Gộp toàn bộ vào dataset_all.pt phục vụ Pretrain SSL
    all_data = {
        "samples": torch.cat([train_data["samples"], val_data["samples"], test_data["samples"]], dim=0),
        "labels": torch.cat([train_data["labels"], val_data["labels"], test_data["labels"]], dim=0),
        "subjects": torch.cat([train_data["subjects"], val_data["subjects"], test_data["subjects"]], dim=0)
    }
    torch.save(all_data, os.path.join(OUTPUT_DIR, "dataset_all.pt"))
    print("\n✅ Hoàn thành đóng gói toàn bộ file .pt cho MotionSense!")


if __name__ == "__main__":
    main()