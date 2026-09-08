"""
===============================================================================
TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: MOTIONSENSE
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
OUTPUT_DIR = MotionSenseConfig.PROCESSED_DIR
FEATURE_COLS = MotionSenseConfig.FEATURE_COLS
WINDOW_SIZE = MotionSenseConfig.WINDOW_SIZE
STRIDE = MotionSenseConfig.STRIDE
TRAIN_SUBJECTS = MotionSenseConfig.TRAIN_SUBJECTS
VAL_SUBJECTS = MotionSenseConfig.VAL_SUBJECTS
TEST_SUBJECTS = MotionSenseConfig.TEST_SUBJECTS

# ✅ LABEL_MAP vẫn để ở đây vì nó ánh xạ từ tên file CSV
# (khác với LABEL_MAP trong config dùng cho tên lớp)
LABEL_MAP = {
    'dws': 0,  # Downstairs
    'ups': 1,  # Upstairs
    'wlk': 2,  # Walking
    'sit': 3,  # Sitting
    'std': 4,  # Standing
    'jog': 5   # Jogging
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

        if act_code not in LABEL_MAP:
            continue

        label = LABEL_MAP[act_code]

        for sub_id in subjects_list:
            file_path = os.path.join(folder, f"sub_{sub_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                sensor_data = df[FEATURE_COLS].values
                num_samples = len(sensor_data)

                # ✅ Dùng WINDOW_SIZE và STRIDE từ config
                for start in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
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
    print(f"📐 Window: {WINDOW_SIZE}, Stride: {STRIDE}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ✅ Dùng TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS từ config
    train_data = process_subject_subset(TRAIN_SUBJECTS, "train")
    val_data = process_subject_subset(VAL_SUBJECTS, "val")
    test_data = process_subject_subset(TEST_SUBJECTS, "test")

    torch.save(train_data, os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "test.pt"))

    all_data = {
        "samples": torch.cat([train_data["samples"], val_data["samples"], test_data["samples"]], dim=0),
        "labels": torch.cat([train_data["labels"], val_data["labels"], test_data["labels"]], dim=0),
        "subjects": torch.cat([train_data["subjects"], val_data["subjects"], test_data["subjects"]], dim=0)
    }
    torch.save(all_data, os.path.join(OUTPUT_DIR, "dataset_all.pt"))
    print("\n✅ Hoàn thành đóng gói toàn bộ file .pt cho MotionSense!")


if __name__ == "__main__":
    main()