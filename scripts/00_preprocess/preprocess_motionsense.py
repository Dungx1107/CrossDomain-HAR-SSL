"""
TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: MOTIONSENSE
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

WINDOW_SIZE = MotionSenseConfig.WINDOW_SIZE
STRIDE = MotionSenseConfig.STRIDE

TRAIN_SUBJECTS = MotionSenseConfig.TRAIN_SUBJECTS
VAL_SUBJECTS = MotionSenseConfig.VAL_SUBJECTS
TEST_SUBJECTS = MotionSenseConfig.TEST_SUBJECTS

RAW_COLS_USER_ACC = MotionSenseConfig.RAW_COLS_USER_ACC
RAW_COLS_GRAVITY = MotionSenseConfig.RAW_COLS_GRAVITY
RAW_COLS_ROTATION = MotionSenseConfig.RAW_COLS_ROTATION

LABEL_MAP = {
    'wlk': 0,  # Walking
    'ups': 1,  # Upstairs
    'dws': 2,  # Downstairs
    'sit': 3,  # Sitting
    'std': 4,  # Standing

    'jog': 5  # Jogging # ⚠️ Chỉ có ở MotionSense (UCI-HAR không có)
}


def process_subject_subset(
        subjects_list,
        subset_name: str = "train"
) -> dict:
    print(f"\n⏳ Đang xử lý tập: {subset_name.upper()} (Người dùng: {subjects_list})...")
    windows = []
    labels = []
    subjects = []

    folder_paths = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*_*")))

    for folder in folder_paths:
        if not os.path.isdir(folder):
            continue
        folder_name = os.path.basename(folder)
        act_code = folder_name.split('_')[0]

        if act_code not in LABEL_MAP: continue

        label = LABEL_MAP[act_code]

        for sub_id in subjects_list:
            file_path = os.path.join(folder, f"sub_{sub_id}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)

                # ---- Trích xuất các cột thô ----
                user_acc = df[RAW_COLS_USER_ACC].values.astype(np.float32)
                gravity = df[RAW_COLS_GRAVITY].values.astype(np.float32)
                gyro = df[RAW_COLS_ROTATION].values.astype(np.float32)

                # ---- Tính total_acc = userAcc + gravity ----
                total_acc = user_acc + gravity

                # Ghép 6 kênh: [total_acc (3), gyro (3)]
                sensor_data = np.concatenate([total_acc, gyro], axis=1)
                num_samples = len(sensor_data)

                # Dùng WINDOW_SIZE và STRIDE từ config  # Bản chất: len(labels) chính là tổng số sample
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
    all_data = {
        "samples": torch.cat([train_data["samples"], val_data["samples"], test_data["samples"]], dim=0),
        "labels": torch.cat([train_data["labels"], val_data["labels"], test_data["labels"]], dim=0),
        "subjects": torch.cat([train_data["subjects"], val_data["subjects"], test_data["subjects"]], dim=0)
    }

    torch.save(train_data, os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "test.pt"))
    torch.save(all_data, os.path.join(OUTPUT_DIR, "dataset_all.pt"))

    print("\n✅ Hoàn thành đóng gói toàn bộ file .pt cho MotionSense!")


def test():
    print("\n" + "=" * 80)
    print("🔍 KIỂM TRA ĐẶC TRƯNG SAU ĐÓNG GÓI (6 KÊNH)")
    print("=" * 80)

    file_path = MotionSenseConfig.DATA_ALL_PATH
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy file: {file_path}")
        return

    data = torch.load(file_path)
    X = data["samples"]  # Shape: (N, 6, 128)
    y = data["labels"]

    print("Shape:", X.shape)
    print("Kênh 0-2 (total_acc) — mean/std:",
          round(X[:, 0:3, :].mean().item(), 4), round(X[:, 0:3, :].std().item(), 4))
    print("Kênh 3-5 (gyro)      — mean/std:",
          round(X[:, 3:6, :].mean().item(), 4), round(X[:, 3:6, :].std().item(), 4))

    # Kiểm tra magnitude trung bình của total_acc ở các hoạt động tĩnh (Sitting = 3, Standing = 4)
    # Kỳ vọng: Gia tốc tổng ở trạng thái tĩnh phải tiệm cận ~1.0g do trọng lực
    static_mask = (y == 3) | (y == 4)
    if static_mask.sum() > 0:
        static_acc = X[static_mask, 0:3, :]
        mag = torch.sqrt((static_acc ** 2).sum(dim=1))
        print("Độ lớn gia tốc ở tư thế tĩnh (Sit/Stand) — mean:", round(mag.mean().item(), 4))
        print("👉 Nhận xét: Giá trị ~1.0g xác nhận thành phần trọng lực đã được cộng vào thành công!")


if __name__ == "__main__":
    main()
    test()
