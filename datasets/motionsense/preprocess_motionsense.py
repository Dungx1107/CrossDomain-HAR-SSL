"""
===============================================================================
1. Đọc dữ liệu cảm biến thô (.csv) từ 'data/raw/motion_sense/'.
2. Sử dụng MotionSenseDataset để cắt cửa sổ trượt (6 kênh: 3 acc + 3 gyro, 128 samples).
3. Chia theo danh sách người dùng (Subject IDs):
   - Train: subjects 1-14
   - Val:   subjects 15-18
   - Test:  subjects 19-24
4. Đóng gói thành PyTorch Tensor {'samples': (N, 6, 128), 'labels': (N,)}
5. Lưu các file .pt vào 'data/processed/motion_sense/'.
===============================================================================
"""

import os
import sys
import torch

# =============================================================================
# 1. TỰ ĐỘNG TÍNH ĐƯỜNG DẪN GỐC DỰ ÁN (PROJECT ROOT)
# File này nằm tại: CrossDomain-HAR-SSL/datasets/motionsense/preprocess_motionsense.py
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset

# Thư mục nguồn thô và thư mục lưu đầu ra
RAW_DATA_DIR = MotionSenseConfig.RAW_DATA_DIR
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "motionsense")


def process_subject_subset(subjects_list, subset_name: str = "train") -> dict:
    """
    Nạp dữ liệu từ danh sách người dùng và chuyển thành Dictionary chứa PyTorch Tensor.
    """
    print(f"\n⏳ Đang xử lý tập: {subset_name.upper()} (Người dùng: {subjects_list})...")

    dataset = MotionSenseDataset(
        data_dir=RAW_DATA_DIR,
        subjects_list=subjects_list,
        config=MotionSenseConfig
    )

    # dataset.windows: Tensor Shape (N, 6, 128)
    # dataset.labels: Tensor Shape (N,)
    samples_tensor = dataset.windows
    labels_tensor = dataset.labels

    print(f"   -> Số lượng mẫu thu được: {len(labels_tensor)}")
    print(f"   -> Kích thước Tensor Samples: {samples_tensor.shape}")
    print(f"   -> Phân phối nhãn: {torch.bincount(labels_tensor).tolist()}")

    return {
        "samples": samples_tensor,
        "labels": labels_tensor
    }


def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: MOTIONSENSE")
    print(f"📂 Thư mục nguồn thô : {RAW_DATA_DIR}")
    print(f"💾 Thư mục lưu đầu ra : {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Xử lý từng tập theo Subject IDs
    train_data = process_subject_subset(MotionSenseConfig.TRAIN_SUBJECTS, "train")
    val_data = process_subject_subset(MotionSenseConfig.VAL_SUBJECTS, "val")
    test_data = process_subject_subset(MotionSenseConfig.TEST_SUBJECTS, "test")

    # 2. Định nghĩa đường dẫn lưu file
    train_file = os.path.join(OUTPUT_DIR, "train.pt")
    val_file = os.path.join(OUTPUT_DIR, "val.pt")
    test_file = os.path.join(OUTPUT_DIR, "test.pt")
    all_file = os.path.join(OUTPUT_DIR, "dataset_all.pt")

    # 3. Lưu các tập riêng biệt
    torch.save(train_data, train_file)
    torch.save(val_data, val_file)
    torch.save(test_data, test_file)

    # 4. Gộp toàn bộ thành file dataset_all.pt
    all_data = {
        "samples": torch.cat([train_data["samples"], val_data["samples"], test_data["samples"]], dim=0),
        "labels": torch.cat([train_data["labels"], val_data["labels"], test_data["labels"]], dim=0)
    }
    torch.save(all_data, all_file)

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH TIỀN XỬ LÝ MOTIONSENSE THÀNH CÔNG!")
    print(f"📁 File Train : {train_file} | Shape: {train_data['samples'].shape}")
    print(f"📁 File Val   : {val_file}   | Shape: {val_data['samples'].shape}")
    print(f"📁 File Test  : {test_file}  | Shape: {test_data['samples'].shape}")
    print(f"📁 File Gộp   : {all_file}   | Shape: {all_data['samples'].shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()