"""
===============================================================================
1. Đọc dữ liệu cảm biến thô (.txt) từ 'data/raw/uci_har/' cho cả Train và Test.
2. Ghép 6 kênh tín hiệu chuẩn: [body_acc_x, y, z, body_gyro_x, y, z].
3. Lọc bỏ hoạt động 'Laying' (nhãn số 6) để lấy 5 lớp hoạt động chung.
4. Ánh xạ nhãn từ 1..5 về 0..4 để tương thích với PyTorch CrossEntropyLoss.
5. Đóng gói thành PyTorch Tensor và lưu vào 'data/processed/uci_har/'.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# =============================================================================
# 1. TỰ ĐỘNG TÍNH ĐƯỜNG DẪN GỐC DỰ ÁN (PROJECT ROOT)
# File này nằm tại: CrossDomain-HAR-SSL/datasets/uci_har/preprocess_uci_har.py
# -> os.path.dirname(__file__)       = datasets/uci_har
# -> Lùi thêm 1 cấp (..)            = datasets
# -> Lùi thêm 1 cấp nữa (../..)     = Thư mục gốc CrossDomain-HAR-SSL
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

# Đường dẫn thư mục dữ liệu thô và thư mục lưu kết quả chuẩn hóa
RAW_UCI_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "uci_har")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "uci_har")

# =============================================================================
# 2. CẤU HÌNH KÊNH CẢM BIẾN VÀ ÁNH XẠ NHÃN
# =============================================================================
# Danh sách 6 file tín hiệu cốt lõi nằm trong thư mục 'Inertial Signals/'
# Bỏ qua 3 file 'total_acc' vì MotionSense dùng 'userAcceleration' (đã trừ trọng lực)
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",  # 3 trục Gia tốc thân người
    "body_gyro_x", "body_gyro_y", "body_gyro_z"  # 3 trục Vận tốc góc con quay
]

# Bảng ánh xạ 5 lớp hoạt động chung:
# - Nhãn 1 -> 5 của UCI-HAR chuyển thành 0 -> 4 cho PyTorch
# - Nhãn 6 (Laying) không xuất hiện trong dict này -> sẽ bị lọc bỏ
LABEL_MAPPING = {
    1: 0,  # 1: WALKING           -> 0
    2: 1,  # 2: WALKING_UPSTAIRS  -> 1
    3: 2,  # 3: WALKING_DOWNSTAIRS-> 2
    4: 3,  # 4: SITTING           -> 3
    5: 4  # 5: STANDING          -> 4
}

CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']


def load_signals(subset_type: str = "train") -> np.ndarray:
    """
    Đọc 6 file .txt tín hiệu trong Inertial Signals/ và xếp chồng thành mảng 3D.
    Args:
        subset_type (str): 'train' hoặc 'test'
    Returns:
        np.ndarray: Mảng tín hiệu có kích thước (N, 6, 128)
                    với N: số lượng cửa sổ, 6: số kênh, 128: số điểm đo.
    """
    subset_dir = os.path.join(RAW_UCI_DIR, subset_type, "Inertial Signals")

    # Kiểm tra fallback đề phòng trường hợp thư mục lồng nhau
    if not os.path.exists(subset_dir):
        subset_dir = os.path.join(RAW_UCI_DIR, "UCI HAR Dataset", subset_type, "Inertial Signals")

    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục tín hiệu: {subset_dir}")

    signals_list = []
    for sig in SIGNAL_NAMES:
        file_path = os.path.join(subset_dir, f"{sig}_{subset_type}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Thiếu file kênh tín hiệu: {file_path}")

        # Đọc file .txt với dấu phân tách khoảng trắng linh hoạt
        # Mỗi dòng trong file tương ứng 128 điểm đo của 1 cửa sổ -> Shape trả về: (N, 128)
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
        signals_list.append(df.values)

    # Xếp chồng 6 kênh theo trục axis=1 (Kênh tín hiệu) -> Kích thước: (N, 6, 128)
    signals = np.stack(signals_list, axis=1)
    return signals


def process_subset(subset_type: str = "train") -> dict:
    """
    Xử lý toàn bộ dữ liệu cho 1 tập (Train hoặc Test):
    Nạp tín hiệu, nạp nhãn, lọc bỏ Laying, ánh xạ nhãn và chuyển sang PyTorch Tensor.
    Args:
        subset_type (str): 'train' hoặc 'test'
    Returns:
        dict: Chứa các Tensor {'samples': (N, 6, 128), 'labels': (N,), 'subjects': (N,)}
    """
    print(f"\n⏳ Đang xử lý tập: {subset_type.upper()}...")

    # 1. Nạp tín hiệu 6 kênh
    signals = load_signals(subset_type)  # Shape: (N, 6, 128)

    # 2. Nạp nhãn hoạt động (y) và ID người thực hiện (subject)
    base_dir = os.path.join(RAW_UCI_DIR, subset_type)
    if not os.path.exists(base_dir):
        base_dir = os.path.join(RAW_UCI_DIR, "UCI HAR Dataset", subset_type)

    y_path = os.path.join(base_dir, f"y_{subset_type}.txt")
    sub_path = os.path.join(base_dir, f"subject_{subset_type}.txt")

    y = pd.read_csv(y_path, sep=r'\s+', header=None).values.squeeze()
    subjects = pd.read_csv(sub_path, sep=r'\s+', header=None).values.squeeze()

    raw_count = len(y)
    print(f"   -> Số lượng mẫu thô ban đầu: {raw_count}")

    # 3. Lọc bỏ nhãn Laying (chỉ giữ các nhãn nằm trong LABEL_MAPPING: 1, 2, 3, 4, 5)
    valid_mask = np.isin(y, list(LABEL_MAPPING.keys()))
    signals_filtered = signals[valid_mask]
    y_filtered = y[valid_mask]
    subjects_filtered = subjects[valid_mask]

    # 4. Ánh xạ các nhãn 1..5 về 0..4
    y_mapped = np.array([LABEL_MAPPING[lbl] for lbl in y_filtered], dtype=np.int64)

    # 5. Chuyển đổi sang PyTorch Tensor
    samples_tensor = torch.tensor(signals_filtered, dtype=torch.float32)  # Shape: (N_filtered, 6, 128)
    labels_tensor = torch.tensor(y_mapped, dtype=torch.long)  # Shape: (N_filtered,)
    subjects_tensor = torch.tensor(subjects_filtered, dtype=torch.long)  # Shape: (N_filtered,)

    print(
        f"   -> Số mẫu sau khi lọc (5 lớp chung): {len(labels_tensor)} (Đã loại bỏ {raw_count - len(labels_tensor)} mẫu 'Laying')")
    print(f"   -> Kích thước Tensor Samples: {samples_tensor.shape}")

    return {
        "samples": samples_tensor,
        "labels": labels_tensor,
        "subjects": subjects_tensor
    }


def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU ĐÍCH: UCI-HAR")
    print(f"📂 Thư mục nguồn thô : {RAW_UCI_DIR}")
    print(f"💾 Thư mục lưu đầu ra : {OUTPUT_DIR}")
    print("=" * 80)

    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Xử lý riêng biệt Train và Test
    train_data = process_subset("train")
    test_data = process_subset("test")

    # Đường dẫn file lưu
    train_file = os.path.join(OUTPUT_DIR, "train.pt")
    test_file = os.path.join(OUTPUT_DIR, "test.pt")
    all_file = os.path.join(OUTPUT_DIR, "dataset_all.pt")

    # 1. Lưu file train.pt và test.pt
    torch.save(train_data, train_file)
    torch.save(test_data, test_file)

    # 2. Gộp cả 2 tập thành 1 file dataset_all.pt để linh hoạt chia lại tập dữ liệu nếu cần
    all_data = {
        "samples": torch.cat([train_data["samples"], test_data["samples"]], dim=0),
        "labels": torch.cat([train_data["labels"], test_data["labels"]], dim=0),
        "subjects": torch.cat([train_data["subjects"], test_data["subjects"]], dim=0)
    }
    torch.save(all_data, all_file)

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH TIỀN XỬ LÝ UCI-HAR THÀNH CÔNG!")
    print(f"📁 File Train : {train_file} | Shape: {train_data['samples'].shape}")
    print(f"📁 File Test  : {test_file}  | Shape: {test_data['samples'].shape}")
    print(f"📁 File Gộp   : {all_file}   | Shape: {all_data['samples'].shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
