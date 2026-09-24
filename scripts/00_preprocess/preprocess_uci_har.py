"""
===============================================================================
TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: UCI-HAR
1. Đọc tín hiệu 6 kênh từ thư mục 'Inertial Signals/'.
2. Giữ nguyên tất cả 6 nhãn hoạt động (bao gồm Laying).
3. Ánh xạ nhãn 1..6 về 0..5 cho PyTorch.
4. Tách tập Train gốc (21 subjects) thành Train (17 subjects) và Val (4 subjects).
5. Giữ nguyên Test gốc (9 subjects).
6. Lưu ra các file: train.pt, val.pt, test.pt, dataset_all.pt.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.uci_har_config import UCIHARConfig

RAW_UCI_DIR = UCIHARConfig.RAW_DATA_DIR
OUTPUT_DIR = UCIHARConfig.PROCESSED_DIR
SIGNAL_NAMES = UCIHARConfig.SIGNAL_NAMES
LABEL_MAPPING = UCIHARConfig.LABEL_MAPPING
VAL_SUBJECTS_UCI = UCIHARConfig.VAL_SUBJECTS


def load_signals(subset_type: str = "train") -> np.ndarray:
    """Đọc 6 file tín hiệu và stack thành 1 tensor (N, 6, 128)."""
    subset_dir = os.path.join(RAW_UCI_DIR, subset_type, "Inertial Signals")

    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục tín hiệu: {subset_dir}")

    signals_list = []
    for sig in SIGNAL_NAMES:
        file_path = os.path.join(subset_dir, f"{sig}_{subset_type}.txt")
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
        signals_list.append(df.values)

    return np.stack(signals_list, axis=1)  # Shape: (N, 6, 128)


def process_raw_subset(subset_type: str = "train"):
    """Đọc tín hiệu, nhãn và subject từ tập train/test."""
    signals = load_signals(subset_type)

    base_dir = os.path.join(RAW_UCI_DIR, subset_type)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục tín hiệu: {base_dir}")

    y_path = os.path.join(base_dir, f"y_{subset_type}.txt")
    sub_path = os.path.join(base_dir, f"subject_{subset_type}.txt")

    y = pd.read_csv(y_path, sep=r'\s+', header=None).values.squeeze()
    subjects = pd.read_csv(sub_path, sep=r'\s+', header=None).values.squeeze()

    y_mapped = np.array([LABEL_MAPPING[lbl] for lbl in y], dtype=np.int64)

    return (
        torch.tensor(signals, dtype=torch.float32),
        torch.tensor(y_mapped, dtype=torch.long),
        torch.tensor(subjects, dtype=torch.long)
    )


def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ VÀ ĐÓNG GÓI DỮ LIỆU: UCI-HAR")
    print(f"📂 Thư mục nguồn thô : {RAW_UCI_DIR}")
    print(f"💾 Thư mục lưu đầu ra : {OUTPUT_DIR}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Đọc tập Train và Test thô
    X_train_raw, y_train_raw, sub_train_raw = process_raw_subset("train")
    X_test, y_test, sub_test = process_raw_subset("test")

    print(f"📊 Train raw: {X_train_raw.shape} samples")
    print(f"📊 Test raw : {X_test.shape} samples")

    # 2. Tách Validation từ tập Train gốc dựa theo Subject ID
    val_mask = torch.isin(sub_train_raw, torch.tensor(VAL_SUBJECTS_UCI))
    train_mask = ~val_mask

    train_data = {
        "samples": X_train_raw[train_mask],
        "labels": y_train_raw[train_mask],
        "subjects": sub_train_raw[train_mask]
    }

    val_data = {
        "samples": X_train_raw[val_mask],
        "labels": y_train_raw[val_mask],
        "subjects": sub_train_raw[val_mask]
    }

    test_data = {
        "samples": X_test,
        "labels": y_test,
        "subjects": sub_test
    }

    # 3. Gộp cả 3 tập vào dataset_all.pt (Dùng cho SSL Pretrain)
    all_data = {
        "samples": torch.cat([X_train_raw, X_test], dim=0),
        "labels": torch.cat([y_train_raw, y_test], dim=0),
        "subjects": torch.cat([sub_train_raw, sub_test], dim=0)
    }

    # 4. Lưu toàn bộ xuống ổ đĩa
    torch.save(train_data, os.path.join(OUTPUT_DIR, "train.pt"))
    torch.save(val_data, os.path.join(OUTPUT_DIR, "val.pt"))
    torch.save(test_data, os.path.join(OUTPUT_DIR, "test.pt"))
    torch.save(all_data, os.path.join(OUTPUT_DIR, "dataset_all.pt"))

    print(f"\n📁 Train: {train_data['samples'].shape} (Subjects: {torch.unique(train_data['subjects']).tolist()})")
    print(f"📁 Val  : {val_data['samples'].shape} (Subjects: {torch.unique(val_data['subjects']).tolist()})")
    print(f"📁 Test : {test_data['samples'].shape} (Subjects: {torch.unique(test_data['subjects']).tolist()})")
    print(f"📁 All  : {all_data['samples'].shape}")
    print("\n✅ Hoàn thành đóng gói toàn bộ file .pt cho UCI-HAR!")

def test():
    print("\n" + "=" * 80)
    print("🔍 KIỂM TRA ĐẶC TRƯNG SAU ĐÓNG GÓI: UCI-HAR")
    print("=" * 80)

    file_path = os.path.join(OUTPUT_DIR, "dataset_all.pt")
    if not os.path.exists(file_path):
        print(f"⚠️ File nahi mila: {file_path}")
        return

    data = torch.load(file_path)
    X = data["samples"]  # Shape: (N, 6, 128)
    y = data["labels"]

    print("Shape:", X.shape)
    print("Kênh 0-2 (total_acc) — mean/std:",
          round(X[:, 0:3, :].mean().item(), 4), round(X[:, 0:3, :].std().item(), 4))
    print("Kênh 3-5 (gyro)      — mean/std:",
          round(X[:, 3:6, :].mean().item(), 4), round(X[:, 3:6, :].std().item(), 4))

    # Sitting (3) aur Standing (4) par total acceleration magnitude check
    static_mask = (y == 3) | (y == 4)
    if static_mask.sum() > 0:
        static_acc = X[static_mask, 0:3, :]
        mag = torch.sqrt((static_acc ** 2).sum(dim=1))
        print("Độ lớn gia tốc ở tư thế tĩnh (Sit/Stand) — mean:", round(mag.mean().item(), 4))

if __name__ == "__main__":
    main()
    test()
