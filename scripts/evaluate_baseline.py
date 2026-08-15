"""
===============================================================================
SCRIPT ĐÁNH GIÁ CHI TIẾT MÔ HÌNH BASELINE (METRICS & CONFUSION MATRIX)
===============================================================================
Mục đích:
    - Nạp file trọng số tốt nhất đã lưu: checkpoints/baseline_cnn1d_motionsense_best.pt
    - Chạy dự đoán trên toàn bộ tập Test (6 người dùng độc lập).
    - Tính toán: Accuracy, Precision, Recall, Macro F1-Score.
    - In Ma trận nhầm lẫn (Confusion Matrix) dạng bảng trực quan.
===============================================================================
"""

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import cấu hình, Loader và Model
from config.motionsense_config import MotionSenseConfig
from datasets.motionsense import get_motionsense_dataloaders
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.baseline.supervised_model import SupervisedHARModel


def evaluate_and_report():
    DATASET_NAME = "motionsense"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Đang sử dụng thiết bị: {device}")

    # 1. Nạp DataLoader tập Test
    _, _, test_loader = get_motionsense_dataloaders()

    # 2. Khởi tạo lại kiến trúc mô hình
    encoder = StandardSensorEncoder1D(
        in_channels=MotionSenseConfig.IN_CHANNELS,
        feature_dim=128
    )
    classifier = ClassifierHead(
        feature_dim=128,
        num_classes=MotionSenseConfig.NUM_CLASSES
    )
    model = SupervisedHARModel(encoder=encoder, classifier=classifier).to(device)

    # 3. Nạp file trọng số Checkpoint đã lưu từ Bước 4
    checkpoint_path = os.path.join("checkpoints", f"baseline_cnn1d_{DATASET_NAME}_best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"❌ KHÔNG TÌM THẤY FILE CHECKPOINT TẠI: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Đã nạp thành công trọng số từ: {checkpoint_path}")

    # 4. Chạy dự đoán trên tập Test
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Lấy nhãn dự đoán có xác suất cao nhất
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # -------------------------------------------------------------------------
    # 5. TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ (METRICS)
    # -------------------------------------------------------------------------
    class_names = MotionSenseConfig.CLASS_NAMES  # ['Downstairs', 'Upstairs', 'Walking', 'Jogging', 'Sitting', 'Standing']

    print("\n" + "=" * 70)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CHI TIẾT (CLASSIFICATION REPORT)")
    print("=" * 70)

    # In Precision, Recall, F1-Score cho từng lớp
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4
    )
    print(report)

    # -------------------------------------------------------------------------
    # 6. IN MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
    # -------------------------------------------------------------------------
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 70)
    print("🧩 MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
    print("   (Hàng dọc: Nhãn Thực Tế | Cột ngang: Nhãn Dự Đoán)")
    print("=" * 70)

    # Header cột
    header = f"{'Thực tế / Dự đoán':<20} | " + " | ".join([f"{name[:6]:^6}" for name in class_names])
    print(header)
    print("-" * len(header))

    # In từng hàng ma trận
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:^6d}" for val in row])
        print(f"{class_names[i]:<20} | {row_str}")
    print("=" * 70)


if __name__ == '__main__':
    evaluate_and_report()
