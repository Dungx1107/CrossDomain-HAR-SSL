"""
===============================================================================
MỤC TIÊU:
    - Đánh giá chuyển giao trực tiếp (Direct Transfer / Zero-shot) của mô hình
      SUPERVISED BASELINE được huấn luyện trên MotionSense sang tập Test của UCI-HAR.
    - Checkpoint sử dụng: checkpoints/baseline_cnn1d_motionsense_best.pt
    - Ánh xạ 6 lớp của MotionSense sang 5 lớp chuẩn của UCI-HAR.
    - Xuất Classification Report và Confusion Matrix chi tiết.
===============================================================================
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# Thiết lập đường dẫn thư mục gốc
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from models.har_classifier import HARClassifier
from utils.logger import SimpleLogger

# -----------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN & CHECKPOINT
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_TEST_PATH = UCIHARConfig.PROCESSED_TEST_PATH

SUPERVISED_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "baseline_cnn1d_motionsense_best.pt")

REPORT_DIR = os.path.join(PROJECT_ROOT, "document", "0_reports", "2_cross_domain_zero_shot")
LOG_FILE_PATH = os.path.join(REPORT_DIR, "supervised_transfer_eval_logs.txt")

# Ánh xạ từ nhãn MotionSense (6 lớp) -> UCI-HAR (5 lớp):
# MotionSense: 0: dws, 1: ups, 2: sit, 3: std, 4: wlk, 5: jog
# UCI-HAR:     0: Walking, 1: Upstairs, 2: Downstairs, 3: Sitting, 4: Standing
SRC_TO_TGT_MAPPING = {
    4: 0,   # wlk -> Walking
    1: 1,   # ups -> Upstairs
    0: 2,   # dws -> Downstairs
    2: 3,   # sit -> Sitting
    3: 4,   # std -> Standing
    5: -1   # jog -> Không có trong UCI-HAR
}

EVAL_LABELS = [0, 1, 2, 3, 4]


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    logger = SimpleLogger(LOG_FILE_PATH)
    sys.stdout = logger

    print("=" * 85)
    print("🚀 ĐÁNH GIÁ CHUYỂN GIAO TRỰC TIẾP: SUPERVISED (MotionSense -> UCI-HAR)")
    print(f"🖥️ Thiết bị sử dụng : {DEVICE}")
    print(f"💾 Checkpoint nguồn : {SUPERVISED_CHECKPOINT}")
    print(f"📄 File lưu log     : {LOG_FILE_PATH}")
    print("=" * 85)

    try:
        # 1. Kiểm tra và nạp dữ liệu Test UCI-HAR
        if not os.path.exists(TARGET_TEST_PATH):
            raise FileNotFoundError(f"❌ Không tìm thấy tập Test UCI-HAR tại: {TARGET_TEST_PATH}")

        test_data = torch.load(TARGET_TEST_PATH, map_location="cpu", weights_only=True)
        x_test, y_test = test_data["samples"], test_data["labels"]
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=64,
            shuffle=False
        )
        print(f"📦 Đã nạp tập Test UCI-HAR: {len(x_test)} mẫu | Shape: {x_test.shape}")

        # 2. Kiểm tra và nạp mô hình Supervised MotionSense
        if not os.path.exists(SUPERVISED_CHECKPOINT):
            raise FileNotFoundError(f"❌ Không tìm thấy file checkpoint tại: {SUPERVISED_CHECKPOINT}")

        model = HARClassifier(
            in_channels=MotionSenseConfig.IN_CHANNELS,
            num_classes=MotionSenseConfig.NUM_CLASSES
        ).to(DEVICE)

        state_dict = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict,strict=False)
        model.eval()

        # 3. Suy luận và ánh xạ nhãn
        all_mapped_preds = []
        all_targets = []

        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(DEVICE)
                logits = model(x_b)
                raw_preds = torch.argmax(logits, dim=1).cpu().numpy()

                for pred, target in zip(raw_preds, y_b.numpy()):
                    mapped_pred = SRC_TO_TGT_MAPPING.get(pred, -1)
                    all_mapped_preds.append(mapped_pred)
                    all_targets.append(target)

        all_mapped_preds = np.array(all_mapped_preds)
        all_targets = np.array(all_targets)

        # 4. Tính toán các chỉ số đánh giá
        acc = accuracy_score(all_targets, all_mapped_preds) * 100
        macro_f1 = f1_score(all_targets, all_mapped_preds, labels=EVAL_LABELS, average="macro", zero_division=0) * 100
        weighted_f1 = f1_score(all_targets, all_mapped_preds, labels=EVAL_LABELS, average="weighted", zero_division=0) * 100

        print("\n" + "=" * 85)
        print("🎯 KẾT QUẢ ZERO-SHOT CỦA PURE SUPERVISED (MotionSense -> UCI-HAR)")
        print("=" * 85)
        print(f"🎯 Test Accuracy   : {acc:6.2f}%")
        print(f"🏆 Test Macro F1   : {macro_f1:6.2f}% (Thước đo chính)")
        print(f"⚖️ Test Weighted F1 : {weighted_f1:6.2f}%")
        print("-" * 85)
        print("📋 BẢNG THỐNG KÊ CHI TIẾT TỪNG LỚP HÀNH ĐỘNG (UCI-HAR):")
        print(classification_report(
            all_targets,
            all_mapped_preds,
            labels=EVAL_LABELS,
            target_names=UCIHARConfig.CLASS_NAMES,
            digits=4,
            zero_division=0
        ))
        print("🔍 MA TRẬN NHẦM LẪN (UCI-HAR Classes: Walk, Up, Down, Sit, Stand):")
        print(confusion_matrix(all_targets, all_mapped_preds, labels=EVAL_LABELS))
        print("=" * 85)

    finally:
        logger.close()


if __name__ == "__main__":
    main()