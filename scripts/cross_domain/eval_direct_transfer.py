"""
===============================================================================
MỤC TIÊU:
    1. Đánh giá khả năng chuyển giao trực tiếp (Direct Transfer / Zero-shot)
       từ Source Domain (MotionSense) sang Target Domain (UCI-HAR).
    2. So sánh 2 mô hình nguồn:
       - Linear Probed Best Model (SSL Features + Linear Head).
       - Full Fine-tuned Best Model (Full Network Fine-tuned).
    3. Tự động ánh xạ đầu ra từ không gian 6 lớp của MotionSense về 5 lớp của UCI-HAR.
    4. Xuất bảng Classification Report, Confusion Matrix và lưu log chi tiết.
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
# CẤU HÌNH ĐƯỜNG DẪN & CHECKPOINTS
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_TEST_PATH = UCIHARConfig.PROCESSED_TEST_PATH

LINEAR_PROBED_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "motionsense_linear_probed_best.pt")
FULL_FINETUNED_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "motionsense_full_finetuned_best.pt")

REPORT_DIR = os.path.join(PROJECT_ROOT, "document", "0_reports", "2_cross_domain_zero_shot")
LOG_FILE_PATH = os.path.join(REPORT_DIR, "direct_transfer_eval_logs.txt")

# Ánh xạ từ chỉ số đầu ra của MotionSense (6 lớp) sang nhãn chuẩn của UCI-HAR (5 lớp):
# MotionSense: 0: dws, 1: ups, 2: sit, 3: std, 4: wlk, 5: jog
# UCI-HAR:     0: Walking, 1: Upstairs, 2: Downstairs, 3: Sitting, 4: Standing
SRC_TO_TGT_MAPPING = {
    4: 0,  # wlk -> Walking
    1: 1,  # ups -> Upstairs
    0: 2,  # dws -> Downstairs
    2: 3,  # sit -> Sitting
    3: 4,  # std -> Standing
    5: -1  # jog -> Không có trong UCI-HAR
}


def evaluate_single_model(model_path, test_loader, model_name):
    """Nạp checkpoint mô hình nguồn và đánh giá trên tập Test của UCI-HAR."""
    print("\n" + "=" * 85)
    print(f"🔍 ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print(f"💾 Checkpoint: {model_path}")
    print("=" * 85)

    if not os.path.exists(model_path):
        print(f"❌ CẢNH BÁO: Chưa tìm thấy file checkpoint tại: {model_path}")
        return None

    # Khởi tạo mô hình theo cấu trúc nguồn của MotionSense (6 kênh, 6 lớp)
    model = HARClassifier(
        in_channels=MotionSenseConfig.IN_CHANNELS,
        num_classes=MotionSenseConfig.NUM_CLASSES
    ).to(DEVICE)

    # Nạp toàn bộ trọng số (gồm cả Backbone và ClassifierHead)
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    all_raw_preds = []
    all_mapped_preds = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(DEVICE)
            logits = model(x_b)
            raw_preds = torch.argmax(logits, dim=1).cpu().numpy()

            for pred, target in zip(raw_preds, y_b.numpy()):
                mapped_pred = SRC_TO_TGT_MAPPING.get(pred, -1)
                all_raw_preds.append(pred)
                all_mapped_preds.append(mapped_pred)
                all_targets.append(target)

    all_mapped_preds = np.array(all_mapped_preds)
    all_targets = np.array(all_targets)

    # Danh sách 5 nhãn mục tiêu của UCI-HAR
    eval_labels = [0, 1, 2, 3, 4]

    # Tính toán các chỉ số chuẩn xác trên 5 lớp (loại trừ nhãn -1 nếu có)
    acc = accuracy_score(all_targets, all_mapped_preds) * 100
    macro_f1 = f1_score(all_targets, all_mapped_preds, labels=eval_labels, average="macro") * 100
    weighted_f1 = f1_score(all_targets, all_mapped_preds, labels=eval_labels, average="weighted") * 100

    print(f"🎯 Test Accuracy   : {acc:6.2f}%")
    print(f"🏆 Test Macro F1   : {macro_f1:6.2f}% (Thước đo chính)")
    print(f"⚖️ Test Weighted F1 : {weighted_f1:6.2f}%")
    print("-" * 85)
    print("📋 BẢNG THỐNG KÊ CHI TIẾT TỪNG LỚP HÀNH ĐỘNG:")
    print(classification_report(
        all_targets,
        all_mapped_preds,
        labels=eval_labels,
        target_names=UCIHARConfig.CLASS_NAMES,
        digits=4,
        zero_division=0
    ))
    print("🔍 MA TRẬN NHẦM LẪN (UCI-HAR Classes):")
    print(confusion_matrix(all_targets, all_mapped_preds, labels=eval_labels))

    return {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    logger = SimpleLogger(LOG_FILE_PATH)
    sys.stdout = logger

    print("=" * 85)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ CHUYỂN GIAO TRỰC TIẾP (DIRECT TRANSFER: MotionSense -> UCI-HAR)")
    print(f"🖥️ Thiết bị sử dụng : {DEVICE}")
    print(f"💾 File lưu logs     : {LOG_FILE_PATH}")
    print("=" * 85)

    try:
        # 1. Nạp tập Test của UCI-HAR
        if not os.path.exists(TARGET_TEST_PATH):
            raise FileNotFoundError(f"❌ Không tìm thấy tập Test UCI-HAR tại: {TARGET_TEST_PATH}")

        test_data = torch.load(TARGET_TEST_PATH, map_location="cpu", weights_only=True)
        x_test, y_test = test_data["samples"], test_data["labels"]
        test_loader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=64,
            shuffle=False
        )
        print(f"📦 Đã nạp tập Test UCI-HAR: {x_test.shape[0]} mẫu | Shape: {x_test.shape}")

        # 2. Đánh giá mô hình 1: Linear Probed Model
        res_linear = evaluate_single_model(
            LINEAR_PROBED_CHECKPOINT,
            test_loader,
            "1. Best Linear Probed Model (SSL Features Frozen)"
        )

        # 3. Đánh giá mô hình 2: Full Fine-tuned Model
        res_finetune = evaluate_single_model(
            FULL_FINETUNED_CHECKPOINT,
            test_loader,
            "2. Best Full Fine-Tuned Model (Unfrozen Network)"
        )

        # 4. Bảng so sánh tổng kết
        print("\n" + "=" * 85)
        print("🏆 BẢNG TỔNG KẾT HIỆU NĂNG CHUYỂN GIAO TRỰC TIẾP (DIRECT TRANSFER BENCHMARK)")
        print("=" * 85)
        print(f"{'Mô hình đánh giá':<40} | {'Macro F1 (%)':<18} | {'Accuracy (%)':<15}")
        print("-" * 85)
        if res_linear:
            print(f"{'Source Linear Probed (SSL Frozen)':<40} | {res_linear['macro_f1']:6.2f}%            | {res_linear['acc']:6.2f}%")
        if res_finetune:
            print(f"{'Source Full Fine-Tuned (Unfrozen)':<40} | {res_finetune['macro_f1']:6.2f}%            | {res_finetune['acc']:6.2f}%")
        print("=" * 85)

    finally:
        logger.close()


if __name__ == "__main__":
    main()