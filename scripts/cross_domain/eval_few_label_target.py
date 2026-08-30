"""
Few shot : encoder ssl + 1 phần dữ liệu trên uci
cập nhật cả encoder và head
"""

import os
import numpy as np
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader

# Tìm đường dẫn đến thư mục chứa file hiện tại
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.uci_har_config import UCIHARConfig
from utils.logger import SimpleLogger
from utils.sampling import sample_subset_by_ratio
from training.finetune_trainer import train_and_eval_finetune

# Cấu hình đường dẫn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_TEST_PATH = UCIHARConfig.PROCESSED_TEST_PATH
TRAIN_TEST_PATH = UCIHARConfig.PROCESSED_TRAIN_PATH

TSTCC_ENCODER_PRETRAINED_MOTIONSENSE = os.path.join(PROJECT_ROOT, "checkpoints/tstcc_encoder_pretrained_motionsense.pt")
TSTCC_ENCODER_PRETRAINED_UCIHAR = os.path.join(PROJECT_ROOT, "checkpoints/tstcc_encoder_pretrained_ucihar.pt")

REPORT_DIR = os.path.join(PROJECT_ROOT, "document", "0_reports", "6_tstcc_few_shot")
# LOG_FILE_PATH = os.path.join(REPORT_DIR, "tstcc_few_shot_from_motionsense_to_uci.txt")
LOG_FILE_PATH = os.path.join(REPORT_DIR, "tstcc_few_shot_from_uci_to_motionsense.txt")

SEEDS = [42, 123, 456, 789, 2024]
LABEL_FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    logger = SimpleLogger(LOG_FILE_PATH)
    sys.stdout = logger

    print("=" * 85)
    # print("Đánh giá TSTCC SSL Few-shot từ MotionSense sang UCI-HAR")
    print("Đánh giá TSTCC SSL Few-shot từ UCI-HAR sang MOTIONSENSE")
    print(f"Thiết bị sử dụng: {DEVICE}")
    # print(f"Checkpoint nguồn: {TSTCC_ENCODER_PRETRAINED_MOTIONSENSE}")
    print(f"Checkpoint nguồn: {TSTCC_ENCODER_PRETRAINED_MOTIONSENSE}")
    print(f"File lưu log: {LOG_FILE_PATH}")
    print("=" * 85)

    try:
        # 1. Nạp dữ liệu Test UCI-HAR
        if not os.path.exists(TARGET_TEST_PATH):
            raise FileNotFoundError(f"Không tìm thấy file: {TARGET_TEST_PATH}")

        test_data = torch.load(TARGET_TEST_PATH, map_location=DEVICE, weights_only=True)
        x_test, y_test = test_data["samples"], test_data["labels"]
        print(f"Đã nạp tập Test UCI-HAR: {len(x_test)} mẫu | Shape: {x_test.shape}")

        # 2. Nạp dữ liệu Train UCI-HAR
        if not os.path.exists(TRAIN_TEST_PATH):
            raise FileNotFoundError(f"Không tìm thấy file: {TRAIN_TEST_PATH}")

        train_data = torch.load(TRAIN_TEST_PATH, map_location=DEVICE, weights_only=True)
        x_train, y_train = train_data["samples"], train_data["labels"]
        print(f"Đã nạp tập Train UCI-HAR: {len(x_train)} mẫu | Shape: {x_train.shape}")

        results_summary = {}

        # 3. Vòng lặp qua các tỷ lệ nhãn và seeds
        for fraction in LABEL_FRACTIONS:
            f1_runs = []
            acc_runs = []
            n_samples = int(len(x_train) * fraction) if fraction < 1.0 else len(x_train)
            print(f"\n▶️ ĐANG THỬ NGHIỆM TỶ LỆ: {fraction * 100:.1f}% (~{n_samples} mẫu)")

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Lấy mẫu phân tầng từ tập train UCI
                x_train_sub, y_train_sub = sample_subset_by_ratio(
                    x_train,
                    y_train,
                    fraction=fraction,
                    seed=seed
                )

                # Chạy Fine-tuning (truyền x_test, y_test làm validation tạm thời)
                acc, f1, _, _ = train_and_eval_finetune(
                    x_train=x_train_sub,
                    y_train=y_train_sub,
                    x_val=x_test,
                    y_val=y_test,
                    x_test=x_test,
                    y_test=y_test,
                    num_classes=UCIHARConfig.NUM_CLASSES,
                    in_channels=UCIHARConfig.IN_CHANNELS,
                    encoder_checkpoint_path=TSTCC_ENCODER_PRETRAINED_MOTIONSENSE,
                    device=DEVICE
                )

                acc_runs.append(acc)
                f1_runs.append(f1)
                print(f"   - Seed {seed:4d} | Samples: {len(x_train_sub):4d} | Test Acc: {acc:.2f}% | Test Macro F1: {f1:.2f}%")

            mean_f1, std_f1 = np.mean(f1_runs), np.std(f1_runs)
            mean_acc, std_acc = np.mean(acc_runs), np.std(acc_runs)
            results_summary[fraction] = (mean_f1, std_f1, mean_acc, std_acc)
            print(f"👉 KẾT QUẢ TỶ LỆ {fraction * 100:.1f}%: Macro F1 = {mean_f1:.2f} ± {std_f1:.2f}% | Acc = {mean_acc:.2f} ± {std_acc:.2f}%")

        # 4. In bảng tổng hợp cuối cùng
        print("\n" + "=" * 85)
        print("🏆 BẢNG TỔNG HỢP HIỆU NĂNG FEW-SHOT ADAPTATION (UCI-HAR)")
        print("=" * 85)
        print(f"{'Label %':<12} | {'Mẫu (Samples)':<15} | {'Macro F1 (%)':<22} | {'Accuracy (%)':<20}")
        print("-" * 85)
        for frac, (m_f1, s_f1, m_acc, s_acc) in results_summary.items():
            n_samples = int(len(x_train) * frac) if frac < 1.0 else len(x_train)
            print(f"{frac * 100:<10.1f}% | {n_samples:<15d} | {m_f1:6.2f} ± {s_f1:5.2f}%      | {m_acc:6.2f} ± {s_acc:5.2f}%")
        print("=" * 85)

    finally:
        logger.close()

if __name__ == "__main__":
    main()