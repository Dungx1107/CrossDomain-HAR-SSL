"""
===============================================================================
EVALUATION PIPELINE: CROSS-DOMAIN FEW-LABEL FINE-TUNING (TS-TCC SSL)
===============================================================================
Mục đích:
    - Nhận vào Source Domain (nơi pretrain SSL) và Target Domain (nơi fine-tune ít nhãn).
    - Tự động nạp dữ liệu Train/Test của miền đích.
    - Lấy mẫu phân tầng (Stratified Sampling) theo các mức tỷ lệ và Seeds.
    - Chạy Fine-tuning toàn bộ mạng với Layer-wise Learning Rate.
    - Xuất log và bảng tổng kết hiệu năng khoa học.
===============================================================================
"""

import os
import sys
import numpy as np
import torch

# Thiết lập đường dẫn thư mục gốc của dự án
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from utils.logger import SimpleLogger
from utils.sampling import sample_subset_by_ratio
from training.finetune_trainer import train_and_eval_finetune

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456, 789, 2024]
LABEL_FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]

# Từ điển ánh xạ cấu hình của từng bộ dữ liệu
DOMAIN_CONFIGS = {
    "motionsense": {
        "train_path": getattr(MotionSenseConfig, "PROCESSED_TRAIN_PATH", None),
        "test_path": getattr(MotionSenseConfig, "PROCESSED_TEST_PATH", None),
        "num_classes": MotionSenseConfig.NUM_CLASSES,
        "in_channels": MotionSenseConfig.IN_CHANNELS,
    },
    "uci_har": {
        "train_path": UCIHARConfig.PROCESSED_TRAIN_PATH,
        "test_path": UCIHARConfig.PROCESSED_TEST_PATH,
        "num_classes": UCIHARConfig.NUM_CLASSES,
        "in_channels": UCIHARConfig.IN_CHANNELS,
    }
}


def evaluate_cross_domain_few_shot(
    source_name: str,
    target_name: str,
    seeds=SEEDS,
    label_fractions=LABEL_FRACTIONS,
    device=DEVICE
):
    """
    Thực hiện đánh giá chuyển giao Few-shot từ miền nguồn (Source) sang miền đích (Target).
    """
    source_key = source_name.lower().replace("-", "_")
    target_key = target_name.lower().replace("-", "_")

    if target_key not in DOMAIN_CONFIGS:
        raise ValueError(f"Chưa hỗ trợ cấu hình cho Target Domain: {target_name}")

    target_cfg = DOMAIN_CONFIGS[target_key]
    train_path = target_cfg["train_path"]
    test_path = target_cfg["test_path"]
    num_classes = target_cfg["num_classes"]
    in_channels = target_cfg["in_channels"]

    # Thiết lập đường dẫn Checkpoint nguồn và File Log
    checkpoint_ssl_path = os.path.join(
        PROJECT_ROOT, "checkpoints", "ssl", f"tstcc_encoder_pretrained_{source_key}.pt"
    )
    # Tương thích ngược nếu checkpoint lưu ở thư mục gốc checkpoints/
    if not os.path.exists(checkpoint_ssl_path):
        fallback_path = os.path.join(PROJECT_ROOT, "checkpoints", f"tstcc_encoder_pretrained_{source_key}.pt")
        if os.path.exists(fallback_path):
            checkpoint_ssl_path = fallback_path

    report_dir = os.path.join(PROJECT_ROOT, "document", "0_reports", "6_tstcc_few_shot")
    os.makedirs(report_dir, exist_ok=True)
    log_file_path = os.path.join(report_dir, f"tstcc_few_shot_from_{source_key}_to_{target_key}.txt")

    logger = SimpleLogger(log_file_path)
    sys.stdout = logger

    print("=" * 85)
    print(f"🚀 ĐÁNH GIÁ CHUYỂN GIAO FEW-SHOT: {source_name.upper()} ➔ {target_name.upper()}")
    print(f"   - Thiết bị sử dụng: {device}")
    print(f"   - Checkpoint SSL nguồn: {checkpoint_ssl_path}")
    print(f"   - File lưu log: {log_file_path}")
    print("=" * 85)

    try:
        # 1. Nạp dữ liệu Test của Target Domain
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Không tìm thấy file Test tại: {test_path}")

        test_data = torch.load(test_path, map_location=device, weights_only=True)
        x_test, y_test = test_data["samples"], test_data["labels"]
        print(f"✅ Đã nạp tập Test {target_name.upper()}: {len(x_test)} mẫu | Shape: {x_test.shape}")

        # 2. Nạp dữ liệu Train của Target Domain
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Không tìm thấy file Train tại: {train_path}")

        train_data = torch.load(train_path, map_location=device, weights_only=True)
        x_train, y_train = train_data["samples"], train_data["labels"]
        print(f"✅ Đã nạp tập Train {target_name.upper()}: {len(x_train)} mẫu | Shape: {x_train.shape}")

        results_summary = {}

        # 3. Vòng lặp qua từng tỷ lệ nhãn và từng Random Seed
        for fraction in label_fractions:
            f1_runs = []
            acc_runs = []
            n_samples = int(len(x_train) * fraction) if fraction < 1.0 else len(x_train)
            print(f"\n▶️ ĐANG THỬ NGHIỆM TỶ LỆ: {fraction * 100:.1f}% (~{n_samples} mẫu)")

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Lấy mẫu phân tầng từ tập Train Target
                x_train_sub, y_train_sub = sample_subset_by_ratio(
                    x_train,
                    y_train,
                    fraction=fraction,
                    seed=seed
                )

                # Chạy Fine-tuning (truyền x_test, y_test làm Validation tạm thời)
                acc, f1, _, _ = train_and_eval_finetune(
                    x_train=x_train_sub,
                    y_train=y_train_sub,
                    x_val=x_test,
                    y_val=y_test,
                    x_test=x_test,
                    y_test=y_test,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    encoder_checkpoint_path=checkpoint_ssl_path,
                    device=device
                )

                acc_runs.append(acc)
                f1_runs.append(f1)
                print(f"   - Seed {seed:4d} | Samples: {len(x_train_sub):4d} | Test Acc: {acc:.2f}% | Test Macro F1: {f1:.2f}%")

            mean_f1, std_f1 = np.mean(f1_runs), np.std(f1_runs)
            mean_acc, std_acc = np.mean(acc_runs), np.std(acc_runs)
            results_summary[fraction] = (mean_f1, std_f1, mean_acc, std_acc)
            print(f"👉 KẾT QUẢ TỶ LỆ {fraction * 100:.1f}%: Macro F1 = {mean_f1:.2f} ± {std_f1:.2f}% | Acc = {mean_acc:.2f} ± {std_acc:.2f}%")

        # 4. In bảng tổng hợp kết quả
        print("\n" + "=" * 85)
        print(f"🏆 BẢNG TỔNG HỢP HIỆU NĂNG FEW-SHOT: {source_name.upper()} ➔ {target_name.upper()}")
        print("=" * 85)
        print(f"{'Label %':<12} | {'Mẫu (Samples)':<15} | {'Macro F1 (%)':<22} | {'Accuracy (%)':<20}")
        print("-" * 85)
        for frac, (m_f1, s_f1, m_acc, s_acc) in results_summary.items():
            n_samples = int(len(x_train) * frac) if frac < 1.0 else len(x_train)
            print(f"{frac * 100:<10.1f}% | {n_samples:<15d} | {m_f1:6.2f} ± {s_f1:5.2f}%      | {m_acc:6.2f} ± {s_acc:5.2f}%")
        print("=" * 85 + "\n")

    finally:
        logger.close()


def main():
    # =========================================================================
    # KỊCH BẢN 1: MOTIONSENSE (Source SSL) ➔ UCI-HAR (Target Few-shot)
    # =========================================================================
    # evaluate_cross_domain_few_shot(
    #     source_name="motionsense",
    #     target_name="uci_har"
    # )

    # =========================================================================
    # KỊCH BẢN 2: UCI-HAR (Source SSL) ➔ MOTIONSENSE (Target Few-shot)
    # =========================================================================
    evaluate_cross_domain_few_shot(
        source_name="uci_har",
        target_name="motionsense"
    )


if __name__ == "__main__":
    main()