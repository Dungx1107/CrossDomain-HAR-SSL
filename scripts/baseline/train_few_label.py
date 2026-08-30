"""
===============================================================================
MỤC ĐÍCH:
    Thực nghiệm Few-Label Supervised Baseline trên MotionSense:
      - Tỷ lệ nhãn: 1%, 5%, 10% cho cả Train và Validation
      - Dùng trực tiếp module sampling: create_few_label_subset
      - 5 Random Seeds chuẩn: [42, 1337, 2024, 7, 99]
      - Đo lường và tổng hợp Macro F1 (Mean ± Std) trên tập Test độc lập.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset
from models.har_classifier import HARClassifier
from training.supervised_trainer import SupervisedTrainer
from training.evaluator import ModelEvaluator
from utils.logger import SimpleLogger
from utils.sampling import create_few_label_subset, SEEDS

CLASS_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
# RATIOS = [0.01, 0.05, 0.10]
RATIOS = [0.001, 0.005]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_txt_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "few_label_supervised_results.txt")
    logger = SimpleLogger(log_txt_path)
    sys.stdout = logger

    print("=" * 80)
    print(" BẮT ĐẦU THỰC NGHIỆM FEW-LABEL SUPERVISED BASELINE")
    print(f"🖥️ Thiết bị sử dụng: {device}")
    print(f"🌱 Danh sách Seeds thực nghiệm: {SEEDS}")
    print("=" * 80)

    # 1. Khởi tạo toàn bộ dữ liệu gốc
    full_train_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TRAIN_SUBJECTS,
        config=MotionSenseConfig
    )

    full_val_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.VAL_SUBJECTS,
        config=MotionSenseConfig
    )

    test_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TEST_SUBJECTS,
        config=MotionSenseConfig
    )

    # Test Loader giữ nguyên 100% không gian kiểm thử
    test_loader = DataLoader(test_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=False)

    results_table = []

    # 2. Vòng lặp quét qua từng tỷ lệ (1%, 5%, 10%) và từng Seed
    for ratio in RATIOS:
        ratio_percent = f"{ratio * 100:g}%"
        f1_scores = []
        acc_scores = []

        print("\n" + "=" * 80)
        print(f"🚀 BẮT ĐẦU THỰC NGHIỆM TỶ LỆ NHÃN: {ratio_percent}% (Train & Val cùng lấy {ratio * 100:.1f}%)")
        print("=" * 80)

        for seed in SEEDS:
            train_subset, _ = create_few_label_subset(full_train_dataset, ratio=ratio, seed=seed)
            val_subset, _ = create_few_label_subset(full_val_dataset, ratio=ratio, seed=seed)

            # Đóng gói DataLoader
            train_loader = DataLoader(
                train_subset,
                batch_size=min(len(train_subset), MotionSenseConfig.BATCH_SIZE),
                shuffle=True,
                drop_last=False
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=min(len(val_subset), MotionSenseConfig.BATCH_SIZE),
                shuffle=False,
                drop_last=False
            )

            temp_checkpoint = os.path.join(PROJECT_ROOT, "checkpoints", "temp_few_label_model.pt")

            # Khởi tạo mô hình
            model = HARClassifier(
                in_channels=MotionSenseConfig.IN_CHANNELS,
                num_classes=MotionSenseConfig.NUM_CLASSES
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=MotionSenseConfig.LEARNING_RATE, weight_decay=1e-4)

            # Huấn luyện
            trainer = SupervisedTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                tracker=None,
                checkpoint_path=temp_checkpoint,
                scheduler=None
            )
            trainer.fit(train_loader, val_loader, epochs=MotionSenseConfig.EPOCHS)

            # Đánh giá trên tập Test độc lập (Hold-out Test Set)
            if os.path.exists(temp_checkpoint):
                model.load_state_dict(torch.load(temp_checkpoint, map_location=device))
                os.remove(temp_checkpoint)

            evaluator = ModelEvaluator(class_names=CLASS_NAMES, device=device)
            eval_metrics = evaluator.evaluate(
                model=model,
                test_loader=test_loader,
                plot_save_path=None,
                title_prefix=f"Supervised {ratio_percent}% (Seed {seed})"
            )

            test_f1 = eval_metrics["macro_f1"] * 100
            test_acc = eval_metrics["accuracy"] * 100

            f1_scores.append(test_f1)
            acc_scores.append(test_acc)

        # 3. Tính Mean ± Std cho từng tỷ lệ
        mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
        mean_acc, std_acc = np.mean(acc_scores), np.std(acc_scores)

        results_table.append({
            "Label_Ratio": f"{ratio_percent}",
            "Mean_Macro_F1": round(mean_f1, 2),
            "Std_Macro_F1": round(std_f1, 2),
            "Mean_Accuracy": round(mean_acc, 2),
            "Std_Accuracy": round(std_acc, 2),
            "Individual_F1_Scores": [round(s, 2) for s in f1_scores]
        })

    # 4. Lưu và in Bảng báo cáo tổng kết
    summary_df = pd.DataFrame(results_table)
    summary_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "few_label_supervised_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print(f"{'BẢNG TỔNG HỢP THỰC NGHIỆM FEW-LABEL SUPERVISED BASELINE':^80}")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"📄 Toàn bộ log chi tiết đã lưu tại: {log_txt_path}")
    print(f"📊 Bảng số liệu tổng hợp CSV đã lưu tại: {summary_path}")


if __name__ == "__main__":
    main()
