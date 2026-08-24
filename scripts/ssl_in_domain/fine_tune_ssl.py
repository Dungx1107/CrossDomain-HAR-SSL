"""
===============================================================================
SCRIPT RUNNER: scripts/ssl_in_domain/evaluate_few_label_ssl.py
MỤC ĐÍCH:
    - Đánh giá toàn diện mô hình SSL trên các tỷ lệ ít nhãn (0.1%, 0.5%, 1.0%).
    - Quét tự động qua 5 Random Seeds chuẩn: [42, 1337, 2024, 7, 99].
    - Hỗ trợ 2 chế độ:
        1. 'linear_probe' : Khóa Encoder, chỉ train Classifier Head.
        2. 'full_finetune': Mở khóa toàn bộ mạng (Differential Learning Rates).
    - Xuất bảng tổng hợp số liệu Mean ± Std ra file CSV để so sánh với Baseline.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

# 1. Định vị thư mục gốc dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset
from utils.sampling import create_few_label_subset, SEEDS
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.baseline.supervised_model import SupervisedHARModel
from training.evaluator import ModelEvaluator

# =============================================================================
# CẤU HÌNH THỰC NGHIỆM TỔNG QUAN
# =============================================================================
# EVAL_MODE = "full_finetune"         # Tùy chọn: 'linear_probe' hoặc 'full_finetune'
EVAL_MODE = "linear_probe"

RATIOS = [0.001, 0.005, 0.01]       # 0.1%, 0.5%, 1.0%
CLASS_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
EPOCHS = 100


class SimpleLogger:
    """Class ghi log song song ra Console và 1 file TXT duy nhất"""
    def __init__(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_txt_path = os.path.join(PROJECT_ROOT, "experiments", "logs", f"few_label_ssl_{EVAL_MODE}_results.txt")
    logger = SimpleLogger(log_txt_path)
    sys.stdout = logger

    try:
        print("=" * 80)
        print(f"🎯 BẮT ĐẦU THỰC NGHIỆM FEW-LABEL SSL (Chế độ: {EVAL_MODE.upper()})")
        print(f"🖥️ Thiết bị sử dụng: {device}")
        print(f"🌱 Danh sách Seeds thực nghiệm: {SEEDS}")
        print(f"📊 Các tỷ lệ nhãn: {[f'{r*100:g}%' for r in RATIOS]}")
        print("=" * 80)

        # 1. Nạp Dataset gốc
        full_train_dataset = MotionSenseDataset(
            data_dir=MotionSenseConfig.RAW_DATA_DIR,
            subjects_list=MotionSenseConfig.TRAIN_SUBJECTS,
            config=MotionSenseConfig
        )
        test_dataset = MotionSenseDataset(
            data_dir=MotionSenseConfig.RAW_DATA_DIR,
            subjects_list=MotionSenseConfig.TEST_SUBJECTS,
            config=MotionSenseConfig
        )

        test_loader = DataLoader(test_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=False)
        pretrained_weights_path = os.path.join(PROJECT_ROOT, "checkpoints", "tstcc_encoder_pretrained_motionsense.pt")

        if not os.path.exists(pretrained_weights_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file trọng số SSL tại: {pretrained_weights_path}")

        results_table = []

        # 2. Vòng lặp quét qua từng Tỷ lệ nhãn và từng Seed
        for ratio in RATIOS:
            ratio_percent = f"{ratio * 100:g}%"
            f1_scores = []
            acc_scores = []

            print("\n" + "#" * 80)
            print(f"🔥 THỰC NGHIỆM TỶ LỆ NHÃN: {ratio_percent} ({EVAL_MODE.upper()})")
            print("#" * 80)

            for seed in SEEDS:
                print(f"\n--- [Ratio: {ratio_percent} | Seed: {seed}] ---")

                # Cắt mẫu Train theo seed
                train_subset, _ = create_few_label_subset(full_train_dataset, ratio=ratio, seed=seed)
                train_loader = DataLoader(
                    train_subset,
                    batch_size=min(len(train_subset), MotionSenseConfig.BATCH_SIZE),
                    shuffle=True,
                    drop_last=False
                )

                # Khởi tạo mô hình và nạp trọng số SSL mới nhất
                encoder_backbone = StandardSensorEncoder1D(
                    in_channels=MotionSenseConfig.IN_CHANNELS,
                    feature_dim=128
                )
                encoder_backbone.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

                classifier_head = ClassifierHead(
                    feature_dim=128,
                    num_classes=MotionSenseConfig.NUM_CLASSES
                )
                model = SupervisedHARModel(encoder=encoder_backbone, classifier=classifier_head).to(device)

                criterion = nn.CrossEntropyLoss()

                # Cấu hình Optimizer theo chế độ
                if EVAL_MODE == "linear_probe":
                    for param in model.encoder.parameters():
                        param.requires_grad = False
                    for param in model.classifier.parameters():
                        param.requires_grad = True
                    optimizer = Adam(model.classifier.parameters(), lr=5e-3, weight_decay=1e-4)
                else:  # full_finetune
                    for param in model.parameters():
                        param.requires_grad = True
                    optimizer = Adam([
                        {'params': model.encoder.parameters(), 'lr': 1e-4},
                        {'params': model.classifier.parameters(), 'lr': 1e-3}
                    ], weight_decay=1e-4)

                # Huấn luyện
                for epoch in range(1, EPOCHS + 1):
                    if EVAL_MODE == "linear_probe":
                        model.eval()
                        model.classifier.train()
                    else:
                        model.train()

                    for x_batch, y_batch in train_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        out = model(x_batch)
                        loss = criterion(out, y_batch)
                        loss.backward()
                        optimizer.step()

                # Đánh giá trên tập Test
                model.eval()
                evaluator = ModelEvaluator(class_names=CLASS_NAMES, device=device)
                eval_metrics = evaluator.evaluate(
                    model=model,
                    test_loader=test_loader,
                    plot_save_path=None,
                    title_prefix=f"SSL {ratio_percent} (Seed {seed})"
                )

                test_f1 = eval_metrics["macro_f1"] * 100
                test_acc = eval_metrics["accuracy"] * 100

                f1_scores.append(test_f1)
                acc_scores.append(test_acc)

            # 3. Tính toán Mean ± Std cho từng tỷ lệ
            mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)
            mean_acc, std_acc = np.mean(acc_scores), np.std(acc_scores)

            results_table.append({
                "Label_Ratio": ratio_percent,
                "Mean_Macro_F1": round(mean_f1, 2),
                "Std_Macro_F1": round(std_f1, 2),
                "Mean_Accuracy": round(mean_acc, 2),
                "Std_Accuracy": round(std_acc, 2),
                "Individual_F1_Scores": [round(s, 2) for s in f1_scores]
            })

        # 4. Lưu và in Bảng tổng hợp số liệu
        summary_df = pd.DataFrame(results_table)
        summary_path = os.path.join(PROJECT_ROOT, "experiments", "logs", f"few_label_ssl_{EVAL_MODE}_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 80)
        print(f"{f'BẢNG TỔNG HỢP FEW-LABEL SSL ({EVAL_MODE.upper()})':^80}")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
        print(f"📄 Toàn bộ log chi tiết đã lưu tại: {log_txt_path}")
        print(f"📊 Bảng số liệu tổng hợp CSV đã lưu tại: {summary_path}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()