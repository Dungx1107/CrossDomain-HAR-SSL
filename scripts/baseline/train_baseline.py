"""
===============================================================================
SCRIPT RUNNER: scripts/baseline/train_baseline.py (TRAIN + EVALUATE)
===============================================================================
Mục đích:
    - Huấn luyện mô hình Supervised 1D-CNN trên MotionSense (Subjects 1-14 Train, 15-18 Val).
    - Tự động nạp Checkpoint tốt nhất và đánh giá ngay trên tập Test (Subjects 19-24).
    - Toàn bộ log, metrics, learning curves và confusion matrix được lưu vào ĐÚNG 1 THƯ MỤC.
===============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.loader import get_motionsense_dataloaders
from models.baseline.supervised_model import SupervisedHARModel
from training.supervised_trainer import SupervisedTrainer
from training.evaluator import ModelEvaluator
from utils.logger import ExperimentTracker

CLASS_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints/baseline_cnn1d_motionsense_best.pt")

    # 1. Khởi tạo Tracker duy nhất: Tạo 1 thư mục experiments/baseline_cnn1d_<timestamp>/
    tracker = ExperimentTracker(
        exp_name="baseline_cnn1d",
        config_obj=MotionSenseConfig,
        notes="Quy trình trọn gói Train + Test Baseline 1D-CNN (9 kênh)"
    )

    try:
        print(f"🖥️ Thiết bị sử dụng: {device}")

        # 2. Nạp toàn bộ Data Loaders (Train, Val, Test)
        train_loader, val_loader, test_loader = get_motionsense_dataloaders()

        # 3. Khởi tạo Model, Loss, Optimizer
        model = SupervisedHARModel(
            in_channels=MotionSenseConfig.IN_CHANNELS,
            num_classes=MotionSenseConfig.NUM_CLASSES
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=MotionSenseConfig.LEARNING_RATE, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        # ĐO ĐỘ PHỨC TẠP TÍNH TOÁN & GHI VÀO CONFIG.JSON
        input_sample_shape = (1, MotionSenseConfig.IN_CHANNELS, MotionSenseConfig.WINDOW_SIZE)
        complexity_info = measure_model_complexity(model, input_size=input_sample_shape, device=device)
        print_complexity_report(complexity_info)
        tracker.log_complexity(complexity_info)


        # 4. GIAI ĐOẠN 1: HUẤN LUYỆN (TRAINING)
        trainer = SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            tracker=tracker,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler
        )
        trainer.fit(train_loader, val_loader, epochs=MotionSenseConfig.EPOCHS)

        # 5. GIAI ĐOẠN 2: ĐÁNH GIÁ TRÊN TẬP TEST (EVALUATION TRÊN BEST MODEL)
        print("\n" + "#" * 85)
        print(f"{'BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (SUBJECTS 19-24)':^85}")
        print("#" * 85)

        # Nạp lại trọng số tốt nhất vừa lưu trong quá trình train
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Đánh giá và lưu thẳng confusion_matrix.png vào cùng thư mục tracker.run_dir
        evaluator = ModelEvaluator(class_names=CLASS_NAMES, device=device)
        cm_save_path = os.path.join(tracker.plot_dir, "confusion_matrix.png")

        evaluator.evaluate(
            model=model,
            test_loader=test_loader,
            plot_save_path=cm_save_path,
            title_prefix="Baseline 1D-CNN Test Set"
        )

    finally:
        tracker.close()  # Giải phóng an toàn luồng ghi log


if __name__ == "__main__":
    main()
