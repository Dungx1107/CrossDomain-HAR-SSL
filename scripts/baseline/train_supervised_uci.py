"""
===============================================================================
SCRIPT RUNNER: TRAIN + EVALUATE SUPERVISED BASELINE (UCI-HAR)
===============================================================================
Mục đích:
    - Huấn luyện mô hình Supervised 1D-CNN từ đầu trên tập Train của UCI-HAR (6 kênh, 5 lớp).
    - Chia tập Validation từ Train Set theo Subject (Subjects 27, 28, 29, 30 làm Val).
    - Tự động lưu và nạp Checkpoint có Val Macro F1 cao nhất để đánh giá trên Test Set.
    - Ghi toàn bộ logs, metrics.csv, learning curves và confusion matrix vào thư mục experiment.
===============================================================================
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Đảm bảo nhận diện thư mục gốc của dự án
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.uci_har_config import UCIHARConfig
from models.har_classifier import HARClassifier
from training.supervised_trainer import SupervisedTrainer
from training.evaluator import ModelEvaluator
from utils.logger import ExperimentTracker
from utils.complexity import measure_model_complexity, print_complexity_report


def get_uci_har_dataloaders(config):
    """
    Nạp dữ liệu UCI-HAR đã tiền xử lý và tạo DataLoader chuẩn (Train / Val / Test).
    Chia tập Train theo Subject để đảm bảo tính khách quan (Subject-wise Split).
    """
    if not os.path.exists(config.PROCESSED_TRAIN_PATH) or not os.path.exists(config.PROCESSED_TEST_PATH):
        raise FileNotFoundError("❌ Chưa tìm thấy file dữ liệu uci_har train.pt hoặc test.pt trong data/processed/uci_har/")

    # 1. Nạp dữ liệu đã xử lý
    train_data = torch.load(config.PROCESSED_TRAIN_PATH, map_location="cpu", weights_only=True)
    test_data = torch.load(config.PROCESSED_TEST_PATH, map_location="cpu", weights_only=True)

    x_train_raw = train_data["samples"]
    y_train_raw = train_data["labels"]
    sub_train = train_data["subjects"]

    x_test = test_data["samples"]
    y_test = test_data["labels"]

    # 2. Chia tập Train thành Train / Val theo Subject (Dùng 4 subjects cuối của train set làm Val)
    # Tập Train UCI gồm các subjects: 1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30
    val_subjects = [27, 28, 29, 30]
    val_mask = np.isin(sub_train.numpy(), val_subjects)
    train_mask = ~val_mask

    x_train, y_train = x_train_raw[train_mask], y_train_raw[train_mask]
    x_val, y_val = x_train_raw[val_mask], y_train_raw[val_mask]

    print(f"📊 Phân chia dữ liệu UCI-HAR:")
    print(f"   - Train samples : {x_train.shape[0]} | Shape: {x_train.shape}")
    print(f"   - Val samples   : {x_val.shape[0]}   | Shape: {x_val.shape}")
    print(f"   - Test samples  : {x_test.shape[0]}  | Shape: {x_test.shape}")

    # 3. Tạo DataLoaders
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đặt seed cố định để đảm bảo tính tái lập (Reproducibility)
    torch.manual_seed(UCIHARConfig.SEED)
    np.random.seed(UCIHARConfig.SEED)

    # 1. Khởi tạo Tracker duy nhất lưu logs và kết quả vào experiments/
    tracker = ExperimentTracker(
        exp_name="baseline_uci_cnn1d",
        config_obj=UCIHARConfig,
        notes=f"Supervised 1D-CNN Baseline trên UCI-HAR ({UCIHARConfig.IN_CHANNELS} kênh, {UCIHARConfig.NUM_CLASSES} lớp)"
    )

    checkpoint_path = os.path.join(tracker.log_dir, "best_model.pt")

    try:
        print(f"🖥️ Thiết bị sử dụng: {device}")

        # 2. Nạp Data Loaders của UCI-HAR
        train_loader, val_loader, test_loader = get_uci_har_dataloaders(UCIHARConfig)

        # 3. Khởi tạo HARClassifier (6 kênh đầu vào, 5 lớp phân loại)
        model = HARClassifier(
            in_channels=UCIHARConfig.IN_CHANNELS,
            num_classes=UCIHARConfig.NUM_CLASSES,
            feature_dim=UCIHARConfig.FEATURE_DIM
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(),
            lr=UCIHARConfig.LEARNING_RATE,
            weight_decay=UCIHARConfig.WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=UCIHARConfig.SCHEDULER_FACTOR,
            patience=UCIHARConfig.SCHEDULER_PATIENCE,
            min_lr=UCIHARConfig.SCHEDULER_MIN_LR
        )

        # 4. Đo độ phức tạp tính toán (FLOPs / Params) & ghi vào config.json
        input_sample_shape = (1, UCIHARConfig.IN_CHANNELS, UCIHARConfig.SEQUENCE_LENGTH)
        complexity_info = measure_model_complexity(model, input_size=input_sample_shape, device=device)
        print_complexity_report(complexity_info)
        tracker.log_complexity(complexity_info)

        # 5. HUẤN LUYỆN MÔ HÌNH (TRAINING PHASE)
        trainer = SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            tracker=tracker,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler
        )
        trainer.fit(train_loader, val_loader, epochs=UCIHARConfig.EPOCHS)

        # 6. ĐÁNH GIÁ TRÊN TẬP TEST CỦA UCI-HAR (EVALUATION PHASE)
        print("\n" + "#" * 85)
        print(f"{'BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP CỦA UCI-HAR (5 LỚP)':^85}")
        print("#" * 85)

        # Nạp lại trọng số tốt nhất đã lưu
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

        # Lưu Confusion Matrix vào thư mục plots của Experiment
        evaluator = ModelEvaluator(class_names=UCIHARConfig.CLASS_NAMES, device=device)
        cm_save_path = os.path.join(tracker.plot_dir, "confusion_matrix.png")

        evaluator.evaluate(
            model=model,
            test_loader=test_loader,
            plot_save_path=cm_save_path,
            title_prefix="UCI-HAR Supervised Baseline Test Set"
        )

    finally:
        tracker.close()


if __name__ == "__main__":
    main()