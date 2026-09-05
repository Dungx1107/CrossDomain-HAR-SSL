"""
===============================================================================
SCRIPT: TRAIN & EVALUATE BASELINE (SUPERVISED)
===============================================================================
Mục đích:
    - Huấn luyện Supervised Baseline từ đầu (Train from Scratch).
    - Tự động chuyển đổi cấu hình giữa MotionSense và UCI-HAR.
    - Lưu lịch sử training ra training_history.csv để vẽ learning curves.
    - Đo lường độ phức tạp tính toán (Params, FLOPs, Latency).
    - Tự động nạp Checkpoint tốt nhất và đánh giá độc lập qua ModelEvaluator.
===============================================================================
"""

import sys
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from datasets.base_dataset import get_har_dataloaders
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.har_classifier import HARClassifier
from engines.supervised.supervised_trainer import SupervisedTrainer
from engines.evaluation.evaluator import ModelEvaluator
from utils.complexity import measure_model_complexity, print_complexity_report


# =============================================================================
# KHỐI CẤU HÌNH THỰC NGHIỆM (CHỈNH TRỰC TIẾP TẠI ĐÂY RỒI BẤM RUN)
# =============================================================================
@dataclass
class ExperimentConfig:
    dataset: str = "motionsense"  # 'motionsense' hoặc 'uci_har'
    encoder_type: str = "1dcnn"  # '1dcnn'
    feature_dim: int = 128
    dropout_rate: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    measure_complexity: bool = True
    device: Optional[str] = None  # None: tự động nhận diện, hoặc "cuda", "cpu"


CONFIG = ExperimentConfig(
    dataset="uci_har",  # Đổi sang "uci_har" khi muốn chạy UCI-HAR
    epochs=40,
    batch_size=64,
    learning_rate=1e-3,
    dropout_rate=0.2,
    measure_complexity=True
)


@dataclass
class DatasetMetadata:
    data_dir: Path
    in_channels: int
    num_classes: int
    class_names: list


def resolve_dataset_meta(dataset_name: str) -> DatasetMetadata:
    """Trích xuất thông số metadata và thư mục dữ liệu chuẩn từ file Config."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "motionsense":
        return DatasetMetadata(
            data_dir=Path(MotionSenseConfig.PROCESSED_DIR),
            in_channels=int(MotionSenseConfig.IN_CHANNELS),
            num_classes=int(MotionSenseConfig.NUM_CLASSES),
            class_names=list(MotionSenseConfig.CLASS_NAMES)
        )
    elif dataset_name in ["uci_har", "ucihar"]:
        return DatasetMetadata(
            data_dir=Path(UCIHARConfig.DATA_DIR),
            in_channels=int(UCIHARConfig.IN_CHANNELS),
            num_classes=int(UCIHARConfig.NUM_CLASSES),
            class_names=list(UCIHARConfig.CLASS_NAMES)
        )
    else:
        raise ValueError(f"❌ Bộ dữ liệu '{dataset_name}' chưa được hỗ trợ!")


def build_model(in_channels: int, num_classes: int, feature_dim: int, dropout_rate: float) -> HARClassifier:
    """Khởi tạo mô hình HARClassifier với Encoder và Head tùy biến."""
    encoder = StandardSensorEncoder1D(in_channels=in_channels, feature_dim=feature_dim)
    head = ClassifierHead(feature_dim=feature_dim, num_classes=num_classes, dropout_rate=dropout_rate)
    return HARClassifier(encoder=encoder, classifier=head)


def main():
    device = torch.device(CONFIG.device) if CONFIG.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = CONFIG.dataset.lower().strip()

    meta = resolve_dataset_meta(dataset_name)

    # 1. Khởi tạo cây thư mục kết quả
    base_exp_dir = PROJECT_ROOT / "experiments" / "baseline" / dataset_name
    ckpt_dir = base_exp_dir / "checkpoint"
    metrics_dir = base_exp_dir / "metrics"
    plots_dir = base_exp_dir / "plots"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN SUPERVISED BASELINE: {dataset_name.upper()}")
    print(f"💻 Thiết bị thực thi : {device}")
    print(f"📂 Thư mục dữ liệu   : {meta.data_dir}")
    print(f"📂 Thư mục kết quả   : {base_exp_dir}")
    print(f"📊 Kênh vào / Số lớp : {meta.in_channels} Kênh / {meta.num_classes} Lớp")
    print("=" * 80)

    # 2. Nạp DataLoaders từ BaseHARDataset
    train_loader, val_loader, test_loader = get_har_dataloaders(
        data_dir=meta.data_dir,
        batch_size=CONFIG.batch_size
    )

    # 3. Khởi tạo mô hình
    model = build_model(
        in_channels=meta.in_channels,
        num_classes=meta.num_classes,
        feature_dim=CONFIG.feature_dim,
        dropout_rate=CONFIG.dropout_rate
    ).to(device)

    # 4. Đo độ phức tạp tính toán (FLOPs, Params, Latency)
    if CONFIG.measure_complexity:
        input_shape = (1, meta.in_channels, 128)
        complexity_info = measure_model_complexity(model, input_size=input_shape, device=device)
        print_complexity_report(complexity_info)
        with open(metrics_dir / "complexity_report.json", "w", encoding="utf-8") as f:
            json.dump(complexity_info, f, indent=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 5. Khởi tạo Engine SupervisedTrainer
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(ckpt_dir),
        scheduler=scheduler
    )

    # 6. Huấn luyện & ghi log lịch sử ra CSV
    csv_file_path = metrics_dir / "training_history.csv"
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "train_macro_f1", "val_loss", "val_acc", "val_macro_f1", "lr"])

        print(f"🚀 Bắt đầu quá trình huấn luyện ({CONFIG.epochs} epochs)...")
        for epoch in range(1, CONFIG.epochs + 1):
            train_loss, train_acc, train_f1 = trainer.train_one_epoch(train_loader)
            val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

            current_lr = float(optimizer.param_groups[0]['lr'])
            if trainer.scheduler is not None:
                trainer.scheduler.step(val_f1)

            # Lưu checkpoint tốt nhất theo Macro F1 tập Val
            if val_f1 > trainer.best_val_f1:
                trainer.best_val_f1 = val_f1
                trainer.best_model_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
                torch.save(trainer.best_model_state, ckpt_dir / "best_baseline_model.pt")

            writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, current_lr])

            if epoch % 5 == 0 or epoch == 1 or epoch == CONFIG.epochs:
                print(
                    f"Epoch [{epoch:03d}/{CONFIG.epochs:03d}] | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% F1: {val_f1 * 100:.2f}%"
                )

    print(f"\n📊 Lịch sử huấn luyện đã lưu tại: {csv_file_path}")

    # 7. Nạp lại trọng số tối ưu nhất để đánh giá trên tập Test
    if trainer.best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in trainer.best_model_state.items()})

    evaluator = ModelEvaluator(class_names=meta.class_names, device=device)
    cm_path = str(plots_dir / "confusion_matrix.png")
    test_metrics = evaluator.evaluate(
        model=model,
        test_loader=test_loader,
        plot_save_path=cm_path,
        title_prefix=f"Supervised {dataset_name.upper()}"
    )

    # 8. Lưu báo cáo tổng kết
    summary_report = {
        "dataset": dataset_name,
        "encoder_type": CONFIG.encoder_type,
        "best_val_macro_f1": float(trainer.best_val_f1 * 100),
        "test_accuracy": float(test_metrics["accuracy"] * 100),
        "test_macro_f1": float(test_metrics["macro_f1"] * 100),
        "test_weighted_f1": float(test_metrics["weighted_f1"] * 100)
    }
    with open(metrics_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    print(f"✅ Hoàn tất! Báo cáo nghiệm thu lưu tại: {base_exp_dir}")


if __name__ == "__main__":
    main()
