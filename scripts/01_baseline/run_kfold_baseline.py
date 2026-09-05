"""
===============================================================================
SCRIPT: K-FOLD CROSS-VALIDATION BASELINE (SUPERVISED)
===============================================================================
Mục đích:
    - Chạy K-Fold Cross-Validation cho một hoặc nhiều bộ dữ liệu được chỉ định trong List.
    - Chia fold phân tầng (StratifiedKFold) trên file dataset_all.pt.
    - Khởi tạo lại mô hình và bộ tối ưu hóa độc lập cho từng fold.
    - Lưu checkpoint, lịch sử huấn luyện CSV, ma trận nhầm lẫn cho từng fold.
    - Tổng hợp báo cáo thống kê Mean ± Std (Accuracy, Macro F1, Weighted F1) ra JSON.
===============================================================================
"""

import sys
import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from datasets.base_dataset import BaseHARDataset
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.har_classifier import HARClassifier
from engines.supervised.supervised_trainer import SupervisedTrainer
from engines.evaluation.evaluator import ModelEvaluator
from utils.complexity import measure_model_complexity, print_complexity_report


# =============================================================================
# KHỐI CẤU HÌNH THỰC NGHIỆM K-FOLD (CHỈNH TRỰC TIẾP TẠI ĐÂY RỒI BẤM RUN)
# =============================================================================
@dataclass
class KFoldExperimentConfig:
    # Truyền danh sách dataset cần chạy tại đây: ["motionsense", "uci_har"]
    datasets_to_run: List[str] = field(default_factory=lambda: ["motionsense", "uci_har"])
    n_splits: int = 5
    seed: int = 42
    feature_dim: int = 128
    dropout_rate: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    measure_complexity: bool = True
    device: Optional[str] = None  # None: tự nhận diện cuda/cpu


CONFIG = KFoldExperimentConfig(
    datasets_to_run=["motionsense", "uci_har"],  # Sau này có thêm dataset thì chỉ cần append vào đây
    n_splits=5,
    epochs=40,
    batch_size=64,
    learning_rate=1e-3,
    dropout_rate=0.2,
    measure_complexity=True
)


# =============================================================================


@dataclass
class DatasetMetadata:
    data_all_path: Path
    in_channels: int
    num_classes: int
    class_names: List[str]


def resolve_dataset_meta(dataset_name: str) -> DatasetMetadata:
    """Trích xuất thông số metadata và đường dẫn dataset_all.pt từ file Config."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "motionsense":
        return DatasetMetadata(
            data_all_path=Path(MotionSenseConfig.DATA_ALL_PATH),
            in_channels=int(MotionSenseConfig.IN_CHANNELS),
            num_classes=int(MotionSenseConfig.NUM_CLASSES),
            class_names=list(MotionSenseConfig.CLASS_NAMES)
        )
    elif dataset_name in ["uci_har", "ucihar"]:
        return DatasetMetadata(
            data_all_path=Path(UCIHARConfig.DATA_ALL_PATH),
            in_channels=int(UCIHARConfig.IN_CHANNELS),
            num_classes=int(UCIHARConfig.NUM_CLASSES),
            class_names=list(UCIHARConfig.CLASS_NAMES)
        )
    else:
        raise ValueError(f"❌ Bộ dữ liệu '{dataset_name}' chưa được hỗ trợ trong resolve_dataset_meta!")


def build_model(in_channels: int, num_classes: int, feature_dim: int, dropout_rate: float) -> HARClassifier:
    """Khởi tạo cấu trúc mô hình HARClassifier chuẩn."""
    encoder = StandardSensorEncoder1D(in_channels=in_channels, feature_dim=feature_dim)
    head = ClassifierHead(feature_dim=feature_dim, num_classes=num_classes, dropout_rate=dropout_rate)
    return HARClassifier(encoder=encoder, classifier=head)


def run_kfold_for_dataset(dataset_name: str, device: torch.device):
    """Quy trình huấn luyện và kiểm định chéo K-Fold hoàn chỉnh cho 1 dataset."""
    meta = resolve_dataset_meta(dataset_name)

    # 1. Khởi tạo cấu trúc thư mục
    base_kfold_dir = PROJECT_ROOT / "experiments" / "kfold_baseline" / dataset_name
    ckpt_dir = base_kfold_dir / "checkpoints"
    metrics_dir = base_kfold_dir / "metrics"
    plots_dir = base_kfold_dir / "plots"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"🌟 TIẾN HÀNH K-FOLD CROSS-VALIDATION ({CONFIG.n_splits} FOLDS): {dataset_name.upper()}")
    print(f"📂 File dữ liệu gốc : {meta.data_all_path}")
    print(f"📁 Thư mục xuất ra  : {base_kfold_dir}")
    print(f"📊 Kênh / Số lớp    : {meta.in_channels} Kênh / {meta.num_classes} Lớp")
    print("#" * 80)

    # 2. Nạp toàn bộ dữ liệu từ dataset_all.pt
    full_dataset = BaseHARDataset(meta.data_all_path, fraction=1.0)
    all_labels = full_dataset.labels.numpy()
    all_subjects = full_dataset.subjects.numpy()  # Trích xuất subject ID

    # Khởi tạo StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=CONFIG.n_splits)

    # 3. Đo độ phức tạp tính toán (FLOPs, Params, Latency) 1 lần đầu
    if CONFIG.measure_complexity:
        sample_model = build_model(meta.in_channels, meta.num_classes, CONFIG.feature_dim, CONFIG.dropout_rate).to(
            device)
        input_shape = (1, meta.in_channels, 128)
        complexity_info = measure_model_complexity(sample_model, input_size=input_shape, device=device)
        print_complexity_report(complexity_info)
        with open(metrics_dir / "complexity_report.json", "w", encoding="utf-8") as f:
            json.dump(complexity_info, f, indent=4)
        del sample_model

    # 4. Thiết lập StratifiedKFold
    skf = StratifiedKFold(n_splits=CONFIG.n_splits, shuffle=True, random_state=CONFIG.seed)

    fold_results = []

    # for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels), start=1):
    # Chia fold: gom theo groups=all_subjects
    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(np.zeros(len(all_labels)), all_labels, groups=all_subjects),
            start=1
    ):
        print("\n" + "=" * 65)
        print(f"🔄 FOLD [{fold:02d}/{CONFIG.n_splits:02d}] - {dataset_name.upper()}")
        print(f"   Train samples: {len(train_idx):,} | Val samples: {len(val_idx):,}")
        print("=" * 65)

        # Tạo DataLoader riêng cho từng Fold
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_sub,
            batch_size=CONFIG.batch_size,
            shuffle=True,
            drop_last=True if len(train_sub) >= CONFIG.batch_size else False
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=CONFIG.batch_size,
            shuffle=False,
            drop_last=False
        )

        # Khởi tạo mô hình và Optimizer hoàn toàn mới cho Fold này
        model = build_model(
            in_channels=meta.in_channels,
            num_classes=meta.num_classes,
            feature_dim=CONFIG.feature_dim,
            dropout_rate=CONFIG.dropout_rate
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        trainer = SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(ckpt_dir),
            scheduler=scheduler
        )

        fold_best_model_path = ckpt_dir / f"best_model_fold_{fold}.pt"
        fold_csv_path = metrics_dir / f"training_history_fold_{fold}.csv"

        # Vòng lặp huấn luyện từng Fold
        with open(fold_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "train_macro_f1", "val_loss", "val_acc", "val_macro_f1", "lr"])

            for epoch in range(1, CONFIG.epochs + 1):
                train_loss, train_acc, train_f1 = trainer.train_one_epoch(train_loader)
                val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

                current_lr = float(optimizer.param_groups[0]['lr'])
                if trainer.scheduler is not None:
                    trainer.scheduler.step(val_f1)

                # Lưu trọng số tốt nhất theo Macro F1 của Fold này
                if val_f1 > trainer.best_val_f1:
                    trainer.best_val_f1 = val_f1
                    trainer.best_model_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
                    torch.save(trainer.best_model_state, fold_best_model_path)

                writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, current_lr])

                if epoch % 10 == 0 or epoch == 1 or epoch == CONFIG.epochs:
                    print(
                        f"   Epoch [{epoch:02d}/{CONFIG.epochs:02d}] | "
                        f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% F1: {val_f1 * 100:.2f}%"
                    )

        # Nạp lại trọng số tốt nhất của Fold để đánh giá chính thức
        if trainer.best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in trainer.best_model_state.items()})

        evaluator = ModelEvaluator(class_names=meta.class_names, device=device)
        cm_save_path = str(plots_dir / f"confusion_matrix_fold_{fold}.png")
        eval_metrics = evaluator.evaluate(
            model=model,
            test_loader=val_loader,
            plot_save_path=cm_save_path,
            title_prefix=f"{dataset_name.upper()} Fold {fold}"
        )

        fold_results.append({
            "fold": fold,
            "best_val_macro_f1": float(trainer.best_val_f1 * 100),
            "accuracy": float(eval_metrics["accuracy"] * 100),
            "macro_f1": float(eval_metrics["macro_f1"] * 100),
            "weighted_f1": float(eval_metrics["weighted_f1"] * 100)
        })

    # 5. Tổng hợp thống kê Mean ± Std qua K Folds
    accs = [r["accuracy"] for r in fold_results]
    macro_f1s = [r["macro_f1"] for r in fold_results]
    weighted_f1s = [r["weighted_f1"] for r in fold_results]

    summary_stats = {
        "dataset": dataset_name,
        "n_splits": CONFIG.n_splits,
        "epochs_per_fold": CONFIG.epochs,
        "batch_size": CONFIG.batch_size,
        "metrics_summary": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s)),
            "weighted_f1_mean": float(np.mean(weighted_f1s)),
            "weighted_f1_std": float(np.std(weighted_f1s))
        },
        "fold_details": fold_results
    }

    summary_json_path = metrics_dir / "kfold_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=4)

    print("\n" + "=" * 70)
    print(f"🏁 KẾT QUẢ TỔNG HỢP {CONFIG.n_splits}-FOLD CHO {dataset_name.upper()}:")
    print(
        f"🎯 Test Accuracy  : {summary_stats['metrics_summary']['accuracy_mean']:.2f}% ± {summary_stats['metrics_summary']['accuracy_std']:.2f}%")
    print(
        f"🏆 Test Macro F1  : {summary_stats['metrics_summary']['macro_f1_mean']:.2f}% ± {summary_stats['metrics_summary']['macro_f1_std']:.2f}%")
    print(
        f"⚖️ Test Weighted F1: {summary_stats['metrics_summary']['weighted_f1_mean']:.2f}% ± {summary_stats['metrics_summary']['weighted_f1_std']:.2f}%")
    print(f"💾 Báo cáo chi tiết đã lưu tại: {summary_json_path}")
    print("=" * 70 + "\n")


def main():
    device = torch.device(CONFIG.device) if CONFIG.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 KHỞI ĐỘNG TIẾN TRÌNH K-FOLD CROSS-VALIDATION")
    print(f"📋 Danh sách tập dữ liệu sẽ chạy: {CONFIG.datasets_to_run}")
    print(f"💻 Thiết bị tính toán: {device}")

    for dataset_name in CONFIG.datasets_to_run:
        run_kfold_for_dataset(dataset_name=dataset_name, device=device)

    print("🎉 TẤT CẢ CÁC BỘ DỮ LIỆU ĐÃ HOÀN TẤT K-FOLD THÀNH CÔNG!")


if __name__ == "__main__":
    main()
