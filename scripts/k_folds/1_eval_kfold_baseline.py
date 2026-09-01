import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Tự động xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import chuẩn xác theo cấu trúc dự án
from models.har_classifier import HARClassifier
from datasets.kfold_splitter import get_kfold_loaders
from training.supervised_trainer import SupervisedTrainer
from training.evaluator import ModelEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_fold_engine(model, train_loader, optimizer, criterion, epochs):
    """
    Tận dụng SupervisedTrainer để thực thi vòng lặp huấn luyện thuần túy cho từng Fold.
    """
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        tracker=None,           # Không tạo log file lẻ tẻ cho từng fold
        checkpoint_path="",     # Không ghi checkpoint đè lên đĩa
        scheduler=None
    )

    for epoch in range(1, epochs + 1):
        trainer.train_one_epoch(train_loader)


def run_kfold_supervised(dataset_name="motionsense", k=5, epochs=30, lr=1e-3, batch_size=64):
    """
    Điều phối luồng chạy K-Fold: chia dữ liệu, huấn luyện và tổng hợp kết quả.
    """
    fold_loaders, in_channels, num_classes = get_kfold_loaders(
        dataset_name=dataset_name,
        k=k,
        batch_size=batch_size
    )

    evaluator = ModelEvaluator(device=device)
    acc_list, f1_list = [], []

    print(f"\n---> [Dataset: {dataset_name.upper()} | Channels: {in_channels} | Classes: {num_classes} | K = {k} | Epochs = {epochs} | Device = {device}]")

    for fold_idx, (train_loader, test_loader) in enumerate(fold_loaders, 1):
        # 1. Khởi tạo mô hình mới (From Scratch) đúng in_channels và num_classes từ dataset
        model = HARClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_dim=128
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 2. Huấn luyện qua Engine SupervisedTrainer
        train_fold_engine(model, train_loader, optimizer, criterion, epochs)

        # 3. Đánh giá qua Engine ModelEvaluator
        metrics = evaluator.evaluate(model=model, test_loader=test_loader, title_prefix=f"Fold {fold_idx}/{k}")

        # Thang đo trong ModelEvaluator là [0, 1] nên quy đổi sang %
        acc = metrics["accuracy"] * 100
        f1 = metrics["macro_f1"] * 100

        acc_list.append(acc)
        f1_list.append(f1)
        print(f"  • Kết quả Fold {fold_idx}/{k} -> Accuracy: {acc:.2f}% | Macro F1: {f1:.2f}%")

    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_f1, std_f1 = np.mean(f1_list), np.std(f1_list)

    print(f"  ⭐ TỔNG KẾT K={k} -> Acc: {mean_acc:.2f} ± {std_acc:.2f}% | F1: {mean_f1:.2f} ± {std_f1:.2f}%")
    return mean_acc, std_acc, mean_f1, std_f1


def main():
    parser = argparse.ArgumentParser(description="K-Fold Supervised Baseline Evaluation")
    parser.add_argument("--dataset", type=str, default="motionsense", help="Tên dataset (motionsense, uci_har,...)")
    parser.add_argument("--k_list", nargs="+", type=int, default=[3, 5, 7, 10], help="Danh sách K cần chạy")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch cho mỗi fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Chạy thử nghiệm kiểm tra luồng, KHÔNG lưu file report")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "document" / "0_reports" / "7_kfold_evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"baseline_{args.dataset}_kfold_report.txt"

    print("=" * 75)
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ SUPERVISED BASELINE K-FOLDS ({args.dataset.upper()})")
    print("=" * 75)

    summary_rows = []
    for k in args.k_list:
        m_acc, s_acc, m_f1, s_f1 = run_kfold_supervised(
            dataset_name=args.dataset,
            k=k,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
        row = f"K = {k:2d} | Accuracy: {m_acc:.2f} ± {s_acc:.2f}% | Macro F1: {m_f1:.2f} ± {s_f1:.2f}%"
        summary_rows.append(row)

    if not args.dry_run:
        report_dir = PROJECT_ROOT / "document" / "0_reports" / "7_kfold_evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"baseline_{args.dataset}_kfold_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"BÁO CÁO K-FOLD SUPERVISED BASELINE - TẬP DỮ LIỆU: {args.dataset.upper()}\n")
            f.write(f"Cấu hình: Epochs={args.epochs}, LR={args.lr}, Batch Size={args.batch_size}\n")
            f.write("=" * 75 + "\n")
            f.write("\n".join(summary_rows) + "\n")
            f.write("=" * 75 + "\n")

        print("\n" + "=" * 75)
        print(f"✅ HOÀN TẤT TOÀN BỘ K-FOLDS! Báo cáo đã lưu tại: {report_path}")
        print("=" * 75)
    else:
        print("\n" + "=" * 75)
        print("✅ KIỂM TRA THỬ HOÀN TẤT THÀNH CÔNG! (Không có file log/report nào được tạo)")
        print("=" * 75)


if __name__ == "__main__":
    main()