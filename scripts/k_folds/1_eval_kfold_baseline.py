import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Tự động trỏ về thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# TẬN DỤNG TOÀN BỘ CÁC MODULE SẴN CÓ
from models.encoder import TSTCCEncoder
from models.classifier import HARClassifier
from datasets.kfold_splitter import get_kfold_loaders
from training.supervised_trainer import SupervisedTrainer
from training.evaluator import Evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_kfold_supervised(dataset_name="motionsense", k=5, epochs=30, lr=1e-3, batch_size=64):
    """
    Chạy Supervised Baseline K-Fold bằng cách tái sử dụng Engine Trainer & Evaluator
    """
    # 1. Tận dụng module kfold_splitter để lấy data
    fold_loaders, num_classes = get_kfold_loaders(
        dataset_name=dataset_name,
        k=k,
        batch_size=batch_size
    )

    acc_list, f1_list = [], []
    print(f"\n---> [Dataset: {dataset_name.upper()} | K = {k} | Epochs = {epochs} | Device = {device}]")

    for fold_idx, (train_loader, test_loader) in enumerate(fold_loaders, 1):
        # 2. Khởi tạo mô hình
        encoder = TSTCCEncoder(in_channels=9).to(device)
        model = HARClassifier(encoder=encoder, num_classes=num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 3. Tận dụng SupervisedTrainer sẵn có để huấn luyện
        trainer = SupervisedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        trainer.train(epochs=epochs)

        # 4. Tận dụng Evaluator sẵn có để tính toán metrics
        evaluator = Evaluator(model=model, test_loader=test_loader, device=device)
        metrics = evaluator.evaluate()  # Trả về dict: {'accuracy': ..., 'f1_macro': ...}

        acc = metrics["accuracy"]
        f1 = metrics["f1_macro"]

        acc_list.append(acc)
        f1_list.append(f1)
        print(f"  • Fold {fold_idx}/{k} | Accuracy: {acc:.2f}% | Macro F1: {f1:.2f}%")

    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_f1, std_f1 = np.mean(f1_list), np.std(f1_list)

    print(f"  ⭐ K={k} Summary -> Acc: {mean_acc:.2f} ± {std_acc:.2f}% | F1: {mean_f1:.2f} ± {std_f1:.2f}%")
    return mean_acc, std_acc, mean_f1, std_f1


def main():
    parser = argparse.ArgumentParser(description="K-Fold Supervised Baseline Evaluation")
    parser.add_argument("--dataset", type=str, default="motionsense", help="Tên dataset (motionsense, uci_har,...)")
    parser.add_argument("--k_list", nargs="+", type=int, default=[3, 5, 7, 10], help="Danh sách K cần chạy")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch cho mỗi fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    report_dir = PROJECT_ROOT / "document" / "0_reports" / "7_kfold_evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"baseline_{args.dataset}_kfold_report.txt"

    print("=" * 70)
    print(f"🚀 BẮT ĐẦU CHẠY SUPERVISED BASELINE K-FOLDS ({args.dataset.upper()})")
    print("=" * 70)

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

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"BÁO CÁO K-FOLD SUPERVISED BASELINE - TẬP DỮ LIỆU: {args.dataset.upper()}\n")
        f.write(f"Cấu hình: Epochs={args.epochs}, LR={args.lr}, Batch Size={args.batch_size}\n")
        f.write("=" * 70 + "\n")
        f.write("\n".join(summary_rows) + "\n")
        f.write("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print(f"✅ HOÀN TẤT! Báo cáo đã được lưu vào: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()