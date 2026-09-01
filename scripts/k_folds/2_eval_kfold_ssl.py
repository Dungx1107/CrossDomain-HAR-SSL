import os
import sys
import argparse
from pathlib import Path

# =============================================================================
# 1. BẮT BUỘC ĐẶT TRÊN CÙNG: Nạp thư mục gốc của project vào sys.path
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# 2. IMPORT CÁC THƯ VIỆN BÊN NGOÀI
# =============================================================================
import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# 3. IMPORT CÁC MODULE NỘI BỘ TRONG PROJECT
# =============================================================================
from datasets.kfold_splitter import get_kfold_loaders
from models.har_classifier import HARClassifier
from training.supervised_trainer import SupervisedTrainer
from training.evaluator import ModelEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_encoder(model, checkpoint_path):
    """
    Nạp trọng số Encoder từ file checkpoint SSL đã pretrain trên Source domain.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file checkpoint tại: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Hỗ trợ cả trường hợp checkpoint lưu raw state_dict hoặc bọc trong dictionary
    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        state_dict = checkpoint["encoder_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Nạp vào submodule encoder của HARClassifier
    missing_keys, unexpected_keys = model.encoder.load_state_dict(state_dict, strict=False)
    print(f"  [Checkpoint Loaded] -> File: {Path(checkpoint_path).name}")
    if missing_keys:
        print(f"  ⚠️ Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"  ⚠️ Unexpected keys: {len(unexpected_keys)}")


def train_fold_engine(model, train_loader, optimizer, criterion, epochs):
    """
    Tận dụng SupervisedTrainer để thực thi vòng lặp huấn luyện từng fold.
    """
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        tracker=None,
        checkpoint_path="",
        scheduler=None
    )

    for epoch in range(1, epochs + 1):
        trainer.train_one_epoch(train_loader)


def run_single_mode(target_dataset, checkpoint_path, mode, k, epochs, lr, batch_size):
    """
    Chạy K-Fold cho một chế độ cụ thể (linear_probe hoặc fine_tune).
    """
    fold_loaders, in_channels, num_classes = get_kfold_loaders(
        dataset_name=target_dataset,
        k=k,
        batch_size=batch_size
    )

    evaluator = ModelEvaluator(device=device)
    acc_list, f1_list = [], []

    print(f"\n---> [MODE: {mode.upper()} | Target: {target_dataset.upper()} | K = {k} | Epochs = {epochs} | LR = {lr}]")

    for fold_idx, (train_loader, test_loader) in enumerate(fold_loaders, 1):
        # 1. Khởi tạo mô hình
        model = HARClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_dim=128
        ).to(device)

        # 2. Nạp trọng số Pretrained SSL vào Encoder
        load_pretrained_encoder(model, checkpoint_path)

        # 3. Đóng băng hoặc mở khóa tham số tùy theo mode
        if mode == "linear_probe":
            for param in model.encoder.parameters():
                param.requires_grad = False
            trainable_params = model.classifier.parameters()
        elif mode == "fine_tune":
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = model.parameters()
        else:
            raise ValueError(f"Mode không hợp lệ: {mode}")

        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # 4. Huấn luyện
        train_fold_engine(model, train_loader, optimizer, criterion, epochs)

        # 5. Đánh giá
        metrics = evaluator.evaluate(model=model, test_loader=test_loader,
                                     title_prefix=f"{mode.upper()} - Fold {fold_idx}/{k}")
        acc = metrics["accuracy"] * 100
        f1 = metrics["macro_f1"] * 100

        acc_list.append(acc)
        f1_list.append(f1)
        print(f"  • [{mode}] Fold {fold_idx}/{k} -> Accuracy: {acc:.2f}% | Macro F1: {f1:.2f}%")

    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_f1, std_f1 = np.mean(f1_list), np.std(f1_list)
    print(
        f"  ⭐ TỔNG KẾT [{mode.upper()}] K={k} -> Acc: {mean_acc:.2f} ± {std_acc:.2f}% | F1: {mean_f1:.2f} ± {std_f1:.2f}%")

    return mean_acc, std_acc, mean_f1, std_f1


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain K-Fold SSL Evaluation (Linear Probing & Fine-tuning)")
    parser.add_argument("--source_dataset", type=str, default="uci_har", help="Tên dataset nguồn đã dùng pretrain SSL")
    parser.add_argument("--target_dataset", type=str, default="motionsense",
                        help="Tên dataset đích cần đánh giá K-Fold")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Đường dẫn đến file checkpoint encoder (.pt)")
    parser.add_argument("--k_list", nargs="+", type=int, default=[3, 5, 7, 10], help="Danh sách K cần chạy")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch cho mỗi fold")
    parser.add_argument("--lr_linear", type=float, default=1e-3, help="Learning rate cho Linear Probing")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="Learning rate cho Full Fine-tuning")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Chạy kiểm tra thử luồng, không lưu file report")
    args = parser.parse_args()

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ CROSS-DOMAIN HAR-SSL K-FOLDS")
    print(
        f"   • Hướng chuyển giao : {args.source_dataset.upper()} (Pretrained) ➔ {args.target_dataset.upper()} (Target)")
    print(f"   • Checkpoint Path   : {args.checkpoint_path}")
    print(f"   • Thứ tự thực thi   : 1. LINEAR PROBING  ➔  2. FULL FINE-TUNING")
    if args.dry_run:
        print("⚠️  CHẾ ĐỘ DRY-RUN: Chỉ kiểm tra luồng chạy, không ghi file report vào đĩa.")
    print("=" * 80)

    # 1. Chạy Linear Probing trước
    linear_results = []
    print("\n" + "#" * 40 + " GIAI ĐOẠN 1: LINEAR PROBING " + "#" * 40)
    for k in args.k_list:
        m_acc, s_acc, m_f1, s_f1 = run_single_mode(
            target_dataset=args.target_dataset,
            checkpoint_path=args.checkpoint_path,
            mode="linear_probe",
            k=k,
            epochs=args.epochs,
            lr=args.lr_linear,
            batch_size=args.batch_size
        )
        linear_results.append(f"K = {k:2d} | Accuracy: {m_acc:.2f} ± {s_acc:.2f}% | Macro F1: {m_f1:.2f} ± {s_f1:.2f}%")

    # 2. Chạy Full Fine-tuning sau
    finetune_results = []
    print("\n" + "#" * 40 + " GIAI ĐOẠN 2: FULL FINE-TUNING " + "#" * 40)
    for k in args.k_list:
        m_acc, s_acc, m_f1, s_f1 = run_single_mode(
            target_dataset=args.target_dataset,
            checkpoint_path=args.checkpoint_path,
            mode="fine_tune",
            k=k,
            epochs=args.epochs,
            lr=args.lr_finetune,
            batch_size=args.batch_size
        )
        finetune_results.append(
            f"K = {k:2d} | Accuracy: {m_acc:.2f} ± {s_acc:.2f}% | Macro F1: {m_f1:.2f} ± {s_f1:.2f}%")

    # 3. Lưu báo cáo tổng hợp
    if not args.dry_run:
        report_dir = PROJECT_ROOT / "document" / "0_reports" / "7_kfold_evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"ssl_cross_{args.source_dataset}_to_{args.target_dataset}_kfold_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"BÁO CÁO CROSS-DOMAIN HAR-SSL K-FOLD EVALUATION\n")
            f.write(
                f"Source (Pretrain): {args.source_dataset.upper()} | Target (Eval): {args.target_dataset.upper()}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Epochs={args.epochs}, Batch Size={args.batch_size}\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. KẾT QUẢ LINEAR PROBING (Khóa Encoder, LR=" + str(args.lr_linear) + "):\n")
            f.write("-" * 80 + "\n")
            f.write("\n".join(linear_results) + "\n\n")
            f.write("2. KẾT QUẢ FULL FINE-TUNING (Mở khóa toàn bộ, LR=" + str(args.lr_finetune) + "):\n")
            f.write("-" * 80 + "\n")
            f.write("\n".join(finetune_results) + "\n")
            f.write("=" * 80 + "\n")

        print("\n" + "=" * 80)
        print(f"✅ ĐÃ HOÀN TẤT TẤT CẢ CÁC CHẾ ĐỘ! Báo cáo đã lưu tại: {report_path}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✅ DRY-RUN HOÀN TẤT THÀNH CÔNG! (Không lưu file report)")
        print("=" * 80)


if __name__ == "__main__":
    main()