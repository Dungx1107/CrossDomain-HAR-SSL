import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

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

    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        state_dict = checkpoint["encoder_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.encoder.load_state_dict(state_dict, strict=False)


def train_fold_engine(model, train_loader, optimizer, criterion, epochs, desc_prefix=""):
    """
    Huấn luyện qua các epoch có thanh tiến trình tqdm.
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

    epoch_bar = tqdm(range(1, epochs + 1), desc=desc_prefix, leave=False)
    for epoch in epoch_bar:
        loss = trainer.train_one_epoch(train_loader)
        if loss is not None:
            epoch_bar.set_postfix({"loss": f"{loss:.4f}"})


def run_single_mode(target_dataset, checkpoint_path, mode, k, epochs, lr, batch_size):
    """
    Chạy K-Fold cho một chế độ cụ thể (linear_probe hoặc fine_tune) kèm thanh tiến trình fold.
    """
    fold_loaders, in_channels, num_classes = get_kfold_loaders(
        dataset_name=target_dataset,
        k=k,
        batch_size=batch_size
    )

    evaluator = ModelEvaluator(device=device)
    acc_list, f1_list = [], []

    print(f"\n---> [MODE: {mode.upper()} | Target: {target_dataset.upper()} | K = {k} | Epochs = {epochs} | LR = {lr}]")

    fold_bar = tqdm(enumerate(fold_loaders, 1), total=k, desc=f"Progress [{mode.upper()} K={k}]")
    for fold_idx, (train_loader, test_loader) in fold_bar:
        model = HARClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            feature_dim=128
        ).to(device)

        load_pretrained_encoder(model, checkpoint_path)

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

        train_fold_engine(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            desc_prefix=f"  Epochs (Fold {fold_idx}/{k})"
        )

        metrics = evaluator.evaluate(model=model, test_loader=test_loader,
                                     title_prefix=f"{mode.upper()} - Fold {fold_idx}/{k}")
        acc = metrics["accuracy"] * 100
        f1 = metrics["macro_f1"] * 100

        acc_list.append(acc)
        f1_list.append(f1)
        fold_bar.set_postfix({"Acc": f"{acc:.2f}%", "F1": f"{f1:.2f}%"})

    mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)
    mean_f1, std_f1 = np.mean(f1_list), np.std(f1_list)
    print(
        f"\n  ⭐ TỔNG KẾT [{mode.upper()}] K={k} -> Acc: {mean_acc:.2f} ± {std_acc:.2f}% | F1: {mean_f1:.2f} ± {std_f1:.2f}%")

    return mean_acc, std_acc, mean_f1, std_f1


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain K-Fold SSL Evaluation")
    parser.add_argument("--source_dataset", type=str, default="uci_har", help="Tên dataset nguồn đã pretrain SSL")
    parser.add_argument("--target_dataset", type=str, default="motionsense", help="Tên dataset đích K-Fold")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Đường dẫn file encoder (.pt)")
    parser.add_argument("--k_list", nargs="+", type=int, default=[3, 5, 7, 10], help="Danh sách K cần chạy")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch")
    parser.add_argument("--lr_linear", type=float, default=1e-3, help="Learning rate cho Linear Probe")
    parser.add_argument("--lr_finetune", type=float, default=1e-4, help="Learning rate cho Fine-tune")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dry_run", action="store_true", help="Chạy kiểm tra thử luồng, không ghi report")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ CROSS-DOMAIN HAR-SSL K-FOLDS")
    print(
        f"   • Hướng chuyển giao : {args.source_dataset.upper()} (Pretrained) ➔ {args.target_dataset.upper()} (Target)")
    print(f"   • Checkpoint Path   : {args.checkpoint_path}")
    print("   • Thứ tự thực thi   : 1. LINEAR PROBING  ➔  2. FULL FINE-TUNING")
    if args.dry_run:
        print("⚠️  CHẾ ĐỘ DRY-RUN: Không lưu file report.")
    print("=" * 80)

    linear_results = []
    print("\n" + "#" * 35 + " GIAI ĐOẠN 1: LINEAR PROBING " + "#" * 35)
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

    finetune_results = []
    print("\n" + "#" * 35 + " GIAI ĐOẠN 2: FULL FINE-TUNING " + "#" * 35)
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

    if not args.dry_run:
        report_dir = PROJECT_ROOT / "document" / "0_reports" / "7_kfold_evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"ssl_cross_{args.source_dataset}_to_{args.target_dataset}_kfold_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"BÁO CÁO CROSS-DOMAIN HAR-SSL K-FOLD EVALUATION\n")
            f.write(f"Source: {args.source_dataset.upper()} | Target: {args.target_dataset.upper()}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Epochs={args.epochs}, Batch Size={args.batch_size}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"1. KẾT QUẢ LINEAR PROBING (LR={args.lr_linear}):\n" + "-" * 80 + "\n")
            f.write("\n".join(linear_results) + "\n\n")
            f.write(f"2. KẾT QUẢ FULL FINE-TUNING (LR={args.lr_finetune}):\n" + "-" * 80 + "\n")
            f.write("\n".join(finetune_results) + "\n" + "=" * 80 + "\n")

        print("\n" + "=" * 80)
        print(f"✅ ĐÃ HOÀN TẤT! Báo cáo đã lưu tại: {report_path}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✅ DRY-RUN HOÀN TẤT THÀNH CÔNG!")
        print("=" * 80)


if __name__ == "__main__":
    main()