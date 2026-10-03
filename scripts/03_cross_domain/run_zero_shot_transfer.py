"""
===============================================================================
CLI SCRIPT: HUẤN LUYỆN HEAD TRÊN NGUỒN & ĐÁNH GIÁ ZERO-SHOT TRÊN TẤT CẢ MIỀN ĐÍCH
===============================================================================
Quy trình thực thi chuẩn CrossHAR:
    1. Nhận 1 Miền Nguồn (Source) và danh sách các Miền Đích (Targets).
    2. Nạp file train.pt của nguồn, trích xuất 10% có nhãn còn lại.
    3. Nạp trọng số Pretrained SSL từ Bước 1, gắn Head và train trên 10% nhãn này.
    4. Dùng val.pt của nguồn chọn mô hình tối ưu nhất (best_source_model.pt).
    5. Đánh giá IN-DOMAIN trên test.pt của chính nguồn.
    6. Đem nguyên mô hình sang chạy ZERO-SHOT trên test.pt của từng miền đích
       (Không train, không fit thêm gì).
    7. Xuất Confusion Matrix, tính Macro F1, Accuracy và in bảng kết quả hoàn chỉnh.
===============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from config.hhar_config import HHARConfig
from engines.transfer.source_classifier_trainer import train_source_classifier
from engines.evaluation.evaluator import ModelEvaluator

COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_CLASSES = len(COMMON_CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BẢNG ÁNH XẠ TẤT CẢ DỮ LIỆU
DATASET_PATHS = {
    "motionsense": {
        "train": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
        "val": Path(MotionSenseConfig.PROCESSED_VAL_PATH),
        "test": Path(MotionSenseConfig.PROCESSED_TEST_PATH),
        "in_channels": int(MotionSenseConfig.IN_CHANNELS)
    },
    "uci_har": {
        "train": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
        "val": Path(UCIHARConfig.PROCESSED_VAL_PATH),
        "test": Path(UCIHARConfig.PROCESSED_TEST_PATH),
        "in_channels": int(UCIHARConfig.IN_CHANNELS)
    },
    "hhar_phone": {
        "train": HHARConfig.PROCESSED_DIR_PHONE / "train.pt",
        "val": HHARConfig.PROCESSED_DIR_PHONE / "val.pt",
        "test": HHARConfig.PROCESSED_DIR_PHONE / "test.pt",
        "in_channels": int(HHARConfig.IN_CHANNELS)
    },
    "hhar_watch": {
        "train": HHARConfig.PROCESSED_DIR_WATCH / "train.pt",
        "val": HHARConfig.PROCESSED_DIR_WATCH / "val.pt",
        "test": HHARConfig.PROCESSED_DIR_WATCH / "test.pt",
        "in_channels": int(HHARConfig.IN_CHANNELS)
    }
}

parser = argparse.ArgumentParser(description="Zero-Shot Cross-Domain Transfer Runner")
parser.add_argument("--source", type=str, required=True,
                    choices=["motionsense", "uci_har", "hhar_phone", "hhar_watch"],
                    help="Miền nguồn (Source domain)")
parser.add_argument("--targets", nargs="+", default=["motionsense", "uci_har", "hhar_phone", "hhar_watch"],
                    help="Danh sách miền đích cần đánh giá zero-shot")
parser.add_argument("--backbone", type=str, default="standard",
                    choices=["standard", "cnn_transformer"],
                    help="Loại kiến trúc backbone")
parser.add_argument("--epochs", type=int, default=40, help="Số epochs train Head trên nguồn")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--freeze_backbone", action="store_true",
                    help="Bật cờ này để đóng băng backbone (Linear Probing trên nguồn)")
args = parser.parse_args()


def load_and_process_file(pt_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nạp file .pt, lọc 5 lớp chung và đưa về dạng chuẩn (N, 6, 128)."""
    raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    X = raw["samples"]
    y = raw["labels"].squeeze()

    mask = (y >= 0) & (y < 5)
    X, y = X[mask], y[mask]

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.float()

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)
    else:
        y = y.long()

    if X.ndim == 3 and X.shape[1] == 128 and X.shape[2] == 6:
        X = X.permute(0, 2, 1)

    return X, y


def get_source_10_percent_finetune(train_path: Path, seed: int = 42):
    """Trích xuất đúng 10% có nhãn từ train.pt của miền nguồn."""
    X_train, y_train = load_and_process_file(train_path)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    _, finetune_idx = next(sss.split(X_train, y_train.numpy()))

    return X_train[finetune_idx], y_train[finetune_idx]


def main():
    src = args.source
    targets = [t for t in args.targets if t != src]  # Lọc bỏ nguồn khỏi danh sách đích

    print("\n" + "=" * 90)
    print(f"🌐 BẮT ĐẦU CHƯƠNG TRÌNH ZERO-SHOT CROSS-DOMAIN BENCHMARK")
    print(f"🎯 NGUỒN (SOURCE): [{src.upper()}] ➔ ĐÍCH (TARGETS): {[t.upper() for t in targets]}")
    print(f"🔧 Backbone: {args.backbone.upper()} | Đóng băng Backbone: {args.freeze_backbone}")
    print("=" * 90)

    # 1. KIỂM TRA ĐƯỜNG DẪN CHECKPOINT SSL PRETRAIN NGUỒN
    ssl_ckpt = (PROJECT_ROOT / "checkpoints" / "zero_shot_pretrain" / src /
                args.backbone / f"tstcc_{args.backbone}_encoder_pretrained_{src}.pt")
    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy Checkpoint Pretrain tại: {ssl_ckpt}\n"
                                f"👉 Hãy chạy script 'run_pretrain_zero_shot.py --datasets {src}' trước!")

    # 2. CHUẨN BỊ DỮ LIỆU ĐỂ HUẤN LUYỆN HEAD TRÊN NGUỒN
    src_cfg = DATASET_PATHS[src]
    X_src_10, y_src_10 = get_source_10_percent_finetune(src_cfg["train"], seed=args.seed)
    X_src_val, y_src_val = load_and_process_file(src_cfg["val"])
    X_src_test, y_src_test = load_and_process_file(src_cfg["test"])

    print(f"\n📊 Dữ liệu miền nguồn ({src.upper()}):")
    print(f"   - 10% Train có nhãn : {X_src_10.shape[0]} mẫu")
    print(f"   - 100% Val nguồn    : {X_src_val.shape[0]} mẫu")
    print(f"   - 100% Test In-domain: {X_src_test.shape[0]} mẫu")

    train_loader_src = DataLoader(
        TensorDataset(X_src_10, y_src_10),
        batch_size=min(args.batch_size, len(X_src_10)),
        shuffle=True
    )
    val_loader_src = DataLoader(
        TensorDataset(X_src_val, y_src_val),
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader_src = DataLoader(
        TensorDataset(X_src_test, y_src_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    # 3. HUẤN LUYỆN CLASSIFIER HEAD TRÊN NGUỒN
    save_dir = PROJECT_ROOT / "checkpoints" / "zero_shot_experiments" / f"{src}_{args.backbone}"
    best_source_model_path = save_dir / "best_source_full_model.pt"

    print("\n⏳ Đang huấn luyện Classifier Head trên miền nguồn...")
    _, model, _ = train_source_classifier(
        train_loader=train_loader_src,
        val_loader=val_loader_src,
        encoder_checkpoint_path=ssl_ckpt,
        save_model_path=best_source_model_path,
        backbone_type=args.backbone,
        num_classes=NUM_CLASSES,
        in_channels=src_cfg["in_channels"],
        epochs=args.epochs,
        freeze_backbone=args.freeze_backbone,
        device=DEVICE
    )

    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))
    results_summary = {}

    # 4. ĐÁNH GIÁ IN-DOMAIN (TRÊN CHÍNH TẬP TEST CỦA NGUỒN)
    print("\n" + "-" * 70)
    print(f"🏠 ĐÁNH GIÁ IN-DOMAIN: [{src.upper()} ➔ {src.upper()}] (Test Set)")
    print("-" * 70)
    in_domain_cm_path = str(save_dir / f"cm_in_domain_{src}.png")
    in_res = evaluator.evaluate(
        model=model,
        test_loader=test_loader_src,
        plot_save_path=in_domain_cm_path,
        title_prefix=f"In-Domain: {src.upper()}"
    )
    results_summary[f"{src} (In-Domain)"] = {
        "accuracy": in_res.get("accuracy", 0.0),
        "macro_f1": in_res.get("macro_f1", 0.0)
    }
    print(f"👉 In-Domain Macro F1: {in_res.get('macro_f1', 0.0):.2f}% | Acc: {in_res.get('accuracy', 0.0):.2f}%")

    # 5. ĐÁNH GIÁ ZERO-SHOT TRANSFER TRÊN CÁC MIỀN ĐÍCH
    for tgt in targets:
        print("\n" + "-" * 70)
        print(f"🚀 ĐÁNH GIÁ ZERO-SHOT: [{src.upper()} ➔ {tgt.upper()}] (0% Nhãn đích)")
        print("-" * 70)

        tgt_cfg = DATASET_PATHS[tgt]
        X_tgt_test, y_tgt_test = load_and_process_file(tgt_cfg["test"])
        test_loader_tgt = DataLoader(
            TensorDataset(X_tgt_test, y_tgt_test),
            batch_size=args.batch_size,
            shuffle=False
        )

        tgt_cm_path = str(save_dir / f"cm_zero_shot_{src}_to_{tgt}.png")
        tgt_res = evaluator.evaluate(
            model=model,
            test_loader=test_loader_tgt,
            plot_save_path=tgt_cm_path,
            title_prefix=f"Zero-Shot: {src.upper()}->{tgt.upper()}"
        )

        results_summary[f"{src} -> {tgt} (Zero-Shot)"] = {
            "accuracy": tgt_res.get("accuracy", 0.0),
            "macro_f1": tgt_res.get("macro_f1", 0.0)
        }
        print(f"👉 Zero-Shot Macro F1 ({tgt}): {tgt_res.get('macro_f1', 0.0):.2f}% | Acc: {tgt_res.get('accuracy', 0.0):.2f}%")

    # 6. XUẤT BẢNG TỔNG KẾT
    summary_json_path = save_dir / "zero_shot_benchmark_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)

    print("\n" + "=" * 90)
    print(f"🏆 BẢNG TỔNG HỢP HIỆU NĂNG ZERO-SHOT TRANSFER (SOURCE: {src.upper()})")
    print("=" * 90)
    print(f"{'Kịch bản chuyển giao':<35} | {'Accuracy (%)':<15} | {'Macro F1 (%)'}")
    print("-" * 90)
    for scenario, metrics in results_summary.items():
        print(f"{scenario:<35} | {metrics['accuracy']:<15.2f} | {metrics['macro_f1']:.2f}%")
    print("=" * 90)
    print(f"📁 Toàn bộ Checkpoint, Báo cáo và Plots đã lưu tại: {save_dir}")


if __name__ == "__main__":
    main()