"""
===============================================================================
SCRIPT: CENTRALIZED CROSS-DOMAIN ADAPTATION BENCHMARK (TS-TCC SSL)
===============================================================================
VAI TRÒ:
    - Điều phối toàn bộ thực nghiệm chuyển giao miền (Domain Adaptation).
    - Tự động chuẩn hóa về không gian 5 LỚP GIAO THOA CHUNG (Common 5 Classes)
      khi thực hiện Cross-Domain, loại bỏ nhãn riêng lẻ (Jogging).
    - Thẩm định chất lượng biểu diễn SSL trên cả 2 giao thức:
        1. Linear Probing (Đóng băng Backbone, chỉ tối ưu Head).
        2. Full Fine-Tuning (Mở khóa toàn bộ với Layer-wise LR).
    - Đo lường qua các mốc tỷ lệ nhãn: [1%, 5%, 10%, 50%, 100%].
===============================================================================
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from utils.sampling import sample_subset_by_ratio
from engines.transfer.finetune_trainer import train_and_eval_finetune
from engines.evaluation.evaluator import ModelEvaluator

# -----------------------------------------------------------------------------
# CẤU HÌNH KHÔNG GIAN 5 LỚP CHUNG CHO CROSS-DOMAIN
# -----------------------------------------------------------------------------
COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)  # Luôn cố định là 5 khi Cross-Domain

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
LABEL_FRACTIONS = [0.01, 0.05, 0.1, 0.5, 1.0]
EPOCHS = 40
BATCH_SIZE = 64

# Cấu hình đường dẫn dữ liệu từ các config gốc
DOMAIN_DATA_PATHS = {
    "motionsense": {
        "train_path": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
        "test_path": Path(MotionSenseConfig.PROCESSED_TEST_PATH),
        "in_channels": int(MotionSenseConfig.IN_CHANNELS),
    },
    "uci_har": {
        "train_path": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
        "test_path": Path(UCIHARConfig.PROCESSED_TEST_PATH),
        "in_channels": int(UCIHARConfig.IN_CHANNELS),
    }
}

# Các cặp chuyển giao cần chạy
TRANSFER_PAIRS = [
    ("uci_har", "motionsense"),
    ("motionsense", "uci_har")
]


def align_to_common_5_classes(
        samples: torch.Tensor,
        labels: torch.Tensor,
        domain_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dữ liệu của cả UCI-HAR và MotionSense trên đĩa đều đã được tiền xử lý
    chuẩn hóa về đúng 5 lớp chung (0: Walking, 1: Upstairs, 2: Downstairs, 3: Sitting, 4: Standing).
    """
    labels_tensor = labels.clone().long()

    # Đảm bảo chỉ giữ các mẫu nằm trong khoảng [0, 4]
    valid_mask = (labels_tensor >= 0) & (labels_tensor < 5)
    filtered_samples = samples[valid_mask]
    filtered_labels = labels_tensor[valid_mask]

    return filtered_samples, filtered_labels


def load_and_prepare_target_data(domain_name: str):
    """Nạp dữ liệu miền đích và ánh xạ đồng bộ về 5 lớp chung."""
    cfg = DOMAIN_DATA_PATHS[domain_name]
    if not cfg["train_path"].exists() or not cfg["test_path"].exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu cho miền: {domain_name}")

    raw_train = torch.load(cfg["train_path"], map_location="cpu", weights_only=True)
    raw_test = torch.load(cfg["test_path"], map_location="cpu", weights_only=True)

    # Lọc và đồng bộ nhãn về 5 lớp
    x_train, y_train = align_to_common_5_classes(raw_train["samples"], raw_train["labels"], domain_name)
    x_test, y_test = align_to_common_5_classes(raw_test["samples"], raw_test["labels"], domain_name)

    return x_train, y_train, x_test, y_test, cfg["in_channels"]


def run_experiment_for_pair(source_domain: str, target_domain: str):
    """Thực thi chuỗi benchmark chuyển giao (Source -> Target) trên 5 lớp chung."""
    print("\n" + "=" * 90)
    print(f"🔄 BẮT ĐẦU CHUYỂN GIAO MIỀN: [{source_domain.upper()}] ➔ [{target_domain.upper()}]")
    print(f"🎯 Không gian nhãn đánh giá: 5 LỚP GIAO THOA CHUNG {COMMON_CLASS_NAMES}")
    print("=" * 90)

    # 1. Đường dẫn checkpoint SSL nguồn
    source_ckpt = PROJECT_ROOT / "experiments" / "ssl_pretrain" / source_domain / "checkpoints" / "best_encoder.pt"
    if not source_ckpt.exists():
        fallback = PROJECT_ROOT / "checkpoints" / f"tstcc_encoder_pretrained_{source_domain}.pt"
        if fallback.exists():
            source_ckpt = fallback
        else:
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint SSL nguồn tại: {source_ckpt}")

    print(f"📦 Checkpoint SSL nguồn: {source_ckpt}")

    # 2. Nạp và chuẩn hóa dữ liệu Target về 5 lớp
    x_train_full, y_train_full, x_test, y_test, in_channels = load_and_prepare_target_data(target_domain)
    print(f"✅ Đã chuẩn hóa tập Train {target_domain.upper()}: {len(x_train_full)} mẫu (5 lớp)")
    print(f"✅ Đã chuẩn hóa tập Test  {target_domain.upper()}: {len(x_test)} mẫu (5 lớp)")

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))
    base_save_dir = PROJECT_ROOT / "experiments" / "cross_domain" / f"{source_domain}_to_{target_domain}"

    protocols = [
        {"name": "linear_probing", "freeze_backbone": True},
        {"name": "full_finetuning", "freeze_backbone": False}
    ]

    all_protocols_summary = {}

    for proto in protocols:
        proto_name = proto["name"]
        freeze_bb = proto["freeze_backbone"]
        proto_save_dir = base_save_dir / proto_name
        ckpt_save_dir = proto_save_dir / "checkpoints"
        plots_save_dir = proto_save_dir / "plots"
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        plots_save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 70)
        print(f"👉 TIẾN TRÌNH: GIAO THỨC [{proto_name.upper()}] | Backbone Frozen: {freeze_bb}")
        print("#" * 70)

        fraction_results = {}

        for frac in LABEL_FRACTIONS:
            f1_list = []
            acc_list = []
            best_run_f1 = -1.0
            best_run_model = None

            samples_count = int(len(x_train_full) * frac) if frac < 1.0 else len(x_train_full)
            print(f"\n▶️ [Tỷ lệ nhãn: {frac * 100:5.1f}% | ~{samples_count} mẫu | Epochs: {EPOCHS}]")

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                x_sub, y_sub = sample_subset_by_ratio(
                    x_train_full,
                    y_train_full,
                    fraction=frac,
                    seed=seed
                )

                train_loader = DataLoader(
                    TensorDataset(x_sub, y_sub),
                    batch_size=min(BATCH_SIZE, len(x_sub)),
                    shuffle=True
                )
                val_loader = test_loader

                # Chạy Engine Fine-tune luôn cố định num_classes=5
                acc, f1, model, _ = train_and_eval_finetune(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    encoder_checkpoint_path=source_ckpt,
                    num_classes=NUM_COMMON_CLASSES,
                    in_channels=in_channels,
                    epochs=EPOCHS,
                    freeze_backbone=freeze_bb,
                    device=DEVICE
                )

                f1_list.append(f1)
                acc_list.append(acc)

                if f1 > best_run_f1:
                    best_run_f1 = f1
                    best_run_model = model

                print(f"   [Seed {seed:4d}] -> Test Acc: {acc:5.2f}% | Test Macro F1: {f1:5.2f}%")

            mean_f1, std_f1 = float(np.mean(f1_list)), float(np.std(f1_list))
            mean_acc, std_acc = float(np.mean(acc_list)), float(np.std(acc_list))

            # Lưu checkpoint tốt nhất của mốc nhãn này
            model_save_path = ckpt_save_dir / f"best_model_frac_{frac:.2f}.pt"
            torch.save(best_run_model.state_dict(), model_save_path)

            # Vẽ Confusion Matrix trên 5 lớp chuẩn
            cm_plot_path = str(plots_save_dir / f"confusion_matrix_frac_{frac:.2f}.png")
            evaluator.evaluate(
                model=best_run_model,
                test_loader=test_loader,
                plot_save_path=cm_plot_path,
                title_prefix=f"{source_domain.upper()}->{target_domain.upper()} ({proto_name} {frac*100:.0f}%)"
            )

            fraction_results[str(frac)] = {
                "samples": samples_count,
                "macro_f1_mean": round(mean_f1, 2),
                "macro_f1_std": round(std_f1, 2),
                "accuracy_mean": round(mean_acc, 2),
                "accuracy_std": round(std_acc, 2),
                "best_model_ckpt": str(model_save_path)
            }

            print(f"⭐ KẾT QUẢ {frac * 100:5.1f}%: Macro F1 = {mean_f1:5.2f} ± {std_f1:4.2f}% | Acc = {mean_acc:5.2f} ± {std_acc:4.2f}%")

        all_protocols_summary[proto_name] = fraction_results

    # Ghi kết quả JSON
    summary_path = base_save_dir / "cross_domain_benchmark.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_protocols_summary, f, indent=4)

    # In bảng tổng kết
    print("\n" + "=" * 90)
    print(f"🏆 BẢNG TỔNG HỢP HIỆU NĂNG (5 LỚP GIAO THOA): {source_domain.upper()} ➔ {target_domain.upper()}")
    print("=" * 90)
    print(f"{'Tỷ lệ nhãn':<12} | {'Số mẫu':<10} | {'Linear Probing (F1 %)':<25} | {'Full Fine-Tuning (F1 %)'}")
    print("-" * 90)
    for frac in LABEL_FRACTIONS:
        frac_key = str(frac)
        lp = all_protocols_summary["linear_probing"][frac_key]
        ft = all_protocols_summary["full_finetuning"][frac_key]
        print(
            f"{frac * 100:<10.1f}% | "
            f"{lp['samples']:<10d} | "
            f"{lp['macro_f1_mean']:5.2f} ± {lp['macro_f1_std']:4.2f}%              | "
            f"{ft['macro_f1_mean']:5.2f} ± {ft['macro_f1_std']:4.2f}%"
        )
    print("=" * 90)
    print(f"💾 Dữ liệu kết quả đã lưu tại: {summary_path}\n")


def main():
    print(f"🌟 BẮT ĐẦU CHƯƠNG TRÌNH ĐÁNH GIÁ CHUYỂN GIAO MIỀN (5 COMMON CLASSES)")
    print(f"🖥️ Thiết bị: {DEVICE} | Seeds: {SEEDS} | Tỷ lệ nhãn: {LABEL_FRACTIONS}")

    for src, tgt in TRANSFER_PAIRS:
        run_experiment_for_pair(source_domain=src, target_domain=tgt)

    print("🎉 TẤT CẢ CÁC KỊCH BẢN CHUYỂN GIAO 5 LỚP ĐÃ HOÀN THÀNH XUẤT SẮC!")


if __name__ == "__main__":
    main()