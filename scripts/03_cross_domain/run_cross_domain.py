"""
===============================================================================
SCRIPT: CENTRALIZED CROSS-DOMAIN ADAPTATION BENCHMARK (TS-TCC SSL)
===============================================================================
"""

import sys
import json
from pathlib import Path
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
from utils.complexity import measure_model_complexity, print_complexity_report

# =============================================================================
# CẤU HÌNH
# =============================================================================
COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
LABEL_FRACTIONS = [0.01, 0.05, 0.1, 0.5, 1.0]
EPOCHS = 40
BATCH_SIZE = 64
MEASURE_COMPLEXITY = True

TRANSFER_PAIRS = [
    ("uci_har", "motionsense"),
    ("motionsense", "uci_har")
]

PROTOCOLS_TO_RUN = [
    {"name": "linear_probing", "freeze_backbone": True},
    {"name": "full_finetuning", "freeze_backbone": False}
]

DOMAIN_DATA_PATHS = {
    "motionsense": {
        "train_path": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
        "val_path": Path(MotionSenseConfig.PROCESSED_VAL_PATH),
        "test_path": Path(MotionSenseConfig.PROCESSED_TEST_PATH),
        "in_channels": int(MotionSenseConfig.IN_CHANNELS),
    },
    "uci_har": {
        "train_path": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
        "val_path": Path(UCIHARConfig.PROCESSED_VAL_PATH),
        "test_path": Path(UCIHARConfig.PROCESSED_TEST_PATH),
        "in_channels": int(UCIHARConfig.IN_CHANNELS),
    }
}


def load_and_prepare_target_data(domain_name: str):
    """
    Nạp dữ liệu miền đích.
    ✅ Lọc chỉ giữ 5 lớp chung (0-4), loại bỏ nhãn 5.
    """
    cfg = DOMAIN_DATA_PATHS[domain_name]
    raw_train = torch.load(cfg["train_path"], map_location="cpu", weights_only=True)
    raw_val = torch.load(cfg["val_path"], map_location="cpu", weights_only=True)
    raw_test = torch.load(cfg["test_path"], map_location="cpu", weights_only=True)

    # ✅ Lọc chỉ giữ 5 lớp chung (0-4)
    def filter_5_classes(samples, labels):
        mask = (labels >= 0) & (labels < 5)
        return samples[mask], labels[mask]

    x_train, y_train = filter_5_classes(raw_train["samples"], raw_train["labels"])
    x_val, y_val = filter_5_classes(raw_val["samples"], raw_val["labels"])
    x_test, y_test = filter_5_classes(raw_test["samples"], raw_test["labels"])

    print(f"   🔍 Sau khi lọc 5 lớp: Train={len(x_train)}, Val={len(x_val)}, Test={len(x_test)}")

    return (
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        cfg["in_channels"]
    )


def run_experiment_for_pair(source_domain: str, target_domain: str):
    """Thực thi benchmark chuyển giao."""
    print("\n" + "=" * 90)
    print(f"🔄 CHUYỂN GIAO MIỀN: [{source_domain.upper()}] ➔ [{target_domain.upper()}]")
    print(f"🎯 {NUM_COMMON_CLASSES} LỚP CHUNG: {COMMON_CLASS_NAMES}")
    print("=" * 90)

    source_ckpt = PROJECT_ROOT / "checkpoints" / "ssl_pretrain" / source_domain / f"tstcc_encoder_pretrained_{source_domain}.pt"
    if not source_ckpt.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy checkpoint SSL nguồn tại: {source_ckpt}")
    print(f"📦 Checkpoint SSL nguồn: {source_ckpt}")

    # Load dữ liệu đã lọc 5 lớp
    x_train_full, y_train_full, x_val_full, y_val_full, x_test, y_test, in_channels = load_and_prepare_target_data(target_domain)

    print(f"✅ Train: {len(x_train_full)} mẫu")
    print(f"✅ Val  : {len(x_val_full)} mẫu")
    print(f"✅ Test : {len(x_test)} mẫu (5 lớp)")

    base_save_dir = PROJECT_ROOT / "checkpoints" / "cross_domain" / f"{source_domain}_to_{target_domain}"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))
    all_protocols_summary = {}

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    for proto in PROTOCOLS_TO_RUN:
        proto_name = proto["name"]
        freeze_bb = proto["freeze_backbone"]
        proto_save_dir = base_save_dir / proto_name
        ckpt_save_dir = proto_save_dir / "checkpoints"
        plots_save_dir = proto_save_dir / "plots"
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        plots_save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 70)
        print(f"👉 GIAO THỨC: [{proto_name.upper()}] | Freeze Backbone: {freeze_bb}")
        print("#" * 70)

        fraction_results = {}

        for frac in LABEL_FRACTIONS:
            f1_list, acc_list = [], []
            best_run_f1 = -1.0
            best_run_model = None
            samples_count = int(len(x_train_full) * frac) if frac < 1.0 else len(x_train_full)

            print(f"\n▶️ Tỷ lệ: {frac * 100:5.1f}% | ~{samples_count} mẫu")

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                x_sub, y_sub = sample_subset_by_ratio(x_train_full, y_train_full, fraction=frac, seed=seed)

                train_loader = DataLoader(
                    TensorDataset(x_sub, y_sub),
                    batch_size=min(BATCH_SIZE, len(x_sub)),
                    shuffle=True
                )

                val_loader = DataLoader(
                    TensorDataset(x_val_full, y_val_full),
                    batch_size=BATCH_SIZE,
                    shuffle=False
                )

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

                print(f"   [Seed {seed:4d}] -> Acc: {acc:5.2f}% | F1: {f1:5.2f}%")

            mean_f1, std_f1 = float(np.mean(f1_list)), float(np.std(f1_list))
            mean_acc, std_acc = float(np.mean(acc_list)), float(np.std(acc_list))

            model_save_path = ckpt_save_dir / f"best_model_frac_{frac:.2f}.pt"
            if best_run_model is not None:
                torch.save(best_run_model.state_dict(), model_save_path)

            cm_plot_path = str(plots_save_dir / f"confusion_matrix_frac_{frac:.2f}.png")
            if best_run_model is not None:
                evaluator.evaluate(
                    model=best_run_model,
                    test_loader=test_loader,
                    plot_save_path=cm_plot_path,
                    title_prefix=f"{source_domain.upper()}->{target_domain.upper()} ({proto_name} {frac * 100:.0f}%)"
                )

            fraction_results[str(frac)] = {
                "samples": samples_count,
                "macro_f1_mean": round(mean_f1, 2),
                "macro_f1_std": round(std_f1, 2),
                "accuracy_mean": round(mean_acc, 2),
                "accuracy_std": round(std_acc, 2),
                "best_model_ckpt": str(model_save_path)
            }

            print(f"⭐ {frac * 100:5.1f}%: F1 = {mean_f1:5.2f} ± {std_f1:4.2f}%")

        all_protocols_summary[proto_name] = fraction_results

    # Lưu config
    config_info = {
        "source_domain": source_domain,
        "target_domain": target_domain,
        "common_classes": COMMON_CLASS_NAMES,
        "num_classes": NUM_COMMON_CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seeds": SEEDS,
        "label_fractions": LABEL_FRACTIONS,
        "protocols": PROTOCOLS_TO_RUN,
        "source_checkpoint": str(source_ckpt),
        "device": DEVICE
    }

    with open(base_save_dir / "config.json", "w") as f:
        json.dump(config_info, f, indent=4)

    summary_path = base_save_dir / "cross_domain_benchmark.json"
    with open(summary_path, "w") as f:
        json.dump(all_protocols_summary, f, indent=4)

    print("\n" + "=" * 90)
    print(f"🏆 BẢNG TỔNG HỢP: {source_domain.upper()} ➔ {target_domain.upper()}")
    print("=" * 90)
    print(f"{'Tỷ lệ':<12} | {'Số mẫu':<10} | {'Linear Probing (F1 %)':<25} | {'Full Fine-Tuning (F1 %)'}")
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


def main():
    print(f"🌟 ĐÁNH GIÁ CHUYỂN GIAO MIỀN (5 COMMON CLASSES)")
    print(f"🖥️ Device: {DEVICE} | Seeds: {SEEDS}")
    total_runs = len(TRANSFER_PAIRS) * len(PROTOCOLS_TO_RUN) * len(LABEL_FRACTIONS) * len(SEEDS)
    print(f"📊 Tổng số lần train/eval: {total_runs}")
    print("=" * 90)

    for src, tgt in TRANSFER_PAIRS:
        run_experiment_for_pair(source_domain=src, target_domain=tgt)

    print("🎉 HOÀN THÀNH!")


if __name__ == "__main__":
    main()