"""
===============================================================================
SCRIPT: FEW-SHOT CROSS-DOMAIN ADAPTATION
- Đọc dữ liệu:        utils/data_io.py        (load_data)
- Lấy mẫu k-shot:     utils/sampling.py       (sample_k_shot)
- Đánh giá + báo cáo: utils/few_shot_report.py (evaluate_model, save_k_shot_report)
- Mỗi (k_shot, protocol) ghi: detailed_all_seeds.json, summary_...json,
  aggregated_confusion_matrix.png.
===============================================================================
"""

import sys, json, argparse, os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

from engines.transfer.finetune_trainer import train_and_eval_finetune
from models.encoders.builder import build_encoder
from config.datasets.registry import DATASETS
from config.datasets.base_dataset_config import BaseDatasetConfig
from utils.data_io import load_data
from utils.sampling import sample_k_shot
from engines.evaluation.few_shot_report import evaluate_model, save_k_shot_report

MODE_DISPLAY_NAMES = {
    "full": "linear_probing",
    "partial": "partial_finetuning",
    "none": "full_finetuning"
}

COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)

parser = argparse.ArgumentParser(description="Few-shot Cross-Domain HAR")
parser.add_argument("--freeze_modes", nargs="+", type=str, default=["full", "partial:1", "none"],
                    help="Các chế độ chạy: 'full', 'none', 'partial' (mặc định 1 block), hoặc 'partial:N' (mở N block)")
parser.add_argument("--method", type=str, default="tstcc", choices=["tstcc", "swav", "crosshar"])
parser.add_argument("--backbone", type=str, default="standard",
                    choices=["standard", "cnn_transformer", "vit_1d"])
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 7, 99])
parser.add_argument("--k_shots", nargs="+", type=int, default=[10, 30, 50, 100, 150, 200])
parser.add_argument("--pairs", nargs="+", type=str, required=True, help="e.g., motionsense:uci_har")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_mode(mode_str):
    """'partial:2' -> ('partial', 2, 'partial_2blocks'); 'full' -> ('full', 1, 'linear_probing')."""
    freeze_mode, _, blocks_str = mode_str.partition(":")
    if freeze_mode not in MODE_DISPLAY_NAMES:
        raise ValueError(f"freeze_mode '{freeze_mode}' không hợp lệ. Chọn trong {list(MODE_DISPLAY_NAMES)}")
    num_blocks = int(blocks_str) if blocks_str else 1
    display_name = f"partial_{num_blocks}blocks" if freeze_mode == "partial" else MODE_DISPLAY_NAMES[freeze_mode]
    return freeze_mode, num_blocks, display_name


MODES = [parse_mode(m) for m in args.freeze_modes]  # kiểm tra sớm, lỗi thì dừng ngay


def run_few_shot(src, tgt):
    print(f"\n{'=' * 70}\n🔄 FEW-SHOT BENCHMARK: [{src.upper()}] ➔ [{tgt.upper()}]\n{'=' * 70}")
    cfg = DATASETS[tgt]

    X_tr_full, y_tr_full = load_data(cfg["train"], NUM_COMMON_CLASSES)
    X_v_full, y_v_full = load_data(cfg["val"], NUM_COMMON_CLASSES)
    X_ts_full, y_ts_full = load_data(cfg["test"], NUM_COMMON_CLASSES)

    source_ckpt = (OUTPUT_ROOT / "checkpoints/ssl_pretrain" / args.method / src / args.backbone
                   / f"{args.method}_{args.backbone}_encoder_pretrained_{src}.pt")
    if not source_ckpt.exists():
        raise FileNotFoundError(f"❌ Thiếu checkpoint pretrain tại: {source_ckpt}")

    # val_loader = DataLoader(TensorDataset(X_v_full, y_v_full), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_ts_full, y_ts_full), batch_size=64, shuffle=False)

    base_save_dir = OUTPUT_ROOT / "outputs_evaluation/cross_k_shot" / args.method / args.backbone / f"{src}_to_{tgt}"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_type": "few_shot_cross_domain", "source_domain": src, "target_domain": tgt,
        "pretrain_method": args.method, "backbone": args.backbone, "checkpoint": str(source_ckpt),
        "k_shots": args.k_shots, "seeds": args.seeds, "epochs": args.epochs, "batch_size": args.batch_size,
        "freeze_modes": args.freeze_modes
    }
    with open(base_save_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    for k in args.k_shots:
        k_shot_name = f"{k}_shot"
        for freeze_mode, num_blocks, display_name in MODES:
            print(f"\n▶ TIẾN TRÌNH: {k_shot_name.upper()} | MODE: {display_name.upper()}")
            seed_results = {}

            for seed in args.seeds:
                torch.manual_seed(seed)
                x_sub_tr, y_sub_tr = sample_k_shot(X_tr_full, y_tr_full, k, seed, NUM_COMMON_CLASSES)
                tr_loader = DataLoader(
                    TensorDataset(x_sub_tr, y_sub_tr),
                    batch_size=min(args.batch_size, len(x_sub_tr)),
                    shuffle=True
                )

                _, _, model, _ = train_and_eval_finetune(
                    train_loader=tr_loader,
                    # val_loader=val_loader,
                    val_loader=None,
                    test_loader=test_loader,
                    encoder_checkpoint_path=source_ckpt,
                    encoder=build_encoder(args.backbone, BaseDatasetConfig.IN_CHANNELS),
                    num_classes=NUM_COMMON_CLASSES,
                    in_channels=BaseDatasetConfig.IN_CHANNELS,
                    epochs=args.epochs,
                    freeze_mode=freeze_mode,
                    num_unfrozen_blocks=num_blocks,
                    device=DEVICE
                )

                result = evaluate_model(model, test_loader, DEVICE, COMMON_CLASS_NAMES)
                result["train_samples_used"] = len(y_sub_tr)
                seed_results[seed] = result
                print(f"   [Seed {seed:<4}] Mẫu Train: {len(y_sub_tr):<4} | "
                      f"Acc: {result['accuracy']:5.2f}% | F1: {result['macro_f1']:5.2f}%")

            summary = save_k_shot_report(
                base_save_dir / k_shot_name / display_name,
                display_name, k_shot_name, k, seed_results, COMMON_CLASS_NAMES
            )
            print(f"✅ Đã lưu -> {k_shot_name} | {display_name}: "
                  f"F1 Mean = {summary['overall_metrics']['macro_f1_mean']}%")


if __name__ == "__main__":
    for p in args.pairs:
        src, tgt = p.split(":")
        run_few_shot(src.strip(), tgt.strip())