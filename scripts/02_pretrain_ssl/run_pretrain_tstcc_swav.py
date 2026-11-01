"""
===============================================================================
SCRIPT: PRETRAIN HYBRID TS-TCC x SwAV ENCODER
===============================================================================
Đặt file tại: scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py

Checkpoint lưu theo đúng quy ước mà scripts/03_cross_domain/run_few_shot.py đọc:
    checkpoints/ssl_pretrain/<method>/<dataset>/<backbone>/
        <method>_<backbone>_encoder_pretrained_<dataset>.pt        (best sau warmup)
        last_<method>_<backbone>_encoder_pretrained_<dataset>.pt   (epoch cuối)
với <method> = "tstcc_swav" (+ "_<variant>" nếu dùng --variant cho ablation).
===============================================================================
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.pretrain_ssl.tstcc_swav.tstcc_swav_trainer import train_tstcc_swav_encoder
from config.datasets.registry import DATASETS
from config.datasets.base_dataset_config import BaseDatasetConfig
from config.methods.tstcc_swav_config import TSTCCSwAVConfig as SwCfg
from models.encoders.builder import build_encoder
from utils.complexity import ComplexityAnalyzer

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

# ================== ARGUMENT PARSER ==================
parser = argparse.ArgumentParser(description="Pretrain hybrid TS-TCC x SwAV trên HAR datasets")
parser.add_argument("--backbone", type=str, default="standard",
                    choices=["tstcc", "standard", "cnn_transformer", "vit_1d"])
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=3e-4)
parser.add_argument("--datasets", nargs="+",
                    default=["uci_har", "motion_sense", "hhar_phone", "hhar_watch"])
parser.add_argument("--num_workers", type=int, default=2,
                    help="Tăng lên (4-8) nếu log cho thấy 'chờ data' chiếm phần lớn thời gian epoch")

# --- Ghi đè config cho ablation (mặc định lấy từ TSTCCSwAVConfig) ---
parser.add_argument("--num_prototypes", type=int, default=None)
parser.add_argument("--lambda_cc", type=float, default=None, help="> 0 để bật thêm NT-Xent (vd 0.7)")
parser.add_argument("--lambda_swav", type=float, default=None)
parser.add_argument("--min_t", type=int, default=None)
parser.add_argument("--warmup_epochs", type=int, default=None)
parser.add_argument("--variant", type=str, default="",
                    help="Hậu tố tên method cho ablation, vd 'cc' -> thư mục tstcc_swav_cc "
                         "(khi chạy few-shot dùng --method tstcc_swav_cc)")
parser.add_argument("--complexity", action="store_true",
                    help="Đo params/MACs/latency của encoder sau khi pretrain (lưu complexity.json)")
args = parser.parse_args()

for attr, val in [("NUM_PROTOTYPES", args.num_prototypes), ("LAMBDA_CC", args.lambda_cc),
                  ("LAMBDA_SWAV", args.lambda_swav), ("MIN_T", args.min_t),
                  ("WARMUP_EPOCHS", args.warmup_epochs)]:
    if val is not None:
        setattr(SwCfg, attr, val)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = "tstcc_swav" + (f"_{args.variant}" if args.variant else "")


def main():
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True   # shape cố định (B, 6, 128) -> chọn kernel nhanh nhất

    print("=" * 80)
    print(f"🌟 PRETRAIN {METHOD.upper()} TRÊN: {args.datasets} | Thiết bị: {DEVICE.upper()}")
    print(f"🧠 Backbone: {args.backbone.upper()} | Epochs: {args.epochs} | Batch: {args.batch_size} "
          f"| K={SwCfg.NUM_PROTOTYPES} | λ_tc={SwCfg.LAMBDA_TC} λ_swav={SwCfg.LAMBDA_SWAV} "
          f"λ_cc={SwCfg.LAMBDA_CC} | min_t={SwCfg.MIN_T}")
    print("=" * 80)

    wall = {}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"⚠️ Bỏ qua dataset không hợp lệ: {name}")
            continue

        data_path = DATASETS[name]["all"]
        if not data_path.exists():
            print(f"❌ Không tìm thấy file dữ liệu: {data_path}")
            continue

        save_dir = OUTPUT_ROOT / "checkpoints" / "ssl_pretrain" / METHOD / name
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"{METHOD}_{args.backbone}_encoder_pretrained_{name}.pt"

        t0 = time.perf_counter()
        train_tstcc_swav_encoder(
            data_path=data_path,
            save_dir=save_dir,
            checkpoint_name=checkpoint_name,
            backbone_type=args.backbone,
            in_channels=BaseDatasetConfig.IN_CHANNELS,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            device=DEVICE,
            num_workers=args.num_workers if DEVICE == "cuda" else 0,
        )
        wall[name] = round(time.perf_counter() - t0, 1)

        if args.complexity:
            info = ComplexityAnalyzer.measure_from_checkpoint(
                checkpoint_path=save_dir / args.backbone / checkpoint_name,
                backbone_type=args.backbone,
                in_channels=BaseDatasetConfig.IN_CHANNELS,
                input_size=(1, BaseDatasetConfig.IN_CHANNELS, 128),
                device=DEVICE,
                build_fn=build_encoder,
            )
            with open(save_dir / args.backbone / "complexity.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4)

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN! Thời gian chạy (giây, gồm cả setup):", wall)
    print("=" * 80)


if __name__ == "__main__":
    main()

# =============================================================================
# CÁCH CHẠY
# =============================================================================
# Test nhanh (1 epoch, 1 dataset) — xem dòng "T_out", "chờ data", "ETA":
#   uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
#       --backbone standard --epochs 1 --datasets uci_har
#
# Chạy thật:
#   uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
#       --backbone standard --epochs 60 --batch_size 64 \
#       --datasets uci_har motion_sense hhar_phone hhar_watch
#
# Ablation có thêm NT-Xent (λ_cc=0.7), lưu ở thư mục riêng tstcc_swav_cc:
#   uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
#       --variant cc --lambda_cc 0.7 --datasets uci_har motion_sense
#
# Few-shot với checkpoint mới: xem hướng dẫn sửa run_few_shot.py trong phản hồi.
# =============================================================================