"""
===============================================================================
SCRIPT: PRETRAIN TS-TCC CONTRASTIVE ENCODER
===============================================================================
"""

import argparse
import os
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.pretrain_ssl.tstcc_trainer import train_tstcc_encoder
from config.datasets.registry import DATASETS
from config.datasets.base_dataset_config import BaseDatasetConfig

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

# ================== ARGUMENT PARSER ==================
parser = argparse.ArgumentParser(
    description="Pretrain TS-TCC SSL trên HAR DATASETS"
)
parser.add_argument(
    "--backbone",
    type=str,
    default="standard",
    choices=["tstcc", "standard", "cnn_transformer", "vit_1d"],
    help="Loại kiến trúc backbone (mặc định: vit_1d)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=60,
    help="Số epoch pretrain (mặc định: 60)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size (mặc định: 64)",
)
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["uci_har", "motion_sense"],
    help="Danh sách dataset cần pretrain",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 80)
    print(f"🌟 BẮT ĐẦU PRETRAIN SSL TRÊN: {args.datasets} | Thiết bị: {DEVICE.upper()}")
    print(f"🧠 Backbone: {args.backbone.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    for name in args.datasets:
        if name not in DATASETS:
            print(f"⚠️ Bỏ qua dataset không hợp lệ: {name}")
            continue

        data_path = DATASETS[name]["all"]

        if not data_path.exists():
            print(f"❌ Không tìm thấy file dữ liệu: {data_path}")
            continue

        save_dir = (
                OUTPUT_ROOT
                / "checkpoints"
                / "ssl_pretrain"
                / "tstcc"
                / name
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"tstcc_{args.backbone}_encoder_pretrained_{name}.pt"

        train_tstcc_encoder(
            data_path=data_path,
            save_dir=save_dir,
            checkpoint_name=checkpoint_name,
            backbone_type=args.backbone,
            in_channels=BaseDatasetConfig.IN_CHANNELS,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=DEVICE,
            measure_complexity=True,
            num_workers=2 if DEVICE == "cuda" else 0
        )

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# =============================================================================
# HƯỚNG DẪN CHẠY (COMMENT — KHÔNG ẢNH HƯỞNG LOGIC)
# =============================================================================
#
# Script này pretrain SSL (KHÔNG dùng nhãn) encoder TS-TCC trên các dataset HAR.
# Checkpoint lưu theo epoch có TRAIN LOSS nhỏ nhất (không phải epoch cuối).
#
# -----------------------------------------------------------------------------
# 1. CHẠY TRÊN MÁY LOCAL (dùng uv)
# -----------------------------------------------------------------------------
#
# Bước 0 — đứng ở project root:
#   cd /home/dungx/Workspace/AI_ML_Projects/Human_Activity_Recognition/CrossDomain-HAR-SSL
#
# Bước 1 — test nhanh 1 epoch, 1 dataset (~1–2 phút):
#   uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc.py  --backbone standard --epochs 1 --batch_size 64 --datasets uci_har
#
# Bước 3 — chạy thật 60 epoch, đủ 4 dataset:
#   uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc.py --backbone standard --epochs 60 --batch_size 64 --datasets uci_har motion_sense hhar_phone hhar_watch
#
#  — thử backbone nặng hơn (vit_1d), giảm batch nếu OOM:
#   uv run python scripts/.../run_pretrain_tstcc.py \
#     --backbone vit_1d --epochs 60 --batch_size 32 \
#     --datasets uci_har motion_sense
#
# Kiểm tra kết quả:
#   ls -la checkpoints/ssl_pretrain/tstcc/uci_har/standard/
#   cat checkpoints/ssl_pretrain/tstcc/uci_har/standard/loss_history.csv
#   cat checkpoints/ssl_pretrain/tstcc/uci_har/standard/complexity.json
#
# Output gồm:
#   - tstcc_<backbone>_encoder_pretrained_<dataset>.pt  (checkpoint encoder)
#   - loss_history.csv / loss_curve.png                 (loss từng epoch)
#   - complexity.json / config.json                     (params, MACs, config)
#   - pretrain_summary.json (ở thư mục cha tstcc/)
#
# -----------------------------------------------------------------------------
# 2. CHẠY TRÊN KAGGLE
# -----------------------------------------------------------------------------
#
# Bước 1 — cài thop ở cell đầu notebook (Kaggle không có sẵn):
#   import sys, subprocess
#   subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "thop"])
#
# Bước 2 — test nhanh trước (tiết kiệm quota GPU):
#   !cd /kaggle/working/CrossDomain-HAR-SSL && \
#     python scripts/.../run_pretrain_tstcc.py \
#       --backbone standard --epochs 1 --datasets uci_har
#
# Bước 3 — chạy thật:
#   !cd /kaggle/working/CrossDomain-HAR-SSL && \
#     python scripts/.../run_pretrain_tstcc.py \
#       --backbone standard --epochs 60 --batch_size 64 \
#       --datasets uci_har motion_sense hhar_phone hhar_watch
#
# Lưu ý Kaggle:
#   - Output tự ghi vào /kaggle/working/checkpoints/ssl_pretrain/tstcc/...
#     (script tự phát hiện biến KAGGLE_KERNEL_RUN_TYPE).
#   - /kaggle/working reset mỗi session -> Save Version hoặc download
#     checkpoint trước khi hết session.
#   - GPU T4 (16GB): batch_size=64 ổn với 'standard'; 'vit_1d' -> 32.
#
# -----------------------------------------------------------------------------
# 3. THAM SỐ
# -----------------------------------------------------------------------------
#
#   --backbone    : standard | tstcc | cnn_transformer | vit_1d   (default: standard)
#   --epochs      : số epoch pretrain                              (default: 60)
#   --batch_size  : số mẫu/batch                                   (default: 64)
#   --datasets    : danh sách key trong config/datasets/registry.py
#                   (uci_har, motion_sense, hhar_phone, hhar_watch)
#
# Gợi ý giá trị:
#   - Test nhanh      : --epochs 1 --datasets uci_har
#   - Chạy thật       : --epochs 60 --datasets uci_har motion_sense hhar_phone hhar_watch
#   - GPU nhỏ (<8GB)  : --batch_size 32
#   - ViT             : --batch_size 32 (nặng hơn standard)
#
# -----------------------------------------------------------------------------
# 4. CHECKLIST TRƯỚC KHI CHẠY THẬT
# -----------------------------------------------------------------------------
#
#   (1) Data tồn tại:
#       ls -la data/processed/uci_har/dataset_all.pt
#
#   (2) thop đã cài (cho MACs):
#       uv pip list | grep -i thop
#
#   (3) IN_CHANNELS = 6:
#       uv run python -c "from config.datasets.base_dataset_config import BaseDatasetConfig as B; print(B.IN_CHANNELS)"
#
#   (4) Test 1 epoch:
#       uv run python scripts/.../run_pretrain_tstcc.py --epochs 1 --datasets uci_har
#
#   Nếu (1)-(3) OK và (4) chạy xong không lỗi -> chạy full 60 epoch an toàn.
#
# -----------------------------------------------------------------------------
# 5. LƯU Ý
# -----------------------------------------------------------------------------
#
#   - Pretrain KHÔNG dùng nhãn -> không có accuracy/F1 trong log.
#     Chỉ có loss_total, loss_tc, loss_cc, grad_norm.
#   - KHÔNG có validation trong code hiện tại -> checkpoint chọn theo
#     TRAIN LOSS thấp nhất. Muốn chặt chẽ hơn: tách 10% data làm val SSL,
#     tính loss_tc+loss_cc trên đó, chọn checkpoint theo val loss.
#   - GRAD_CLIP_MAX_NORM = 2.0 trong config/methods/tstcc_config.py
#     -> chống exploding gradient. Xem cột grad_norm trong CSV để biết
#     clip có can thiệp mạnh không (grad_norm ~ 2.0 liên tục = clip mạnh).
#   - save_dir hiện là .../ssl_pretrain/tstcc/<dataset>/<backbone>/.
#     Nếu downstream load từ .../contrastive/... thì cần đồng bộ path.
#
# =============================================================================
