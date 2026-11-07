"""
===============================================================================
SCRIPT: PRETRAIN TS-TCC CONTRASTIVE ENCODER
===============================================================================
"""

import argparse
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.pretrain_ssl.tstcc.tstcc_trainer import train_tstcc_encoder
from config.datasets.registry import DATASETS
from config.datasets.base_dataset_config import BaseDatasetConfig



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
parser.add_argument("--tag", type=str, default="",
                    help="Tag phụ để phân biệt lần chạy (vd 'bs64', 'bs128'). Rỗng = không tag.")
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

        train_tstcc_encoder(
            data_path=data_path,
            dataset_name=name,
            backbone_type=args.backbone,
            in_channels=BaseDatasetConfig.IN_CHANNELS,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=DEVICE,
            num_workers=2 if DEVICE == "cuda" else 0,
            tag=args.tag,
        )
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN!")
    print("=" * 80)


if __name__ == "__main__":
    main()
