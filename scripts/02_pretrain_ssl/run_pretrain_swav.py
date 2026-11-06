"""
===============================================================================
SCRIPT: RUN PRETRAIN SWAV (REPLACING OLD PROTOTYPE SCRIPT)
===============================================================================
"""

import argparse
from config.datasets.registry import DATASETS
from config.methods.swav_config import SwAVConfig
from engines.pretrain_ssl.swav.swav_trainer import train_swav_encoder
from config.datasets.motionsense_config import MotionSenseConfig
from config.datasets.hhar_config import HHARConfig
from config.datasets.uci_har_config import UCIHARConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain SSL models using SwAV (Prototypical) method.")
    parser.add_argument("--datasets", nargs="+", required=True, help="List of datasets (e.g. uci pamap2)")
    parser.add_argument(
        "--backbone",
        type=str,
        default="standard",
        choices=["tstcc", "standard", "cnn_transformer", "vit_1d"],
        help="Loại kiến trúc backbone (mặc định: standard)",
    )
    parser.add_argument("--epochs", type=int, default=SwAVConfig.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=SwAVConfig.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=SwAVConfig.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Num workers for DataLoader")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag phụ để phân biệt lần chạy (vd 'bs64', 'bs128'). Rỗng = không tag.")
    return parser.parse_args()


def main():
    args = parse_args()

    DATASET_CFG = {
        "uci_har": UCIHARConfig,
        "motion_sense": MotionSenseConfig,
        "hhar_phone": HHARConfig,
        "hhar_watch": HHARConfig,
    }

    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"⚠️ Bỏ qua {ds_name}: Không tìm thấy trong Dataset Registry.")
            continue

        ds_info = DATASETS[ds_name]
        ds_cfg = DATASET_CFG[ds_name]

        train_swav_encoder(
            data_path=ds_info["all"],
            dataset_name=ds_name,
            backbone_type=args.backbone,
            in_channels=ds_cfg.IN_CHANNELS,
            seq_len=ds_cfg.WINDOW_SIZE,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            device=args.device,
            tag=args.tag,
        )


if __name__ == "__main__":
    main()
