"""
===============================================================================
SCRIPT HUẤN LUYỆN PRETRAIN: MASKED SENSOR MODELING (MSM)
===============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from config.hhar_config import HHARConfig

from datasets.base_dataset import get_har_all_loader, get_har_dataloaders
from models.ssl.masked.cnn1d_masked import StandardCNNMaskedAutoEncoder
from models.encoders.cnn_transformer import CNNTransformerEncoder
from models.ssl.masked.mask_generator import SegmentMaskGenerator
from engines.pretrain_ssl.masked_trainer import MaskedSSLTrainer
from utils.logger import setup_logger


class CrossHARMaskedAutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, d_model: int = 128, mask_ratio: float = 0.15):
        super().__init__()
        self.feature_dim = d_model  # Đã thêm lưu feature_dim tường minh
        self.mask_generator = SegmentMaskGenerator(mask_ratio=mask_ratio)
        self.encoder = CNNTransformerEncoder(in_channels=in_channels, d_model=d_model)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.ConvTranspose1d(64, in_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor):
        mask = self.mask_generator(x)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        z = self.encoder(x_masked)
        x_recon = self.decoder(z)

        mask_f = mask.float()
        loss = torch.sum(((x_recon - x) ** 2) * mask_f) / (mask_f.sum() + 1e-8)
        return {"loss": loss, "x_recon": x_recon, "mask": mask, "latent": z}


def get_dataset_dir(dataset_name: str) -> Path:
    mapping = {
        "uci_har": UCIHARConfig.PROCESSED_DIR,
        "motionsense": MotionSenseConfig.PROCESSED_DIR,
        "hhar_phone": HHARConfig.PROCESSED_DIR_PHONE,
        "hhar_watch": HHARConfig.PROCESSED_DIR_WATCH,
        "hhar_combined": HHARConfig.PROCESSED_DIR_COMBINED
    }
    return Path(mapping[dataset_name])


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Masked Sensor Modeling HAR")
    parser.add_argument("--datasets", nargs="+", default=["motionsense", "uci_har"],
                        help="Danh sách dataset cần pretrain (vd: --datasets motionsense uci_har)")
    parser.add_argument("--backbone", type=str, default="standard",
                        choices=["standard", "cnn_transformer"],
                        help="Loại Backbone: 'standard' (1D-CNN) hoặc 'cnn_transformer' (CrossHAR)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate ban đầu")
    parser.add_argument("--mask_ratio", type=float, default=0.15, help="Tỷ lệ che dữ liệu (15%)")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng CPU nạp dữ liệu (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("RunMaskedPretrain")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in args.datasets:
        data_dir = get_dataset_dir(dataset_name)

        save_dir = PROJECT_ROOT / "checkpoints" / "ssl_pretrain" / "masked" / dataset_name / args.backbone
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"masked_{args.backbone}_encoder_pretrained_{dataset_name}.pt"

        logger.info(f"\n" + "=" * 80)
        logger.info(f"🚀 BẮT ĐẦU PRETRAIN MASKED: {dataset_name.upper()} | Backbone: {args.backbone}")
        logger.info(f"💾 Thư mục lưu: {save_dir}")
        logger.info(f"📦 Tên Checkpoint: {checkpoint_name}")
        logger.info("=" * 80)

        train_loader = get_har_all_loader(data_dir=data_dir, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        _, val_loader, _ = get_har_dataloaders(data_dir=data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        if args.backbone == "standard":
            model = StandardCNNMaskedAutoEncoder(in_channels=6, feature_dim=128, mask_ratio=args.mask_ratio)
        elif args.backbone == "cnn_transformer":
            model = CrossHARMaskedAutoEncoder(in_channels=6, d_model=128, mask_ratio=args.mask_ratio)
        else:
            raise ValueError(f"Backbone không hợp lệ: {args.backbone}")

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        config_record = {
            "dataset": dataset_name,
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "mask_ratio": args.mask_ratio,
            "checkpoint_name": checkpoint_name,
            "device": str(device)
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_record, f, indent=4)

        trainer = MaskedSSLTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            save_dir=save_dir,
            checkpoint_name=checkpoint_name,
            max_grad_norm=2.0,
            logger=logger
        )

        trainer.fit(epochs=args.epochs)

    logger.info("🎉 HOÀN TẤT PRETRAIN TOÀN BỘ DATASET!")


if __name__ == "__main__":
    main()