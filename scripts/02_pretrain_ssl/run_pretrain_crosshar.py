"""
===============================================================================
CLI SCRIPT: PRETRAIN CROSSHAR CHUẨN (3D PERMUTATION 6X + SEQUENTIAL UPDATING)
NẠP TRỌN VẸN 100% DỮ LIỆU MIỀN NGUỒN (dataset_all.pt)
===============================================================================
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from config.hhar_config import HHARConfig
from datasets.crosshar_dataset import Physical3DAugmentation, CrossHARPretrainDataset
from engines.pretrain_ssl.crosshar_sequential_trainer import train_crosshar_sequential

parser = argparse.ArgumentParser(description="CrossHAR Hierarchical Pretrain Runner")
parser.add_argument("--datasets", nargs="+", default=["motionsense"])
parser.add_argument("--backbone", type=str, default="cnn_transformer", choices=["standard", "cnn_transformer"])
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--warmup_msm_epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--expand_6x", action="store_true", default=True)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_DATASET_MAP = {
    "motionsense": {"all": Path(MotionSenseConfig.PROCESSED_DIR) / "dataset_all.pt", "in_channels": int(MotionSenseConfig.IN_CHANNELS)},
    "uci_har": {"all": Path(UCIHARConfig.PROCESSED_DIR) / "dataset_all.pt", "in_channels": int(UCIHARConfig.IN_CHANNELS)},
    "hhar_phone": {"all": HHARConfig.PROCESSED_DIR_PHONE / "dataset_all.pt", "in_channels": int(HHARConfig.IN_CHANNELS)},
    "hhar_watch": {"all": HHARConfig.PROCESSED_DIR_WATCH / "dataset_all.pt", "in_channels": int(HHARConfig.IN_CHANNELS)}
}

def load_dataset_all_tensor(file_path: Path) -> torch.Tensor:
    if not file_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu toàn phần: {file_path}")

    data = torch.load(file_path, map_location="cpu", weights_only=True)
    X = data["samples"]
    y = data["labels"].squeeze()

    mask = (y >= 0) & (y < 5)
    X = X[mask]

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.float()

    if X.ndim == 3 and X.shape[1] == 128 and X.shape[2] == 6:
        X = X.permute(0, 2, 1)

    return X

def main():
    print("=" * 90)
    print("🌟 BẮT ĐẦU CHƯƠNG TRÌNH PRETRAIN CROSSHAR CHUẨN (100% DATASET_ALL)")
    print(f"🔧 Backbone: {args.backbone.upper()} | Tổng Epochs: {args.epochs} | Thiết bị: {DEVICE.upper()}")
    print("=" * 90)

    aug3d = Physical3DAugmentation()
    torch.manual_seed(args.seed)

    for ds_name in args.datasets:
        if ds_name not in SOURCE_DATASET_MAP:
            print(f"⚠️ Dataset không hợp lệ: {ds_name}")
            continue

        cfg = SOURCE_DATASET_MAP[ds_name]
        print(f"\n📂 Đang nạp 100% dữ liệu miền nguồn [{ds_name.upper()}]: {cfg['all']}")

        X_all = load_dataset_all_tensor(cfg["all"])
        print(f"   - Tổng số mẫu 100% gốc: {X_all.shape[0]}")

        # Train lấy trọn vẹn 100%, Val dùng chính tập này (chưa expand) để theo dõi loss tiến trình
        X_train = X_all
        X_val = X_all

        if args.expand_6x:
            X_train_pretrain = aug3d.expand_dataset_6x(X_train)
            print(f"   - Sau mở rộng 6x ma trận hoán vị 3D: {X_train_pretrain.shape[0]} mẫu")
        else:
            X_train_pretrain = X_train

        train_ds = CrossHARPretrainDataset(X_train_pretrain)
        val_ds = CrossHARPretrainDataset(X_val)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True if len(train_ds) >= args.batch_size else False
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        save_dir = PROJECT_ROOT / "checkpoints" / "ssl_pretrain" / "crosshar" / ds_name / args.backbone
        ckpt_name = f"crosshar_{args.backbone}_encoder_pretrained_{ds_name}.pt"

        train_crosshar_sequential(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            checkpoint_name=ckpt_name,
            backbone_type=args.backbone,
            in_channels=cfg["in_channels"],
            epochs=args.epochs,
            warmup_msm_epochs=args.warmup_msm_epochs,
            learning_rate=args.lr,
            device=DEVICE
        )

if __name__ == "__main__":
    main()