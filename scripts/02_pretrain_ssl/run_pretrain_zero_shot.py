"""
===============================================================================
CLI SCRIPT: ĐIỀU PHỐI PRETRAIN SSL TRÊN 90% TẬP TRAIN NGUỒN (CHỐNG RÒ RỈ TEST)
===============================================================================
Chức năng:
    - Nhận danh sách dataset nguồn (motionsense, uci_har, hhar_phone, hhar_watch).
    - Tự động cắt 90% không nhãn của `train.pt` để làm dữ liệu pretrain SSL.
    - Đóng gói với ContrastiveDatasetWrapper (tạo cặp Weak / Strong).
    - Dùng `val.pt` nguồn để theo dõi hàm loss và lưu model tối ưu nhất.
===============================================================================
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from config.hhar_config import HHARConfig
from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from engines.pretrain_ssl.zero_shot_contrastive_trainer import train_zero_shot_contrastive_encoder

parser = argparse.ArgumentParser(description="Zero-Shot SSL Pretrain")
parser.add_argument("--datasets", nargs="+", default=["motionsense", "uci_har"],
                    help="Danh sách dataset nguồn cần pretrain (vd: motionsense uci_har)")
parser.add_argument("--backbone", type=str, default="standard",
                    choices=["standard", "cnn_transformer"],
                    help="Kiến trúc backbone (standard hoặc cnn_transformer)")
parser.add_argument("--epochs", type=int, default=40, help="Số epochs pretrain")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BẢNG ÁNH XẠ ĐƯỜNG DẪN TRAIN VÀ VAL NGUỒN
SOURCE_DATASET_MAP = {
    "motionsense": {
        "train": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
        "val": Path(MotionSenseConfig.PROCESSED_VAL_PATH),
        "in_channels": int(MotionSenseConfig.IN_CHANNELS)
    },
    "uci_har": {
        "train": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
        "val": Path(UCIHARConfig.PROCESSED_VAL_PATH),
        "in_channels": int(UCIHARConfig.IN_CHANNELS)
    },
    "hhar_phone": {
        "train": HHARConfig.PROCESSED_DIR_PHONE / "train.pt",
        "val": HHARConfig.PROCESSED_DIR_PHONE / "val.pt",
        "in_channels": int(HHARConfig.IN_CHANNELS)
    },
    "hhar_watch": {
        "train": HHARConfig.PROCESSED_DIR_WATCH / "train.pt",
        "val": HHARConfig.PROCESSED_DIR_WATCH / "val.pt",
        "in_channels": int(HHARConfig.IN_CHANNELS)
    }
}


def load_and_split_90_pretrain(train_pt_path: Path, seed: int = 42) -> torch.Tensor:
    """Nạp file train.pt và lấy đúng 90% không nhãn bằng Stratified split."""
    data = torch.load(train_pt_path, map_location="cpu", weights_only=True)
    X = data["samples"]
    y = data["labels"].squeeze()

    # Lọc 5 lớp chung
    mask = (y >= 0) & (y < 5)
    X, y = X[mask], y[mask]

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.float()

    if X.ndim == 3 and X.shape[1] == 128 and X.shape[2] == 6:
        X = X.permute(0, 2, 1)

    # Chia 90% Pretrain / 10% Finetune
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    pretrain_idx, _ = next(sss.split(X, y.numpy()))

    return X[pretrain_idx]


def load_val_tensor(val_pt_path: Path) -> torch.Tensor:
    """Nạp file val.pt để đánh giá Loss kiểm định cho SSL."""
    data = torch.load(val_pt_path, map_location="cpu", weights_only=True)
    X = data["samples"]
    y = data["labels"].squeeze()
    mask = (y >= 0) & (y < 5)
    X = X[mask].float()
    if X.ndim == 3 and X.shape[1] == 128 and X.shape[2] == 6:
        X = X.permute(0, 2, 1)
    return X


def main():
    print("=" * 80)
    print(f"🌟 BẮT ĐẦU QUY TRÌNH PRETRAIN ZERO-SHOT (90% TRAIN NGUỒN)")
    print(f"🔧 Backbone: {args.backbone} | Epochs: {args.epochs} | Thiết bị: {DEVICE}")
    print("=" * 80)

    for ds_name in args.datasets:
        if ds_name not in SOURCE_DATASET_MAP:
            print(f"⚠️ Bỏ qua dataset không hợp lệ: {ds_name}")
            continue

        cfg = SOURCE_DATASET_MAP[ds_name]
        print(f"\n📂 Đang xử lý miền nguồn: {ds_name.upper()}")

        # 1. Trích xuất 90% Train và 100% Val
        X_train_90 = load_and_split_90_pretrain(cfg["train"], seed=args.seed)
        X_val = load_val_tensor(cfg["val"])

        print(f"   -> Mẫu pretrain SSL (90% train): {X_train_90.shape[0]}")
        print(f"   -> Mẫu validation SSL (val set) : {X_val.shape[0]}")

        # 2. Đóng gói thành Contrastive DataLoader (Tạo view Weak/Strong)
        train_ds = ContrastiveDatasetWrapper(X_train_90)
        val_ds = ContrastiveDatasetWrapper(X_val)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            drop_last=True if len(train_ds) >= args.batch_size else False
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
        )

        # 3. Thư mục lưu Checkpoint
        save_dir = PROJECT_ROOT / "checkpoints" / "zero_shot_pretrain" / ds_name / args.backbone
        ckpt_name = f"tstcc_{args.backbone}_encoder_pretrained_{ds_name}.pt"

        # 4. Huấn luyện
        train_zero_shot_contrastive_encoder(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            checkpoint_name=ckpt_name,
            backbone_type=args.backbone,
            in_channels=cfg["in_channels"],
            epochs=args.epochs,
            device=DEVICE
        )

    print("\n🎉 HOÀN TẤT PRETRAIN CHO CÁC MIỀN NGUỒN!")


if __name__ == "__main__":
    main()