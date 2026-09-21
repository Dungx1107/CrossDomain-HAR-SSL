"""
===============================================================================
SCRIPT: PRETRAIN TS-TCC CONTRASTIVE ENCODER
===============================================================================
"""

import argparse
from pathlib import Path
import sys
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from engines.pretrain_ssl.contrastive_trainer import train_contrastive_encoder

# ================== ARGUMENT PARSER ==================
parser = argparse.ArgumentParser(
    description="Pretrain TS-TCC SSL trên UCI-HAR & MotionSense"
)
parser.add_argument(
    "--backbone",
    type=str,
    default="vit_1d",
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
    default=["uci_har", "motionsense"],
    help="Danh sách dataset cần pretrain",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_MAP = {
    "motionsense": (
        Path(MotionSenseConfig.DATA_ALL_PATH),
        int(MotionSenseConfig.IN_CHANNELS),
    ),
    "uci_har": (
        Path(UCIHARConfig.DATA_ALL_PATH),
        int(UCIHARConfig.IN_CHANNELS),
    ),
}


def main():
  print("=" * 80)
  print(
      f"🌟 BẮT ĐẦU PRETRAIN SSL TRÊN: {args.datasets} | Thiết bị: {DEVICE.upper()}"
  )
  print(
      f"🧠 Backbone: {args.backbone.upper()} | Epochs: {args.epochs} | Batch"
      f" Size: {args.batch_size}"
  )
  print("=" * 80)

  for name in args.datasets:
    if name not in DATASET_MAP:
      print(f"⚠️ Bỏ qua dataset không hợp lệ: {name}")
      continue

    data_path, in_channels = DATASET_MAP[name]

    save_dir = (
        PROJECT_ROOT
        / "checkpoints"
        / "ssl_pretrain"
        / "contrastive"
        / name
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"tstcc_{args.backbone}_encoder_pretrained_{name}.pt"

    print(f"\n🚀 ĐANG PRETRAIN: {name.upper()}")
    print(f"📂 Dữ liệu: {data_path}")
    print(f"🧠 Backbone: {args.backbone.upper()}")
    print(f"💾 Thư mục lưu: {save_dir}")
    print("-" * 50)

    results = train_contrastive_encoder(
        data_path=data_path,
        save_dir=save_dir,
        checkpoint_name=checkpoint_name,
        backbone_type=args.backbone,
        in_channels=in_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=DEVICE,
        measure_complexity=True,
    )

    print(f"   📁 Checkpoint : {results['best_ckpt']}")
    print(f"   📉 Best Loss  : {results['best_loss']:.5f}")
    print(f"   📊 Tổng số mẫu: {results['total_samples']:,}")

  print("\n" + "=" * 80)
  print("🎉 HOÀN THÀNH PRETRAIN TẤT CẢ DATASET!")
  print("=" * 80)


if __name__ == "__main__":
  main()