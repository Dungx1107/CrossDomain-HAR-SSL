import sys
from pathlib import Path
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from engines.pretrain_ssl.contrastive_trainer import train_contrastive_encoder

# 1. Danh sách dataset và cấu hình chạy
DATASETS = ["uci_har", "motionsense"]
BACKBONE_TYPE = "vit_1d"  # "cnn_transformer" | "vit_1d" | "standard"
EPOCHS = 40
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. mapping đường dẫn bằng Dict
DATASET_MAP = {
    "motionsense": (Path(MotionSenseConfig.DATA_ALL_PATH), int(MotionSenseConfig.IN_CHANNELS)),
    "uci_har": (Path(UCIHARConfig.DATA_ALL_PATH), int(UCIHARConfig.IN_CHANNELS)),
}


def main():
    print(f"🌟 BẮT ĐẦU PRETRAIN SSL TRÊN: {DATASETS} | Thiết bị: {DEVICE}")

    for name in DATASETS:
        data_path, in_channels = DATASET_MAP[name]
        save_dir = PROJECT_ROOT / "checkpoints" / "ssl_pretrain" / name
        checkpoint_name = f"tstcc_{BACKBONE_TYPE}_encoder_pretrained_{name}.pt"

        print(f"\n🚀 ĐANG PRETRAIN: {name.upper()}")
        print(f"📂 Dữ liệu: {data_path}")
        print(f"🧠 Backbone       : {BACKBONE_TYPE.upper()}")
        print(f"💾 Lưu tại: {save_dir}")
        print("-" * 50)

        results = train_contrastive_encoder(
            data_path=data_path,
            save_dir=save_dir,
            checkpoint_name=checkpoint_name,
            backbone_type=BACKBONE_TYPE,
            in_channels=in_channels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            measure_complexity=True
        )

        # print(f"✅ Đã pretrain xong {name}!")
        print(f"   📁 Checkpoint: {results['best_ckpt']}")
        print(f"   📉 Best Loss: {results['best_loss']:.5f}")
        print(f"   📊 Tổng số mẫu: {results['total_samples']:,}")

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN TẤT CẢ DATASET!")
    print("=" * 80)


if __name__ == "__main__":
    main()
