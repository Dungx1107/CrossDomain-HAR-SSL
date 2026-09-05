import sys
import json
from pathlib import Path
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from engines.pretrain_ssl.contrastive_trainer import train_contrastive_encoder

# 1. Danh sách dataset và cấu hình chạy trực tiếp (không cần class)
DATASETS = ["uci_har", "motionsense"]
EPOCHS = 40
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Gom mapping đường dẫn bằng Dict ngắn gọn (không cần hàm get_dataset_info)
DATASET_MAP = {
    "motionsense": (Path(MotionSenseConfig.DATA_ALL_PATH), int(MotionSenseConfig.IN_CHANNELS)),
    "uci_har": (Path(UCIHARConfig.DATA_ALL_PATH), int(UCIHARConfig.IN_CHANNELS)),
}

def main():
    print(f"🌟 BẮT ĐẦU PRETRAIN SSL TRÊN: {DATASETS} | Thiết bị: {DEVICE}")

    for name in DATASETS:
        data_path, in_channels = DATASET_MAP[name]
        save_dir = PROJECT_ROOT / "experiments" / "ssl_pretrain" / name

        # Gọi trực tiếp engine đã có
        results = train_contrastive_encoder(
            data_path=data_path,
            save_dir=save_dir,
            in_channels=in_channels,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            device=DEVICE
        )

        # Lưu lại kết quả
        with open(save_dir / "metrics" / "pretrain_summary.json", "w", encoding="utf-8") as f:
            json.dump({"dataset": name, "epochs": EPOCHS, "results": results}, f, indent=4)

    print("🎉 HOÀN THÀNH PRETRAIN TẤT CẢ DATASET!")

if __name__ == "__main__":
    main()