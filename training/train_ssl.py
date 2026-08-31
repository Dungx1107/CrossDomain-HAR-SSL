import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Thiết lập đường dẫn thư mục gốc của dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.uci_har_config import UCIHARConfig
from models.ssl.contrastive.augmentations import TS_TCC_Augmentation
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.ssl.contrastive.ts_tcc_model import TSTCCModel
from losses.nt_xent import NTXentLoss


class PretrainDatasetWrapper(Dataset):
    """
    Bộ bọc Dataset để nhận chuỗi tín hiệu thô (bỏ qua nhãn)
    và sinh ra cặp views (Weak, Strong) phục vụ Contrastive Learning.
    Hỗ trợ cả PyTorch Dataset lẫn Tensor thô.
    """
    def __init__(self, raw_data, aug_pipeline):
        self.raw_data = raw_data
        self.aug_pipeline = aug_pipeline

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]
        # Nếu item là tuple (x, y) từ Dataset thông thường thì chỉ lấy x
        x_raw = item[0] if isinstance(item, (tuple, list)) else item
        return self.aug_pipeline(x_raw)


def train_ssl_encoder(
    train_dataset,
    dataset_name: str,
    in_channels: int = 6,
    feature_dim: int = 128,
    projection_dim: int = 64,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    epochs: int = 40,
    temperature: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = os.path.join(PROJECT_ROOT, "checkpoints", "ssl")
):
    """
    Huấn luyện TS-TCC SSL Encoder trên một tập dữ liệu bất kỳ và lưu checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tstcc_encoder_pretrained_{dataset_name.lower()}.pt")

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN TS-TCC SSL TRÊN TẬP: {dataset_name.upper()}")
    print(f"   - Kích thước tập Train: {len(train_dataset)} mẫu")
    print(f"   - Số kênh đầu vào: {in_channels} | Feature Dim: {feature_dim}")
    print(f"   - Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"   - Thiết bị: {device}")
    print(f"   - Đường dẫn lưu: {save_path}")
    print("=" * 80)

    # 1. DataLoader với bộ biến đổi Augmentation
    aug_pipeline = TS_TCC_Augmentation()
    ssl_dataset = PretrainDatasetWrapper(train_dataset, aug_pipeline)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # 2. Khởi tạo mô hình TS-TCC
    encoder_backbone = StandardSensorEncoder1D(in_channels=in_channels, feature_dim=feature_dim)
    model = TSTCCModel(encoder=encoder_backbone, feature_dim=feature_dim, projection_dim=projection_dim).to(device)

    criterion = NTXentLoss(temperature=temperature)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # 3. Vòng lặp huấn luyện tự giám sát (Contrastive SSL Loop)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_w, x_s in ssl_loader:
            x_w, x_s = x_w.to(device), x_s.to(device)

            optimizer.zero_grad()
            h_w, h_s = model(x_w, x_s)
            loss = criterion(h_w, h_s)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        print(f"Epoch [{epoch:^3d}/{epochs:^3d}] | 📉 Contrastive Loss: {avg_loss:.5f}")

    # 4. Lưu lại riêng trọng số Backbone Encoder sạch
    torch.save(model.encoder.state_dict(), save_path)
    print("=" * 80)
    print(f"✅ Hoàn tất huấn luyện SSL cho {dataset_name.upper()}!")
    print(f"💾 File checkpoint đã lưu thành công tại: {save_path}")
    print("=" * 80 + "\n")

    return save_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # KỊCH BẢN 1: TIỀN HUẤN LUYỆN TS-TCC TRÊN TẬP MOTIONSENSE
    # =========================================================================
    # print("\n>>> Bắt đầu với MotionSense...")
    # motionsense_train_dataset = MotionSenseDataset(
    #     data_dir=MotionSenseConfig.RAW_DATA_DIR,
    #     subjects_list=MotionSenseConfig.TRAIN_SUBJECTS,
    #     config=MotionSenseConfig
    # )
    # train_ssl_encoder(
    #     train_dataset=motionsense_train_dataset,
    #     dataset_name="motionsense",
    #     in_channels=MotionSenseConfig.IN_CHANNELS,
    #     batch_size=MotionSenseConfig.BATCH_SIZE,
    #     learning_rate=MotionSenseConfig.LEARNING_RATE,
    #     epochs=MotionSenseConfig.EPOCHS,
    #     device=device
    # )

    # =========================================================================
    # KỊCH BẢN 2: TIỀN HUẤN LUYỆN TS-TCC TRÊN TẬP UCI-HAR
    # =========================================================================
    print("\n>>> Bắt đầu với UCI-HAR...")
    if not os.path.exists(UCIHARConfig.PROCESSED_TRAIN_PATH):
        raise FileNotFoundError(f"Không tìm thấy dữ liệu UCI-HAR tại: {UCIHARConfig.PROCESSED_TRAIN_PATH}")

    uci_train_data = torch.load(UCIHARConfig.PROCESSED_TRAIN_PATH, map_location="cpu", weights_only=True)
    # Lấy ra tensor samples (Bỏ qua nhãn y)
    uci_train_samples = uci_train_data["samples"]

    train_ssl_encoder(
        train_dataset=uci_train_samples,
        dataset_name="uci_har",
        in_channels=UCIHARConfig.IN_CHANNELS,
        batch_size=64,
        learning_rate=1e-3,
        epochs=40,
        device=device
    )


if __name__ == "__main__":
    main()