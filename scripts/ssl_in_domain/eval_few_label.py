import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset
from models.ssl.contrastive.augmentations import TS_TCC_Augmentation  # Module tăng cường của bạn
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.ssl.contrastive.tstcc_model import TSTCCModel
from losses.nt_xent import NTXentLoss


class PretrainDatasetWrapper(torch.utils.data.Dataset):
    """ Bộ bọc Dataset để tự động can thiệp biến đổi chuỗi thô thành bộ đôi views """

    def __init__(self, base_dataset, aug_pipeline):
        self.base_dataset = base_dataset
        self.aug_pipeline = aug_pipeline

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Đọc chuỗi thô từ Dataset Giai đoạn 1 (Bỏ qua không lấy nhãn)
        x_raw, _ = self.base_dataset[idx]
        # Gọi __call__ của file augmentations.py để lấy (tensor_weak, tensor_strong)
        return self.aug_pipeline(x_raw)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🚀 KHỞI CHẠY PHA TIỀN HUẤN LUYỆN TỰ GIÁM SÁT ĐƠN MIỀN (TS-TCC)")
    print("=" * 80)

    # 1. Nạp 100% dữ liệu thô vùng tập Train (Không sử dụng nhãn hoạt động)
    raw_train_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TRAIN_SUBJECTS,
        config=MotionSenseConfig
    )

    aug_pipeline = TS_TCC_Augmentation()
    ssl_dataset = PretrainDatasetWrapper(raw_train_dataset, aug_pipeline)
    ssl_loader = DataLoader(ssl_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Khởi tạo cấu hình Mạng bộ mã hóa nền tảng (Backbone) và Khung Siamese
    encoder_backbone = StandardSensorEncoder1D(in_channels=MotionSenseConfig.IN_CHANNELS, feature_dim=128)
    model = TSTCCModel(encoder=encoder_backbone, feature_dim=128, projection_dim=64).to(device)

    criterion = NTXentLoss(temperature=0.5)
    optimizer = Adam(model.parameters(), lr=MotionSenseConfig.LEARNING_RATE)

    # 3. Vòng lặp huấn luyện tự giám sát qua các Epochs
    for epoch in range(1, MotionSenseConfig.EPOCHS + 1):
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

        print(
            f"Epoch [{epoch:^3d}/{MotionSenseConfig.EPOCHS}] | 📉 SSL Contrastive Loss: {total_loss / len(ssl_loader):.5f}")

    # 4. Tách biệt giải phóng và lưu lại bộ trọng số Encoder sạch hữu ích
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "tstcc_encoder_pretrained_motionsense.pt")

    # Chỉ lưu trọng số nội bộ của lớp backbone encoder để pha sau nạp vào tinh chỉnh ít nhãn
    torch.save(model.encoder.state_dict(), save_path)
    print("=" * 80)
    print(f"✅ Hoàn tất tiền huấn luyện tự giám sát!")
    print(f"💾 Trọng số bộ mã hóa sạch (Pretrained Backbone) đã lưu tại: {save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()