"""
===============================================================================
MÔ HÌNH PHÂN LOẠI HOÀN CHỈNH (SUPERVISED HAR MODEL)
===============================================================================
Mục đích:
    Kết hợp giữa Encoder (Trích xuất đặc trưng) và ClassifierHead (Phân loại).
    Nhận Tensor cảm biến thô (B, 6, 128) -> Cho ra Dự đoán (B, 6)
===============================================================================
"""

import torch
import torch.nn as nn


class SupervisedHARModel(nn.Module):
    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        """
        Tham số:
            encoder (nn.Module): Khung xương trích xuất đặc trưng (vd: StandardSensorEncoder1D)
            classifier (nn.Module): Đầu phân loại (vd: ClassifierHead)
        """
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng dữ liệu:
            x (B, 6, 128) -> Encoder -> features (B, 128) -> Classifier -> logits (B, 6)
        """
        features = self.encoder(x)  # Nén từ (B, 6, 128) về (B, 128)
        logits = self.classifier(features)  # Chuyển từ (B, 128) về (B, 6)
        return logits


# =============================================================================
# SCRIPT CHẠY TEST GHÉP NỐI MÔ HÌNH
# =============================================================================
if __name__ == '__main__':
    # Import 2 thành phần từ các module đã code
    from models.encoders.cnn1d import StandardSensorEncoder1D
    from models.heads.classifier import ClassifierHead

    print("⏳ Đang kiểm tra ghép nối Mô hình SupervisedHARModel...")

    # 1. Giả lập Batch dữ liệu cảm biến: 64 cửa sổ, 6 kênh, 128 mẫu
    dummy_x = torch.randn(64, 6, 128)

    # 2. Khởi tạo 2 module riêng biệt
    encoder_block = StandardSensorEncoder1D(in_channels=6, feature_dim=128)
    classifier_block = ClassifierHead(feature_dim=128, num_classes=6)

    # 3. Lắp ráp thành mô hình hoàn chỉnh
    full_model = SupervisedHARModel(encoder=encoder_block, classifier=classifier_block)

    # 4. Chạy forward pass
    logits = full_model(dummy_x)

    print("✅ Ghép nối thành công!")
    print(f"   - Đầu vào Dữ liệu thô (Raw Sensors) : {dummy_x.shape}")
    print(f"   - Đầu ra Dự đoán (Logits 6 Lớp)    : {logits.shape}")
