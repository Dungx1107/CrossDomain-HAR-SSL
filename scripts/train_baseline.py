"""
===============================================================================
SCRIPT HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH BASELINE SUPERVISED (MOTIONSENSE)
===============================================================================
Mục đích:
    - Nạp dữ liệu từ MotionSense DataLoader (Step 1).
    - Khởi tạo Mô hình SupervisedHARModel (Step 2 + Step 3).
    - Thực hiện vòng lặp Huấn luyện (Training Loop) & Đánh giá (Evaluation Loop).
    - Lưu Trọng số Mô hình tốt nhất (Best Model Checkpoint) vào thư mục checkpoints/.
===============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Import cấu hình và Data Loader (Step 1)
from config.motionsense_config import MotionSenseConfig
from datasets.motionsense import get_motionsense_dataloaders

# 2. Import Encoder, Head và Model (Step 2 & Step 3)
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.baseline.supervised_model import SupervisedHARModel


def train_one_epoch(model,
                    dataloader,
                    criterion,
                    optimizer,
                    device):
    """
    Hàm thực hiện 1 Epoch Huấn luyện (Duyệt qua toàn bộ tập Train)
    """
    model.train()  # Bật chế độ Train (Dropout bật, BatchNorm cập nhật)
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for inputs, targets in dataloader:
        # Đẩy Tensor lên thiết bị tính toán (GPU hoặc CPU)
        # inputs shape: (B, 6, 128) | targets shape: (B,)
        inputs, targets = inputs.to(device), targets.to(device)

        # Step A: Xóa Gradient cũ
        optimizer.zero_grad()

        # Step B: Forward pass (Lan truyền tiến -> Ra Logits (B, 6))
        outputs = model(inputs)

        # Step C: Tính Loss (CrossEntropy)
        loss = criterion(outputs, targets)

        # Step D: Backward pass (Tính Đạo hàm Gradient)
        loss.backward()

        # Step E: Cập nhật Trọng số
        optimizer.step()

        # Tích lũy Thống kê
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)  # Lấy vị trí có xác suất lớn nhất
        correct_preds += torch.sum(preds == targets.data)
        total_samples += inputs.size(0)

    # Tính Loss và Accuracy trung bình của cả Epoch Train
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_preds.double() / total_samples).item() * 100.0

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Hàm thực hiện Đánh giá trên tập Test (Không cập nhật trọng số)
    """
    model.eval()  # Bật chế độ Evaluation (Khóa Dropout, khóa BatchNorm)
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # Tắt tự động tính Gradient để tiết kiệm RAM/VRAM
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct_preds += torch.sum(preds == targets.data)
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_preds.double() / total_samples).item() * 100.0

    return epoch_loss, epoch_acc


def main():
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN SUPERVISED BASELINE (1D-CNN)...")

    # -------------------------------------------------------------------------
    # 1. THIẾT LẬP THIẾT BỊ TÍNH TOÁN (GPU / CPU)
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Thiết bị tính toán đang sử dụng: {device}")

    # -------------------------------------------------------------------------
    # 2. NẠP DỮ LIỆU MOTIONSENSE
    # -------------------------------------------------------------------------
    train_loader, test_loader = get_motionsense_dataloaders()

    # -------------------------------------------------------------------------
    # 3. KHỞI TẠO MÔ HÌNH BASELINE
    # -------------------------------------------------------------------------
    encoder = StandardSensorEncoder1D(
        in_channels=MotionSenseConfig.IN_CHANNELS,
        feature_dim=128
    )
    classifier = ClassifierHead(
        feature_dim=128,
        num_classes=MotionSenseConfig.NUM_CLASSES,
        dropout_rate=0.2
    )
    model = SupervisedHARModel(encoder=encoder, classifier=classifier).to(device)

    # -------------------------------------------------------------------------
    # 4. KHỞI TẠO HÀM LOSS VÀ OPTIMIZER
    # -------------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MotionSenseConfig.LEARNING_RATE)

    # -------------------------------------------------------------------------
    # 5. VÒNG LẶP HUẤN LUYỆN QUA CÁC EPOCHS
    # -------------------------------------------------------------------------
    epochs = MotionSenseConfig.EPOCHS
    best_test_acc = 0.0

    # Tạo thư mục checkpoints để lưu file trọng số tốt nhất
    DATASET_NAME = "motionsense"
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"baseline_cnn1d_{DATASET_NAME}_best.pt")

    print(f"\n🔄 Bắt đầu chạy {epochs} Epochs...\n")
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc (%)':^14} | {'Test Loss':^12} | {'Test Acc (%)':^14}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        # 1. Chạy Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 2. Chạy Test
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # In bảng theo dõi kết quả
        print(f"{epoch:^8d} | {train_loss:^12.4f} | {train_acc:^14.2f} | {test_loss:^12.4f} | {test_acc:^14.2f}",
              end="")

        # 3. Lưu Trọng số nếu đạt Test Accuracy cao nhất
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(" ⭐ [Lưu Model Best!]")
        else:
            print("")

    print("-" * 68)
    print(f"🎉 Huấn luyện hoàn tất!")
    print(f"🏆 Đạt Test Accuracy cao nhất: {best_test_acc:.2f}%")
    print(f"📁 Trọng số tốt nhất đã được lưu tại: {save_path}")


if __name__ == '__main__':
    main()
