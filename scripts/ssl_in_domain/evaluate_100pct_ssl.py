"""
===============================================================================
SCRIPT RUNNER: scripts/ssl_in_domain/evaluate_100pct_ssl.py
MỤC ĐÍCH:
    - Đánh giá chất lượng biểu diễn SSL trên 100% dữ liệu có nhãn.
    - Hỗ trợ 2 chế độ:
        1. 'linear_probe': Đóng băng Encoder, chỉ huấn luyện Classifier Head.
        2. 'full_finetune': Mở khóa toàn bộ mạng để tinh chỉnh.
    - Đánh giá trên tập Test độc lập (Subjects 19-24).
===============================================================================
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.baseline.supervised_model import SupervisedHARModel
from training.evaluator import ModelEvaluator

# Chọn chế độ: 'linear_probe' hoặc 'full_finetune'
# EVAL_MODE = "linear_probe"  # Đổi thành 'full_finetune' khi muốn mở khóa toàn bộ
EVAL_MODE = "full_finetune"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🎯 ĐÁNH GIÁ TRẦN HIỆU NĂNG SSL TRÊN 100% DỮ LIỆU NHÃN (Chế độ: {EVAL_MODE.upper()})")
    print("=" * 80)

    # 1. Nạp 100% Dữ liệu Train, Val, Test chuẩn
    train_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TRAIN_SUBJECTS,
        config=MotionSenseConfig
    )
    val_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.VAL_SUBJECTS,
        config=MotionSenseConfig
    )
    test_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TEST_SUBJECTS,
        config=MotionSenseConfig
    )

    train_loader = DataLoader(train_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=False)

    # 2. Khởi tạo mô hình và nạp trọng số SSL
    encoder_backbone = StandardSensorEncoder1D(in_channels=MotionSenseConfig.IN_CHANNELS, feature_dim=128)
    pretrained_weights_path = os.path.join(PROJECT_ROOT, "checkpoints", "tstcc_encoder_pretrained_motionsense.pt")

    if os.path.exists(pretrained_weights_path):
        encoder_backbone.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        print(f"✅ Đã nạp thành công trọng số SSL từ: {pretrained_weights_path}")
    else:
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số tại: {pretrained_weights_path}")

    classifier_head = ClassifierHead(feature_dim=128, num_classes=MotionSenseConfig.NUM_CLASSES)
    model = SupervisedHARModel(encoder=encoder_backbone, classifier=classifier_head).to(device)

    # 3. Cấu hình Optimizer theo kịch bản đánh giá
    if EVAL_MODE == "linear_probe":
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 30
    else:  # full_finetune
        for param in model.parameters():
            param.requires_grad = True
        optimizer = Adam([
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-4)
        epochs = MotionSenseConfig.EPOCHS

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_weights_path = os.path.join(PROJECT_ROOT, "checkpoints", "temp_best_100pct_eval.pt")

    # 4. Vòng lặp huấn luyện
    print(f"\n⏳ Bắt đầu huấn luyện {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        if EVAL_MODE == "linear_probe":
            model.eval()
            model.classifier.train()
        else:
            model.train()

        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                val_loss += criterion(model(x_v), y_v).item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:^2d}/{epochs}] | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")

    # 5. Đánh giá trên tập Test độc lập với Checkpoint tốt nhất
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        os.remove(best_weights_path)

    evaluator = ModelEvaluator(
        class_names=["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"],
        device=device
    )
    print("\n" + "=" * 80)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET (100% Labels - {EVAL_MODE.upper()}):")
    print("=" * 80)
    evaluator.evaluate(model=model, test_loader=test_loader, title_prefix=f"SSL 100% {EVAL_MODE}")


if __name__ == "__main__":
    main()