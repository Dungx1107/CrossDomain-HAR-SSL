import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset
from utils.sampling import create_few_label_subset  # Bộ băm nhãn phân tầng của bạn
from utils.logger import SimpleLogger
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.baseline.supervised_model import SupervisedHARModel
from training.evaluator import ModelEvaluator


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🎯 BẮT ĐẦU PHA ĐÁNH GIÁ FINE-TUNE ÍT NHÃN SAU HUẤN LUYỆN SSL")
    print("=" * 80)

    # 1. Khởi tạo tập dữ liệu và Loader kiểm thử giữ nguyên 100% không gian
    test_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TEST_SUBJECTS,
        config=MotionSenseConfig
    )
    test_loader = DataLoader(test_dataset, batch_size=MotionSenseConfig.BATCH_SIZE, shuffle=False)

    # 2. Tạo tập huấn luyện giả lập thiếu nhãn kịch sàn (Giữ lại đúng RATIO = 0.001 ~ 0.1%)
    full_train_dataset = MotionSenseDataset(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        subjects_list=MotionSenseConfig.TRAIN_SUBJECTS,
        config=MotionSenseConfig
    )
    few_label_train_dataset, _ = create_few_label_subset(full_train_dataset, ratio=0.001, seed=42)
    train_loader = DataLoader(few_label_train_dataset, batch_size=len(few_label_train_dataset), shuffle=True)

    # 3. Khởi tạo mô hình mạng và NẠP TRỌNG SỐ SẠCH ĐÃ HỌC TỪ PHA TIỀN HUẤN LUYỆN SSL
    encoder_backbone = StandardSensorEncoder1D(in_channels=MotionSenseConfig.IN_CHANNELS, feature_dim=128)

    pretrained_weights_path = os.path.join(PROJECT_ROOT, "checkpoints", "tstcc_encoder_pretrained_motionsense.pt")
    if os.path.exists(pretrained_weights_path):
        encoder_backbone.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        print(f"✅ Đã nạp thành công trọng số tự giám sát từ: {pretrained_weights_path}")
    else:
        print(f"⚠️ Không tìm thấy file trọng số SSL tại {pretrained_weights_path}. Mạng sẽ chạy ngẫu nhiên!")

    # Gắn đầu phân loại để tạo mô hình hoàn chỉnh
    classifier_head = ClassifierHead(feature_dim=128, num_classes=MotionSenseConfig.NUM_CLASSES)
    model = SupervisedHARModel(encoder=encoder_backbone, classifier=classifier_head).to(device)

    # 💡 LÝ THUYẾT QUAN TRỌNG: Đóng băng (Freeze) toàn bộ trọng số lớp backbone encoder
    # Chỉ cho phép Gradients cập nhật trọng số của Classifier Head để kiểm chứng đặc trưng SSL
    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=1e-3)  # Chỉ tối ưu lớp classifier

    # 4. Huấn luyện nhanh đầu phân loại trên tập con 0.1% nhãn
    print(f"⏳ Đang tinh chỉnh đầu phân loại trên {len(few_label_train_dataset)} mẫu có nhãn...")
    model.train()
    for epoch in range(1, 15):  # Chỉ cần 10-15 epochs vì đặc trưng ẩn đã rất tối ưu
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

    # 5. Chạy Engine đánh giá chuyên trách để xuất Ma trận nhầm lẫn và so đối chứng điểm số
    evaluator = ModelEvaluator(class_names=["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"],
                               device=device)
    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ ĐỐI CHỨNG HIỆU NĂNG SAU KHI QUA KHUNG TỰ GIÁM SÁT SSL:")
    print("=" * 80)
    evaluator.evaluate(model=model, test_loader=test_loader, title_prefix="TSTCC + Few-Label Fine-tune 0.1%")


if __name__ == "__main__":
    main()