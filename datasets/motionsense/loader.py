from torch.utils.data import DataLoader

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset


# =============================================================================
# HÀM TẠO DATALOADER HOÀN CHỈNH CHO CẢ TRAIN VÀ TEST
# =============================================================================
def get_motionsense_dataloaders(
        data_dir=MotionSenseConfig.RAW_DATA_DIR,
        config=MotionSenseConfig):
    """
    Hàm tiện ích trả về bộ đôi (train_loader, test_loader) sẵn sàng cho mô hình
    """
    print("⏳ Đang khởi tạo và nạp dữ liệu MotionSense...")

    # 1. Tạo Dataset cho tập TRAIN (Dùng Subjects 1 -> 14)
    train_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.TRAIN_SUBJECTS,
        config=config
    )

    # 2. Tạo Dataset cho tập VALIDATE (Dùng Subjects 15 -> 18)
    val_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.VAL_SUBJECTS,
        config=config
    )

    # 3. Tạo Dataset cho tập TEST (Dùng Subjects 19 -> 24)
    test_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.TEST_SUBJECTS,
        config=config
    )

    # 4. Tạo DataLoader cho TRAIN (Có xáo trộn dữ liệu - shuffle=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # 5. Tạo DataLoader cho VALIDATE (Có xáo trộn dữ liệu - shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Xáo trộn ngẫu nhiên các batch trong lúc train để mô hình không học theo thứ tự
        drop_last=True  # Bỏ qua batch cuối nếu không đủ 64 mẫu
    )

    # 6. Tạo DataLoader cho TEST (Không xáo trộn - shuffle=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    print(f"✅ Nạp dữ liệu hoàn tất!")
    print(
        f"   - Tập TRAIN ({len(config.TRAIN_SUBJECTS)} người): Thu được {len(train_dataset)} cửa sổ (Windows) | {len(train_loader)} Batches")
    print(
        f"   - Tập VAL   ({len(config.VAL_SUBJECTS)} người)  : {len(val_dataset)} windows | {len(val_loader)} batches")
    print(
        f"   - Tập TEST  ({len(config.TEST_SUBJECTS)} người) : Thu được {len(test_dataset)} cửa sổ (Windows) | {len(test_loader)} Batches")

    return train_loader, val_loader, test_loader


# =============================================================================
# SCRIPT CHẠY KHỞI TẠO VÀ TEST TRỰC TIẾP PIPELINE
# =============================================================================
if __name__ == '__main__':
    # Đoạn code này chỉ chạy khi bạn thực thi trực tiếp file này (python datasets/motionsense_loader.py)
    # Dùng để KIỂM TRA BẮT LỖI Data Pipeline trước khi bước sang làm Model.

    train_loader, val_loader, test_loader = get_motionsense_dataloaders()

    # Lấy thử 1 Batch dữ liệu từ train_loader ra để soi kích thước Tensor
    for images, labels in train_loader:
        print("\n--- SOI KÍCH THƯỚC TENSOR BATCH ĐẦU TIÊN ---")
        print(f"Shape dữ liệu đầu vào (X) : {images.shape}  -> Chuẩn shape: (Batch=64, Kênh=6, Mẫu=128)")
        print(f"Shape nhãn đầu ra    (Y) : {labels.shape}  -> Chuẩn shape: (Batch=64,)")
        print(f"Ví dụ 5 nhãn đầu tiên     : {labels[:5].numpy()}")
        break
