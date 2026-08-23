"""
===============================================================================
MODULE: MOTIONSENSE DATALOADER
===============================================================================
    File này cung cấp hàm get_motionsense_dataloaders() để tạo và trả về
    3 DataLoader cho các tập Train, Validation và Test.

Luồng dữ liệu:
    CSV files (MotionSense) 
        -> MotionSenseDataset (đọc, tạo windows, gán nhãn)
            -> DataLoader (tạo batch, shuffle)
                -> train_loader, val_loader, test_loader

Chia subjects:
    - Train: subjects 1-14  (14 người) - Dùng để huấn luyện
    - Val:   subjects 15-18 (4 người)  - Dùng để chọn checkpoint, early stopping
    - Test:  subjects 19-24 (6 người)  - Cô lập hoàn toàn, chỉ đánh giá cuối cùng
===============================================================================
"""

from torch.utils.data import DataLoader

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset


def get_motionsense_dataloaders(
        data_dir: str = MotionSenseConfig.RAW_DATA_DIR,
        config: MotionSenseConfig = MotionSenseConfig) -> tuple:
    """
    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu MotionSense.
                         Mặc định lấy từ MotionSenseConfig.RAW_DATA_DIR.
        config (MotionSenseConfig): Đối tượng chứa cấu hình dự án (batch_size,
                                    window_size, subjects list, v.v.).

    Returns: tuple: (train_loader, val_loader, test_loader)
    """
    print("⏳ Đang khởi tạo và nạp dữ liệu MotionSense...")

    # ============================================================
    # PHẦN 1: TẠO DATASET CHO TỪNG TẬP (TRAIN, VAL, TEST)
    # ============================================================

    # 1. Dataset cho TRAIN: subjects 1 -> 14
    #    Dùng để huấn luyện model (cập nhật trọng số)
    train_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.TRAIN_SUBJECTS,  # [1, 2, ..., 14]
        config=config
    )

    # 2. Dataset cho VALIDATION: subjects 15 -> 18
    #    Dùng để chọn checkpoint tốt nhất, early stopping
    #    KHÔNG dùng để cập nhật trọng số
    val_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.VAL_SUBJECTS,  # [15, 16, 17, 18]
        config=config
    )

    # 3. Dataset cho TEST: subjects 19 -> 24
    #    Cô lập hoàn toàn, CHỈ dùng 1 lần duy nhất để đánh giá cuối cùng
    #    Không được nhìn thấy trong suốt quá trình train
    test_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.TEST_SUBJECTS,  # [19, 20, 21, 22, 23, 24]
        config=config
    )

    # ============================================================
    # PHẦN 2: TẠO DATALOADER TỪ CÁC DATASET
    # ============================================================

    # 4. DataLoader cho TRAIN
    #    - shuffle=True:  Xáo trộn dữ liệu mỗi epoch để model khái quát hóa tốt
    #    - drop_last=True: Bỏ batch cuối nếu không đủ batch_size, vì BatchNorm
    #                      cần batch đủ lớn để tính mean/std ổn định
    #    - batch_size:     Lấy từ config, thường là 64 hoặc 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Xáo trộn ngẫu nhiên các batch để mô hình không học theo thứ tự
        drop_last=True  # Bỏ qua batch cuối nếu không đủ 64 mẫu
    )

    # 5. DataLoader cho VALIDATION
    #    - shuffle=False:  Không xáo trộn vì chỉ đánh giá, không train
    #    - drop_last:      Mặc định False, giữ tất cả dữ liệu để đánh giá chính xác
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False  # Không xáo trộn để kết quả đánh giá nhất quán
    )

    # 6. DataLoader cho TEST
    #    - shuffle=False:  Không xáo trộn vì chỉ đánh giá cuối cùng
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False  # Không xáo trộn để kết quả đánh giá cuối cùng ổn định
    )

    # _print_info(
    #     config,
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     train_loader,
    #     val_loader,
    #     test_loader
    # )
    return train_loader, val_loader, test_loader  # TRẢ VỀ 3 DATALOADER


def _print_info(config,
                train_dataset,
                val_dataset,
                test_dataset,
                train_loader,
                val_loader,
                test_loader):
    # ============================================================
    # PHẦN 3: IN THÔNG TIN KIỂM TRA
    # ============================================================

    print(f"✅ Nạp dữ liệu hoàn tất!")
    print(
        f"   - Tập TRAIN ({len(config.TRAIN_SUBJECTS)} người): "
        f"Thu được {len(train_dataset)} cửa sổ (Windows) | "
        f"{len(train_loader)} Batches"
    )
    print(
        f"   - Tập VAL   ({len(config.VAL_SUBJECTS)} người)  : "
        f"{len(val_dataset)} windows | "
        f"{len(val_loader)} batches"
    )
    print(
        f"   - Tập TEST  ({len(config.TEST_SUBJECTS)} người) : "
        f"Thu được {len(test_dataset)} cửa sổ (Windows) | "
        f"{len(test_loader)} Batches"
    )


"""
from torch.utils.data import DataLoader

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset


def get_motionsense_dataloaders(
        data_dir: str = MotionSenseConfig.RAW_DATA_DIR,
        config: MotionSenseConfig = MotionSenseConfig) -> tuple:

    train_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.TRAIN_SUBJECTS, 
        config=config
    )

    val_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.VAL_SUBJECTS,  
        config=config
    )

    test_dataset = MotionSenseDataset(
        data_dir=data_dir,
        subjects_list=config.TEST_SUBJECTS,  
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  
        drop_last=True 
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False  
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False 
    )

    return train_loader, val_loader, test_loader 
"""