import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. TỰ ĐỘNG XÁC ĐỊNH GỐC DỰ ÁN & MÔI TRƯỜNG
# -----------------------------------------------------------------------------
# File nằm tại: CrossDomain-HAR-SSL/config/motionsense_config.py
# parent 1 lần = 'config', parent 2 lần = 'CrossDomain-HAR-SSL'
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tự động nhận diện môi trường Kaggle hay Local
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


class MotionSenseConfig:
    # -------------------------------------------------------------------------
    # 2. ĐƯỜNG DẪN DỮ LIỆU (DATA PATHS)
    # -------------------------------------------------------------------------
    if IS_KAGGLE:
        # Đường dẫn khi chạy trên Kaggle
        BASE_INPUT = Path("/kaggle/input")
        RAW_DATA_DIR = BASE_INPUT / "motionsense-raw-data"
        PROCESSED_DIR = BASE_INPUT / "har-processed-data" / "motionsense"
        CHECKPOINT_SSL_PRETRAINED_PATH = BASE_INPUT / "har-ssl-checkpoints-vault" / "tstcc_encoder_pretrained_motionsense.pt"
    else:
        # Đường dẫn khi chạy trên máy Local
        RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "motion_sense"
        PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "motionsense"
        CHECKPOINT_SSL_PRETRAINED_PATH = PROJECT_ROOT / "checkpoints" / "tstcc_encoder_pretrained_motionsense.pt"

    # Đường dẫn file dữ liệu đã xử lý
    DATA_ALL_PATH = PROCESSED_DIR / "dataset_all.pt"  # Dùng cho K-Fold / Clustering
    PROCESSED_TRAIN_PATH = PROCESSED_DIR / "train.pt"  # Dùng cho Subject Split
    PROCESSED_VAL_PATH = PROCESSED_DIR / "val.pt"
    PROCESSED_TEST_PATH = PROCESSED_DIR / "test.pt"

    # -------------------------------------------------------------------------
    # 3. ÁNH XẠ NHÃN HOẠT ĐỘNG (LABEL MAPPING)
    # -------------------------------------------------------------------------
    LABEL_MAP = {
        'dws': 0,  # Downstairs (Đi xuống cầu thang)
        'ups': 1,  # Upstairs (Đi lên cầu thang)
        'wlk': 2,  # Walking (Đi bộ)
        'jog': 3,  # Jogging (Chạy bộ)
        'sit': 4,  # Sitting (Ngồi)
        'std': 5  # Standing (Đứng)
    }

    CLASS_NAMES = ['Downstairs', 'Upstairs', 'Walking', 'Jogging', 'Sitting', 'Standing']
    NUM_CLASSES = len(CLASS_NAMES)  # 6 classes

    # -------------------------------------------------------------------------
    # 4. CHIA DỮ LIỆU THEO SUBJECT (CHỐNG RÒ RỈ DỮ LIỆU - DATA LEAKAGE)
    # -------------------------------------------------------------------------
    # Tổng cộng có 24 subjects (sub_1 đến sub_24)
    TRAIN_SUBJECTS = list(range(1, 15))  # Sub 1 -> 14 (14 người ~ 58.3%)
    VAL_SUBJECTS = list(range(15, 19))  # Sub 15 -> 18 (4 người ~ 16.7%)
    TEST_SUBJECTS = list(range(19, 25))  # Sub 19 -> 24 (6 người ~ 25.0%)

    # -------------------------------------------------------------------------
    # 5. THAM SỐ CỬA SỔ TRƯỢT (SLIDING WINDOW)
    # -------------------------------------------------------------------------
    # Sampling rate: 50Hz (50 samples/s)
    # 128 samples tương đương 2.56 giây
    WINDOW_SIZE = 128

    # Overlap 50% -> Bước trượt là 64 samples
    STRIDE = 64

    # -------------------------------------------------------------------------
    # 6. KÊNH CẢM BIẾN (FEATURE COLUMNS)
    # -------------------------------------------------------------------------
    # 6 kênh cơ bản tương ứng với chuẩn của UCI-HAR (3 gia tốc + 3 vận tốc góc)
    FEATURE_COLS = [
        'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z'
    ]
    IN_CHANNELS = len(FEATURE_COLS)  # 6 kênh

    # -------------------------------------------------------------------------
    # 7. SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 30