"""
===============================================================================
CẤU HÌNH CHO BỘ DỮ LIỆU MOTIONSENSE
===============================================================================
Mục tiêu:
    - Quản lý cấu hình, đường dẫn và siêu tham số cho bộ dữ liệu MotionSense.
    - Đồng bộ LABEL MAP với UCI-HAR để hỗ trợ Cross-Domain.
===============================================================================
"""

import os
from pathlib import Path

# 1. XÁC ĐỊNH GỐC DỰ ÁN & MÔI TRƯỜNG CHẠY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


class MotionSenseConfig:
    # 2. ĐƯỜNG DẪN DỮ LIỆU & CHECKPOINT
    if IS_KAGGLE:
        BASE_INPUT = Path("/kaggle/input/datasets/nguyendung009/har-data/processed")
        PROCESSED_DIR = BASE_INPUT / "motionsense"
    else:
        RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "motion_sense"
        PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "motionsense"

    # Đường dẫn file dữ liệu đã xử lý
    DATA_ALL_PATH = PROCESSED_DIR / "dataset_all.pt"
    PROCESSED_TRAIN_PATH = PROCESSED_DIR / "train.pt"
    PROCESSED_VAL_PATH = PROCESSED_DIR / "val.pt"
    PROCESSED_TEST_PATH = PROCESSED_DIR / "test.pt"

    # Tên lớp theo thứ tự index
    CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Jogging']
    NUM_CLASSES = len(CLASS_NAMES)  # 6 lớp

    # 4. CHIA DỮ LIỆU THEO SUBJECT (CHỐNG RÒ RỈ DỮ LIỆU)
    TRAIN_SUBJECTS = list(range(1, 15))  # Sub 1 -> 14 (14 người ~ 58.3%)
    VAL_SUBJECTS = list(range(15, 19))  # Sub 15 -> 18 (4 người ~ 16.7%)
    TEST_SUBJECTS = list(range(19, 25))  # Sub 19 -> 24 (6 người ~ 25.0%)

    # 5. THAM SỐ CỬA SỔ TRƯỢT (SLIDING WINDOW)
    WINDOW_SIZE = 128  # 128 samples ~ 2.56 giây @ 50Hz
    STRIDE = 64  # Overlap 50%

    # 6. KÊNH CẢM BIẾN
    RAW_COLS_USER_ACC = [
        'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
    ]
    RAW_COLS_GRAVITY = [
        'gravity.x', 'gravity.y', 'gravity.z',
    ]
    RAW_COLS_ROTATION = [
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
    ]

    CHANNEL_NAMES = [
        'total_acc_x', 'total_acc_y', 'total_acc_z',
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z'
    ]
    IN_CHANNELS = len(CHANNEL_NAMES)  # 6 kênh = 3 (total_acc) + 3 (gyro)

