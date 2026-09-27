"""
===============================================================================
CẤU HÌNH CHO BỘ DỮ LIỆU HHAR (CẬP NHẬT CHUẨN ĐÓNG GÓI)
===============================================================================
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


class HHARConfig:
    PROJECT_ROOT = PROJECT_ROOT

    # 1. Đường dẫn thư mục
    if IS_KAGGLE:
        BASE_INPUT = Path("/kaggle/input/datasets/nguyendung009/har-data")
        RAW_DATA_DIR = BASE_INPUT / "raw" / "hhar"
        PROCESSED_DIR = BASE_INPUT / "processed"
    else:
        RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "hhar"
        PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    PROCESSED_DIR_PHONE = PROCESSED_DIR / "hhar_phone"
    PROCESSED_DIR_WATCH = PROCESSED_DIR / "hhar_watch"
    PROCESSED_DIR_COMBINED = PROCESSED_DIR / "hhar_combined"

    # 2. Ánh xạ nhãn 5 lớp chuẩn
    LABEL_MAPPING = {
        'walk': 0,        # Walking
        'stairsup': 1,    # Upstairs
        'stairsdown': 2,  # Downstairs
        'sit': 3,         # Sitting
        'stand': 4,       # Standing
    }
    CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    NUM_CLASSES = len(CLASS_NAMES)  # 5 lớp

    # 3. Phân chia Subject (5 Train / 2 Val / 2 Test) - Mã hóa 'a'=0 .. 'i'=8
    TRAIN_SUBJECTS = [0, 1, 3, 4, 7]  # a, b, d, e, h
    VAL_SUBJECTS = [5, 6]  # f, g
    TEST_SUBJECTS = [2, 8]  # c, i

    # 4. Kênh cảm biến và cửa sổ trượt
    IN_CHANNELS = 6      # 3 total_acc + 3 gyro
    WINDOW_SIZE = 128    # 2.56 giây @ 50Hz
    STRIDE = 64          # Overlap 50%

    # 5. Tham số xử lý tín hiệu số (DSP)
    TARGET_FS = 50.0
    GAP_THRESHOLD_S = 1.0  # Cắt đoạn nếu khoảng cách thời gian > 1.0s
    LOWPASS_CUTOFF = 10.0  # Lọc thông thấp 10 Hz
    FILTER_ORDER = 4
    MIN_SAMPLES = 256      # Tối thiểu 256 mẫu (~5.12s) để có ít nhất 3 windows