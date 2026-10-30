"""
===============================================================================
CẤU HÌNH CHO BỘ DỮ LIỆU MOTIONSENSE
===============================================================================
Mục tiêu:
    - Kế thừa BaseDatasetConfig để có sẵn path, window, channels.
    - Chỉ khai báo phần RIÊNG của MotionSense: classes, subject split,
      tên cột thô (dùng cho script preprocess).
    - Đồng bộ LABEL MAP với UCI-HAR và HHAR để hỗ trợ Cross-Domain.

Thông tin bộ dữ liệu:
    - 24 subjects, 6 hoạt động: Walking, Upstairs, Downstairs, Sitting,
      Standing, Jogging.
    - Cảm biến: iPhone 6s gắn ở eo, tần số 50Hz.
    - Trong cross-domain, chỉ dùng 5 lớp đầu (bỏ Jogging) để khớp
      UCI-HAR và HHAR.
===============================================================================
"""

from config.datasets.base_dataset_config import BaseDatasetConfig


class MotionSenseConfig(BaseDatasetConfig):
    """
    Cấu hình cho MotionSense.

    Kế thừa từ BaseDatasetConfig:
        - WINDOW_SIZE=128, STRIDE=64, IN_CHANNELS=6
        - Path tự động: PROCESSED_DATA_DIR = data/processed/motion_sense
        - NUM_CLASSES tự động = len(CLASS_NAMES) = 6
    """

    # Tên thư mục con -> dùng để xây path tự động trong base.
    DATASET_NAME = "motion_sense"

    LABEL_MAPPING = {
        'wlk': 0,  # Walking
        'ups': 1,  # Upstairs
        'dws': 2,  # Downstairs
        'sit': 3,  # Sitting
        'std': 4,  # Standing

        'jog': 5  # Jogging # ⚠️ Chỉ có ở MotionSense (UCI-HAR không có)
    }

    # -------------------------------------------------------------------------
    # 1. TÊN LỚP
    # -------------------------------------------------------------------------
    # Thứ tự index chuẩn 0..5.
    # Lớp Jogging (5) chỉ có ở MotionSense, bị lọc bỏ trong cross-domain.
    CLASS_NAMES = [
        'Walking',
        'Upstairs',
        'Downstairs',
        'Sitting',
        'Standing',
        'Jogging',
    ]

    # -------------------------------------------------------------------------
    # 2. CHIA DỮ LIỆU THEO SUBJECT (CHỐNG RÒ RỈ DỮ LIỆU)
    # -------------------------------------------------------------------------
    # MotionSense có 24 subjects, chia theo tỉ lệ 14/4/6:
    #   - Train: subject 1..14  (~58.3%)
    #   - Val  : subject 15..18 (~16.7%)
    #   - Test : subject 19..24 (~25.0%)
    TRAIN_SUBJECTS = list(range(1, 15))   # 1, 2, ..., 14
    VAL_SUBJECTS = list(range(15, 19))    # 15, 16, 17, 18
    TEST_SUBJECTS = list(range(19, 25))   # 19, 20, ..., 24

    # -------------------------------------------------------------------------
    # 3. TÊN CỘT DỮ LIỆU THÔ (DÙNG CHO SCRIPT PREPROCESS)
    # -------------------------------------------------------------------------
    # File CSV gốc của MotionSense có các cột tên dài.
    # Script preprocess_motionsense.py sẽ đọc các cột này để tạo tensor 6 kênh.
    # Tên trong CHANNEL_NAMES là tên chuẩn hóa (dùng trong tensor .pt).
    RAW_COLS_USER_ACC = [
        'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
    ]
    RAW_COLS_GRAVITY = [
        'gravity.x', 'gravity.y', 'gravity.z',
    ]
    RAW_COLS_ROTATION = [
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
    ]

    # 6 kênh chuẩn: 3 total_acc (user_acc + gravity) + 3 rotation (gyro).
    CHANNEL_NAMES = [
        'total_acc_x', 'total_acc_y', 'total_acc_z',
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
    ]