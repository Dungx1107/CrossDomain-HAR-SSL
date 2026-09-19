"""
===============================================================================
CẤU HÌNH CHO BỘ DỮ LIỆU UCI-HAR
===============================================================================
Mục tiêu:
    - Quản lý cấu hình, đường dẫn và siêu tham số cho bộ dữ liệu UCI-HAR.
    - Đồng bộ LABEL MAP với MotionSense để hỗ trợ Cross-Domain.
===============================================================================
"""

import os
from pathlib import Path

# 1. XÁC ĐỊNH GỐC DỰ ÁN & MÔI TRƯỜNG CHẠY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


class UCIHARConfig:
    # 2. ĐƯỜNG DẪN DỮ LIỆU & CHECKPOINT
    if IS_KAGGLE:
        BASE_INPUT = Path("/kaggle/input")
        DATA_DIR = BASE_INPUT / "har-processed-data" / "uci_har"
        CHECKPOINT_SSL_PRETRAINED_PATH = BASE_INPUT / "har-ssl-checkpoints-vault" / "tstcc_encoder_pretrained_uci_har.pt"
        REPORT_DIR = Path("/kaggle/working/reports/uci_har")
    else:
        RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "uci_har"
        DATA_DIR = PROJECT_ROOT / "data" / "processed" / "uci_har"
        CHECKPOINT_SSL_PRETRAINED_PATH = PROJECT_ROOT / "checkpoints" / "tstcc_encoder_pretrained_uci_har.pt"
        REPORT_DIR = PROJECT_ROOT / "document" / "0_reports" / "1_baseline_uci_har"

    # Đường dẫn file dữ liệu đã xử lý
    DATA_ALL_PATH = DATA_DIR / "dataset_all.pt"
    PROCESSED_TRAIN_PATH = DATA_DIR / "train.pt"
    PROCESSED_VAL_PATH = DATA_DIR / "val.pt"
    PROCESSED_TEST_PATH = DATA_DIR / "test.pt"

    # ========================================================================
    # 3. ÁNH XẠ NHÃN (LABEL MAP) - ĐỒNG BỘ VỚI MOTIONSENSE
    # ========================================================================
    # ⚠️ QUAN TRỌNG: Các lớp chung phải có cùng index với MotionSense
    # - 5 lớp chung: Walking, Upstairs, Downstairs, Sitting, Standing
    # - 1 lớp riêng: Laying (chỉ có ở UCI-HAR)

    # Ánh xạ từ file txt (1-6) sang index (0-5)
    LABEL_MAPPING = {
        1: 0,  # WALKING     ✅ Giống MotionSense (wlk: 0)
        2: 1,  # WALKING_UPSTAIRS ✅ Giống MotionSense (ups: 1)
        3: 2,  # WALKING_DOWNSTAIRS ✅ Giống MotionSense (dws: 2)
        4: 3,  # SITTING     ✅ Giống MotionSense (sit: 3)
        5: 4,  # STANDING    ✅ Giống MotionSense (std: 4)

        6: 5,  # LAYING      ⚠️ Chỉ có ở UCI-HAR
    }

    # Tên lớp theo thứ tự index (dùng cho hiển thị)
    CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying']
    NUM_CLASSES = len(CLASS_NAMES)  # 6 lớp

    # ========================================================================
    # 4. CHIA DỮ LIỆU THEO SUBJECT (CHỐNG RÒ RỈ DỮ LIỆU)
    # ========================================================================
    # 4 subjects dùng làm Validation tách từ tập Train gốc
    VAL_SUBJECTS = [27, 28, 29, 30]

    # ========================================================================
    # 5. ĐẶC TẢ TÍN HIỆU
    # ========================================================================
    # 6 kênh tín hiệu: 3 gia tốc cơ thể + 3 con quay
    SIGNAL_NAMES = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z"
    ]
    IN_CHANNELS = len(SIGNAL_NAMES)  # 6 kênh

    # 6. THAM SỐ CỬA SỔ (ĐỒNG BỘ VỚI MOTIONSENSE)
    WINDOW_SIZE = 128   # 2.56 giây @ 50Hz
    STRIDE = 64         # Overlap 50%