"""
===============================================================================
Mục tiêu:
    - Quản lý cấu hình, đường dẫn và siêu tham số cho bộ dữ liệu UCI-HAR.
    - Hỗ trợ tự động chuyển đổi môi trường giữa Máy cá nhân (Local) và Kaggle.
    - Cung cấp checkpoint chuẩn cho các tác vụ Pretrain SSL và Cross-Domain.
===============================================================================
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. XÁC ĐỊNH GỐC DỰ ÁN & MÔI TRƯỜNG CHẠY
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

class UCIHARConfig:
    # -------------------------------------------------------------------------
    # 2. ĐƯỜNG DẪN DỮ LIỆU & CHECKPOINT
    # -------------------------------------------------------------------------
    if IS_KAGGLE:
        BASE_INPUT = Path("/kaggle/input")
        DATA_DIR = BASE_INPUT / "har-processed-data" / "uci_har"
        CHECKPOINT_SSL_PRETRAINED_PATH = BASE_INPUT / "har-ssl-checkpoints-vault" / "tstcc_encoder_pretrained_uci_har.pt"
        REPORT_DIR = Path("/kaggle/working/reports/uci_har")
    else:
        DATA_DIR = PROJECT_ROOT / "data" / "processed" / "uci_har"
        CHECKPOINT_SSL_PRETRAINED_PATH = PROJECT_ROOT / "checkpoints" / "tstcc_encoder_pretrained_uci_har.pt"
        REPORT_DIR = PROJECT_ROOT / "document" / "0_reports" / "1_baseline_uci_har"

    # Dữ liệu đã qua tiền xử lý
    DATA_ALL_PATH = DATA_DIR / "dataset_all.pt"
    PROCESSED_TRAIN_PATH = DATA_DIR / "train.pt"
    PROCESSED_TEST_PATH = DATA_DIR / "test.pt"

    # -------------------------------------------------------------------------
    # 3. ĐẶC TẢ TÍN HIỆU & KHÔNG GIAN NHÃN CHUNG (COMMON ACTION SPACE)
    # -------------------------------------------------------------------------
    IN_CHANNELS = 6         # [body_acc_x, y, z, body_gyro_x, y, z]
    SEQUENCE_LENGTH = 128   # 128 timesteps (~2.56s)
    FEATURE_DIM = 128       # Chiều vector đặc trưng sau Encoder Backbone

    # Lưu ý: 5 lớp hành vi giao nhau giữa UCI-HAR và MotionSense
    CLASS_NAMES = [
        'Walking',
        'Upstairs',
        'Downstairs',
        'Sitting',
        'Standing'
    ]
    NUM_CLASSES = len(CLASS_NAMES)  # 5 lớp

    # Ánh xạ nhãn nội bộ để đảm bảo thứ tự index khớp nhau
    LABEL_MAP = {
        'Walking': 0,
        'Upstairs': 1,
        'Downstairs': 2,
        'Sitting': 3,
        'Standing': 4
    }

    # -------------------------------------------------------------------------
    # 4. SIÊU THAM SỐ HUẤN LUYỆN
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.3

    # Scheduler & Early Stopping
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-5
    EARLY_STOPPING_PATIENCE = 12