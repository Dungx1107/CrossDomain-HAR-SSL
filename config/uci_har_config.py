"""
===============================================================================
CẤU HÌNH THÍ NGHIỆM: UCI-HAR SUPERVISED BASELINE
Vị trí file: config/uci_har_config.py
Mục tiêu:
    - Quản lý siêu tham số huấn luyện Supervised từ đầu (Scratch) trên UCI-HAR.
    - Cấu hình 6 kênh tín hiệu chuẩn hóa và 5 lớp hoạt động chung.
===============================================================================
"""

import os


class UCIHARConfig:
    # -------------------------------------------------------------------------
    # 1. ĐƯỜNG DẪN DỮ LIỆU & CHECKPOINT
    # -------------------------------------------------------------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Dữ liệu đã qua tiền xử lý (5 lớp chung, 6 kênh)
    PROCESSED_TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "uci_har", "train.pt")
    PROCESSED_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "uci_har", "test.pt")

    # Nơi lưu checkpoint tốt nhất
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "baseline_cnn1d_uci_har_best.pt")

    # Nơi lưu báo cáo kết quả
    REPORT_DIR = os.path.join(PROJECT_ROOT, "document", "0_reports", "1_baseline_uci_har")
    LOG_FILE_PATH = os.path.join(REPORT_DIR, "baseline_uci_har_eval_logs.txt")

    # -------------------------------------------------------------------------
    # 2. ĐẶC TẢ TÍN HIỆU & KHÔNG GIAN NHÃN
    # -------------------------------------------------------------------------
    IN_CHANNELS = 6  # 6 trục: [body_acc_x, y, z, body_gyro_x, y, z]
    SEQUENCE_LENGTH = 128  # Chiều dài mỗi cửa sổ thời gian
    NUM_CLASSES = 5  # 5 lớp hoạt động: Walking, Upstairs, Downstairs, Sitting, Standing
    FEATURE_DIM = 128  # Kích thước Feature Vector đầu ra từ Encoder Backbone

    CLASS_NAMES = [
        'Walking',
        'Upstairs',
        'Downstairs',
        'Sitting',
        'Standing'
    ]

    # -------------------------------------------------------------------------
    # 3. SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.3

    # Learning Rate Scheduler (Giảm LR khi Loss chững lại)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-5

    # Early Stopping để tránh Overfitting
    EARLY_STOPPING_PATIENCE = 12