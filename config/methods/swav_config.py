"""
===============================================================================
MODULE: SWAV CONFIGURATION (PROTOTYPICAL LEARNING)
===============================================================================
LƯU Ý: Đây là config chuyên dụng cho quá trình SSL Pretraining.
"""


class SwAVConfig:
    # 1. SwAV Hyperparameters
    FEATURE_DIM = 128  # Kích thước vector đặc trưng (z)
    PROJECTION_DIM = 64  # Kích thước không gian chiếu (projection p)
    NUM_PROTOTYPES = 45  # Số lượng tâm cụm (K) - đại diện cho các mẫu hành vi
    TAU_S = 0.1  # Softmax temperature (nhiệt độ của dự đoán phân bố)
    EPSILON = 0.05  # Sinkhorn temperature (điều tiết thuật toán Sinkhorn-Knopp)

    # 2. Training Configuration
    EPOCHS = 40
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5  # Số epoch warmup learning rate ban đầu
    FREEZE_PROTOTYPES_EPOCHS = 1  # Số epoch đầu KHÔNG cập nhật prototype (SwAV gốc: 313 iter ≈ 1 epoch)
    SINKHORN_ITERS = 3  # số vòng Sinkhorn (bài báo gốc dùng 3)
    # Cấu hình phụ trợ
    MAX_ENTROPY_COLLAPSE_THRESHOLD = 0.3  # Ngưỡng (30% max entropy) để cảnh báo sụp cụm