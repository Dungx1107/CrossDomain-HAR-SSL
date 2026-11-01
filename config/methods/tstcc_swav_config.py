"""
Đặt file tại: config/methods/tstcc_swav_config.py

Chỉ khai báo phần MỚI (SwAV + hybrid). Các tham số của TC (TIMESTEPS, TC_HIDDEN_DIM,
TC_DEPTH, TC_HEADS, TC_MLP_DIM, GRAD_CLIP_MAX_NORM) dùng lại từ TSTCCConfig.
"""


class TSTCCSwAVConfig:
    # --- SwAV ---
    NUM_PROTOTYPES = 45
    TAU = 0.1               # nhiệt độ softmax của prediction
    EPSILON = 0.05          # nhiệt độ Sinkhorn (áp lên cosine thô)
    SINKHORN_ITERS = 3

    # --- Trọng số loss ---
    LAMBDA_TC = 1.0
    LAMBDA_SWAV = 1.0
    LAMBDA_CC = 0.0         # > 0 (vd 0.7) để bật NT-Xent cho ablation
    CC_TEMPERATURE = 0.2

    # --- TC ---
    MIN_T = 0               # tăng lên (3-4) nếu context quá ngắn làm loss dao động

    # --- Huấn luyện ---
    WARMUP_EPOCHS = 5       # không lưu "best" trong các epoch đầu (SwAV chưa ổn định)