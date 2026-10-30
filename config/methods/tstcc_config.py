class TSTCCConfig:
    # 1. Temporal Contrasting
    TIMESTEPS = 6  # Số bước thời gian tương lai cần dự đoán.
    TC_HIDDEN_DIM = 64  # Số chiều hidden state của mạng Temporal Contrasting.
    TC_DEPTH = 4  # Số lớp Transformer Encoder trong mạng Temporal Contrasting.
    TC_HEADS = 4  # Số attention heads trong mỗi lớp Transformer.
    TC_MLP_DIM = 64  # Số chiều tầng ẩn của Feed-Forward Network trong Transformer.

    # 2. Contrastive Learning
    LAMBDA1 = 1.0  # Trọng số của Temporal Contrasting Loss.
    LAMBDA2 = 0.7  # Trọng số của Contextual Contrasting Loss.
    TEMPERATURE = 0.2  # Hệ số nhiệt độ trong NT-Xent Loss, điều chỉnh độ sắc nét của phân phối xác suất.

    # 3. Encoder Architecture
    KERNEL_SIZE = 8  # Kích thước kernel trong các lớp Convolution 1D.
    DROPOUT = 0.35  # Tỷ lệ neuron bị vô hiệu hóa ngẫu nhiên để giảm overfitting.

    # 4. Training Configuration
    EPOCHS = 60  # Số lần mô hình duyệt qua toàn bộ tập huấn luyện.
    BATCH_SIZE = 64  # Số mẫu được xử lý trong mỗi batch.
    LEARNING_RATE = 3e-4  # Tốc độ học, điều chỉnh độ lớn bước cập nhật trọng số.
    WEIGHT_DECAY = 3e-4  # Hệ số điều chuẩn L2 giúp hạn chế overfitting.
    GRAD_CLIP_MAX_NORM = 2.0  # Clip norm gradient chống bùng nổ gradient, giúp train ổn định
# =============================================================================
# CÁCH DÙNG
# =============================================================================
#
# Trong engines/pretrain_ssl/tstcc_trainer.py:
#
#     from config.methods.tstcc_config import TSTCCConfig
#
#     ssl_model = TSTCCModel(
#         encoder=build_encoder(backbone_type=backbone_type, in_channels=6),
#         in_channels=6,
#         feature_dim=128,
#         timesteps=TSTCCConfig.TIMESTEPS,
#         tc_hidden_dim=TSTCCConfig.TC_HIDDEN_DIM,
#         tc_depth=TSTCCConfig.TC_DEPTH,
#         tc_heads=TSTCCConfig.TC_HEADS,
#         tc_mlp_dim=TSTCCConfig.TC_MLP_DIM,
#         lambda1=TSTCCConfig.LAMBDA1,
#         lambda2=TSTCCConfig.LAMBDA2,
#         temperature=TSTCCConfig.TEMPERATURE,
#     )
#
#     optimizer = AdamW(
#         ssl_model.parameters(),
#         lr=TSTCCConfig.LEARNING_RATE,
#         weight_decay=TSTCCConfig.WEIGHT_DECAY,
#     )
#
# LƯU Ý PHÂN BIỆT PRETRAIN vs DOWNSTREAM:
#     - File này là config cho SSL PRETRAINING.
#     - Downstream (Linear Probing / Full FT) có hyperparameter riêng
#       trong engines/transfer/finetune_trainer.py:
#           backbone_lr = 1e-4, head_lr = 1e-3,
#           epochs = 40, batch_size = 16.
#       KHÔNG trộn lẫn hai bộ số này.
#
# NGUỒN THAM CHIẾU:
#     - Bài báo: Eldele et al., IJCAI 2021, mục 4.2 Implementation Details.
#     - Code gốc: emadeldeen24/TS-TCC.
#     - Code dự án: engines/pretrain_ssl/contrastive_trainer.py.
# =============================================================================
