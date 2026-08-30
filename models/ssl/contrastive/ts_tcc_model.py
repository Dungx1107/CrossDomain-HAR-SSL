"""
===============================================================================
VAI TRÒ TRONG HỆ THỐNG:
    - Đóng gói toàn bộ kiến trúc mạng Siamese (Siamese Network Architecture)
      cho giai đoạn Huấn luyện Tự Giám Sát Đối Chiếu (TS-TCC Pre-training).
    - Kết hợp 2 thành phần độc lập theo nguyên lý Lego:
        1. Backbone (StandardSensorEncoder1D): Trích xuất đặc trưng chuỗi thời gian.
        2. Head (ProjectionHead): Chiếu vector sang không gian tính Contrastive Loss.

NGUYÊN LÝ HOẠT ĐỘNG:
    - Nhận vào 2 góc nhìn tăng cường (Augmented Views) từ cùng 1 tín hiệu gốc:
        + x_weak   : Góc nhìn tăng cường yếu (Jitter + Scaling nhẹ).
        + x_strong : Góc nhìn tăng cường mạnh (Permutation + Time-warping).
    - Cả 2 góc nhìn đều đi qua CÙNG MỘT ENCODER (chia sẻ trọng số - Weight Sharing).
    - Đầu ra trả về 2 vector chiếu (h_weak, h_strong) để đưa vào hàm tính mất mát
      đối chiếu thời gian - ngữ cảnh (Temporal-Context Contrastive Loss).
===============================================================================
"""

import torch
import torch.nn as nn
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.projection import ProjectionHead


class TSTCCModel(nn.Module):
    """
    Mô hình Siamese Network TS-TCC hoàn chỉnh cho học tự giám sát chuỗi thời gian HAR.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        projection_head: nn.Module = None,
        in_channels: int = 6,
        feature_dim: int = 128,
        projection_dim: int = 64
    ):
        """
        Khởi tạo mô hình TS-TCC. Hỗ trợ truyền Module tùy chỉnh hoặc tự động khởi tạo.
        Tham số:
            encoder (nn.Module, optional): Backbone trích xuất đặc trưng tùy chọn.
                                          Nếu None, sẽ tự tạo StandardSensorEncoder1D.
            projection_head (nn.Module, optional): Đầu chiếu MLP tùy chọn.
                                                  Nếu None, sẽ tự tạo ProjectionHead.
            in_channels (int): Số kênh cảm biến đầu vào (mặc định: 6 kênh Acc + Gyro).
            feature_dim (int): Kích thước vector biểu diễn trung gian từ Encoder (mặc định: 128).
            projection_dim (int): Kích thước vector sau tầng chiếu để tính Loss (mặc định: 64).
        """
        super().__init__()

        # ---------------------------------------------------------------------
        # 1. KHỞI TẠO KHUNG XƯƠNG (ENCODER BACKBONE)
        # Nhận vào: Sóng thô (B, 6, 128) -> Xuất ra: Vector đặc trưng (B, 128)
        # ---------------------------------------------------------------------
        self.encoder = encoder if encoder is not None else StandardSensorEncoder1D(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        # ---------------------------------------------------------------------
        # 2. KHỞI TẠO ĐẦU CHIẾU TỰ GIÁM SÁT (PROJECTION HEAD)
        # Nhận vào: Vector (B, 128) -> Xuất ra: Vector chiếu đơn vị (B, 64)
        # ---------------------------------------------------------------------
        self.projection_head = projection_head if projection_head is not None else ProjectionHead(
            feature_dim=feature_dim,
            projection_dim=projection_dim
        )

    def forward(self, x_weak: torch.Tensor, x_strong: torch.Tensor):
        """
        Lan truyền tiến song song (Forward Pass) cho 2 nhánh Siamese Network.
        Tham số:
            x_weak   (torch.Tensor): Batch tín hiệu góc nhìn yếu, shape (B, in_channels, seq_len).
            x_strong (torch.Tensor): Batch tín hiệu góc nhìn mạnh, shape (B, in_channels, seq_len).
        Trả về:
            h_weak   (torch.Tensor): Vector biểu diễn đã chuẩn hóa L2 của nhánh Yếu, shape (B, projection_dim).
            h_strong (torch.Tensor): Vector biểu diễn đã chuẩn hóa L2 của nhánh Mạnh, shape (B, projection_dim).
        """
        # =====================================================================
        # NHÁNH 1: XỬ LÝ GÓC NHÌN YẾU (WEAK VIEW BRANCH)
        # =====================================================================
        # 1.1. Rút trích đặc trưng qua 1D-CNN Backbone -> shape: (B, 128)
        z_weak = self.encoder(x_weak)
        # 1.2. Chiếu sang không gian đối chiếu và chuẩn hóa L2 -> shape: (B, 64)
        h_weak = self.projection_head(z_weak, normalize=True)

        # =====================================================================
        # NHÁNH 2: XỬ LÝ GÓC NHÌN MẠNH (STRONG VIEW BRANCH)
        # (Dùng chung trọng số với Nhánh 1)
        # =====================================================================
        # 2.1. Rút trích đặc trưng qua 1D-CNN Backbone -> shape: (B, 128)
        z_strong = self.encoder(x_strong)
        # 2.2. Chiếu sang không gian đối chiếu và chuẩn hóa L2 -> shape: (B, 64)
        h_strong = self.projection_head(z_strong, normalize=True)

        return h_weak, h_strong