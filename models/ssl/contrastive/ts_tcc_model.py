"""
===============================================================================
VAI TRÒ TRONG HỆ THỐNG:
    - Đóng gói toàn bộ kiến trúc mạng Siamese (Siamese Network Architecture)
      cho giai đoạn Huấn luyện Tự Giám Sát Đối Chiếu (TS-TCC Pre-training).
    - Tự động tương thích với mọi loại Backbone (Standard 1D-CNN, ViT-1D, CNN-Transformer):
        + Nếu Backbone xuất tensor 2D (B, 128): Đưa thẳng vào ProjectionHead.
        + Nếu Backbone xuất tensor 3D (B, 128, L): Tự động nén qua Global Average Pooling
          dọc theo trục thời gian (dim=-1) để đưa vào ProjectionHead.
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
        super().__init__()

        # 1. Khởi tạo Khung xương (Encoder Backbone)
        self.encoder = encoder if encoder is not None else StandardSensorEncoder1D(
            in_channels=in_channels,
            feature_dim=feature_dim
        )

        # 2. Khởi tạo Đầu chiếu tự giám sát (Projection Head)
        self.projection_head = projection_head if projection_head is not None else ProjectionHead(
            feature_dim=feature_dim,
            projection_dim=projection_dim
        )

    def _pool_if_needed(self, z: torch.Tensor) -> torch.Tensor:
        """
        Đảm bảo tensor đầu ra từ encoder có dạng 2D (B, feature_dim).
        Nếu encoder trả về dạng 3D (B, feature_dim, L), áp dụng Global Average Pooling theo chiều thời gian.
        """
        if z.dim() == 3:
            # (B, 128, L) -> (B, 128)
            z = z.mean(dim=-1)
        elif z.dim() != 2:
            raise ValueError(f"❌ Tensor đầu ra từ encoder phải là 2D hoặc 3D, nhận được shape: {tuple(z.shape)}")
        return z

    def forward(self, x_weak: torch.Tensor, x_strong: torch.Tensor):
        """
        Lan truyền tiến song song (Forward Pass) cho 2 nhánh Siamese Network.
        Tham số:
            x_weak   (torch.Tensor): Batch tín hiệu góc nhìn yếu (B, in_channels, seq_len).
            x_strong (torch.Tensor): Batch tín hiệu góc nhìn mạnh (B, in_channels, seq_len).
        Trả về:
            h_weak   (torch.Tensor): Vector biểu diễn chuẩn hóa L2 của nhánh Yếu (B, projection_dim).
            h_strong (torch.Tensor): Vector biểu diễn chuẩn hóa L2 của nhánh Mạnh (B, projection_dim).
        """
        # =====================================================================
        # NHÁNH 1: GÓC NHÌN YẾU (WEAK VIEW BRANCH)
        # =====================================================================
        z_weak = self.encoder(x_weak)
        z_weak_pooled = self._pool_if_needed(z_weak)
        h_weak = self.projection_head(z_weak_pooled, normalize=True)

        # =====================================================================
        # NHÁNH 2: GÓC NHÌN MẠNH (STRONG VIEW BRANCH)
        # =====================================================================
        z_strong = self.encoder(x_strong)
        z_strong_pooled = self._pool_if_needed(z_strong)
        h_strong = self.projection_head(z_strong_pooled, normalize=True)

        return h_weak, h_strong


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 KIỂM TRA TSTCCModel VỚI CẢ 3 BACKBONE")
    print("=" * 70)

    from models.encoders.builder import build_encoder

    x1 = torch.randn(4, 6, 128)
    x2 = torch.randn(4, 6, 128)

    for backbone in ["standard", "cnn_transformer", "vit_1d"]:
        try:
            enc = build_encoder(backbone_type=backbone, in_channels=6)
            model = TSTCCModel(encoder=enc, in_channels=6, feature_dim=128, projection_dim=64)
            h_w, h_s = model(x1, x2)
            print(f"✅ Backbone [{backbone.upper()}]:")
            print(f"   - Output h_weak shape   : {tuple(h_w.shape)}")
            print(f"   - Output h_strong shape : {tuple(h_s.shape)}")
        except Exception as e:
            print(f"❌ Backbone [{backbone.upper()}] LỖI: {e}")

    print("=" * 70)