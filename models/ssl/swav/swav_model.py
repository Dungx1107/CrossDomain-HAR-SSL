"""
===============================================================================
MODULE: SWAV MODEL — backbone + projection head + PROTOTYPES (không chứa loss)
===============================================================================
VAI TRÒ
    Biến 2 view đầu vào thành 2 ma trận cosine (B, K) giữa embedding và prototype.
    Việc gán cụm (Sinkhorn) và loss nằm ở losses/swav_loss.py; lịch học, đóng băng
    prototype nằm ở engines/pretrain_ssl/swav/swav_trainer.py.

DÒNG CHẢY SHAPE (B = batch, F = feature_dim, D = projection_dim, K = num_prototypes)
    x            (B, C, T)
      └ encoder            ─► (B, F)  hoặc (B, F, T') rồi gộp bằng AdaptiveAvgPool1d -> (B, F)
      └ projection_head    ─► (B, D)
      └ L2-normalize       ─► p  (B, D), độ dài = 1                         ("embedding")
    prototypes   (K, D)  ─ L2-normalize ─► w (K, D), mỗi hàng độ dài = 1
    cos = p @ w.T        ─► (B, K), giá trị trong [-1, 1]                    (đầu ra của forward)

[ĐỐI CHIẾU] Bản gốc (resnet50.py của facebookresearch/swav): prototypes là nn.Linear(D, K, bias=False);
            embedding được L2-normalize rồi nhân với prototype -> cùng ý nghĩa với `p @ w.T` ở đây.
===============================================================================
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.projection import ProjectionHead


class SwAVModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            feature_dim: int,
            projection_dim: int = 64,
            num_prototypes: int = 45,
    ):
        """
        Args:
            encoder        : backbone, đầu ra (B, F) hoặc (B, F, T').
            feature_dim    : F, phải KHỚP kích thước đầu ra thật của encoder.
            projection_dim : D, chiều của không gian nơi prototype sống.
            num_prototypes : K, số cụm. [TÙY CHỈNH] giữ B/K >= ~1 (B = batch size), xem swav_config.py.

        [LƯU Ý] Nhiệt độ τ, ε KHÔNG còn nằm ở đây nữa: chúng thuộc về losses/swav_loss.py.
        """
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.num_prototypes = num_prototypes

        # Gộp theo trục thời gian khi encoder trả (B, F, T').
        self.pool = nn.AdaptiveAvgPool1d(1)

        # [TÙY CHỈNH] Đổi hidden_dim / số tầng của head ở models/heads/projection.py.
        self.projection_head = ProjectionHead(
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=feature_dim,
        )

        # ---- PROTOTYPES: K tâm cụm học được, mỗi tâm là 1 vector D chiều -------------
        # Orthogonal init: các tâm ban đầu vuông góc nhau -> rải đều trên mặt cầu.
        # [LƯU Ý] Cần K <= D để orthogonal thật sự (45 <= 64 thì ổn). Nếu K > D, các hàng chỉ
        #         "gần" vuông góc; nếu muốn dùng D nhỏ hơn K hãy kiểm tra lại cách khởi tạo.
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, projection_dim))
        nn.init.orthogonal_(self.prototypes)

    # -------------------------------------------------------------------------
    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, C, T) -> đặc trưng (B, F)."""
        feat = self.encoder(x)
        if feat.dim() == 2:
            return feat                              # encoder đã trả (B, F)
        if feat.dim() == 3:
            return self.pool(feat).flatten(1)        # (B, F, T') -> (B, F)
        raise ValueError(f"Tensor encoder không hợp lệ: {feat.shape}")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x (B, C, T) -> embedding đã chuẩn hóa L2, shape (B, D).
        [MỞ RỘNG - queue] Queue lưu CHÍNH tensor này (không phải điểm số), rồi nhân lại với
        prototype hiện tại ở mỗi iteration.
        """
        z = self.forward_backbone(x)                  # (B, F)
        p = self.projection_head(z)                   # (B, D)
        return F.normalize(p, dim=-1, p=2)            # (B, D), ||p|| = 1

    # -------------------------------------------------------------------------
    def forward(self, x_w: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_w : view weak   (B, C, T)
            x_s : view strong (B, C, T)
        Returns:
            cos_w, cos_s : (B, K) cosine THÔ (chưa chia nhiệt độ nào), nằm trong [-1, 1].

        [LƯU Ý] Hai view đi qua backbone bằng HAI lần forward riêng, nên BatchNorm tính thống kê
                theo từng view. Bản gốc nối các view cùng độ phân giải thành 1 batch rồi forward
                một lần. Muốn giống bản gốc: torch.cat([x_w, x_s]) rồi tách đôi kết quả.
        """
        p_w = self.embed(x_w)                         # (B, D)
        p_s = self.embed(x_s)                         # (B, D)

        # ---- Chuẩn hóa prototype NGAY TRONG forward (cách "B") ------------------------
        # Mỗi lần forward, prototype được đưa về độ dài 1 nên cos luôn trong [-1, 1].
        # Gradient chảy qua phép chuẩn hóa này (chỉ còn ảnh hưởng đến HƯỚNG prototype).
        # [ĐỐI CHIẾU] Bản gốc chuẩn hóa IN-PLACE, ngoài đồ thị, mỗi iteration (torch.no_grad + copy_).
        # [TÙY CHỈNH] Muốn đúng bản gốc ("cách A"):
        #     1) ở đây dùng: w = self.prototypes            (bỏ F.normalize)
        #     2) trong trainer, ngay TRƯỚC forward mỗi iteration:
        #            with torch.no_grad():
        #                model.prototypes.copy_(F.normalize(model.prototypes, dim=1))
        #     (nếu quên bước 2, cos sẽ ra ngoài [-1, 1] mà KHÔNG báo lỗi -> vì vậy mặc định dùng cách B)
        w = F.normalize(self.prototypes, dim=-1, p=2)  # (K, D)

        cos_w = p_w @ w.t()                            # (B, K)
        cos_s = p_s @ w.t()                            # (B, K)
        return cos_w, cos_s