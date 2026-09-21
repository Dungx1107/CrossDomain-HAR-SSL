import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.projection import ProjectionHead


class PrototypicalHARModel(nn.Module):
    """
    Mô hình Prototypical Learning (SwAV-inspired) cho chuỗi cảm biến HAR 1D.
    Học biểu diễn ngữ nghĩa thông qua ma trận Prototype có thể tối ưu hóa.
    """

    def __init__(
            self,
            encoder: nn.Module,
            feature_dim: int,
            projection_dim: int = 64,
            num_prototypes: int = 45,
            temperature: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Non-linear Projection Head (chuẩn SimCLR/SwAV)
        self.projection_head = ProjectionHead(
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=feature_dim
        )

        # Prototype matrix (K, projection_dim) — dùng nn.Parameter, không bias
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, projection_dim))
        nn.init.orthogonal_(self.prototypes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)

        # 1. Trường hợp encoder đã pool sẵn về dạng vector 2D (B, D)
        if feat.dim() == 2:
            return feat

        # 2. Trường hợp encoder trả về tensor 3D
        if feat.dim() == 3:
            # Nếu encoder có định dạng (B, L, D) -> hoán vị về (B, D, L) chuẩn Conv1d
            if self.channel_last:
                feat = feat.transpose(1, 2)

            # Global Average Pooling theo trục thời gian (L) -> (B, D, 1) -> (B, D)
            feat = self.pool(feat).flatten(1)
            return feat

        raise ValueError(f"Tensor đầu ra của encoder có số chiều không hợp lệ: {feat.shape}")

    def forward(self, x_w: torch.Tensor, x_s: torch.Tensor):
        z_w = self.forward_backbone(x_w)
        z_s = self.forward_backbone(x_s)

        p_w = F.normalize(self.projection_head(z_w), dim=-1, p=2)
        p_s = F.normalize(self.projection_head(z_s), dim=-1, p=2)

        w = F.normalize(self.prototypes, dim=-1, p=2)

        scores_w = torch.matmul(p_w, w.t()) / self.temperature
        scores_s = torch.matmul(p_s, w.t()) / self.temperature

        return scores_w, scores_s
