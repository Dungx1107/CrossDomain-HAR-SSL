"""
===============================================================================
MODULE: TS-TCC x SwAV HYBRID MODEL
===============================================================================
Ý tưởng:
    Giữ nguyên Encoder + TC (Seq_Transformer) của TS-TCC, nhưng thay Contextual
    Contrasting (NT-Xent) bằng SwAV trên K prototype.

Luồng:
    x_weak, x_strong (B, 6, 128)
        -> TSTCCEncoder            -> z_w, z_s            (B, 128, 18)
        -> TC 2 chiều (cùng t)     -> loss_tc, c_w, c_s   (B, 32)  [chưa L2-norm]
        -> L2-normalize            -> cos_w, cos_s = c @ proto.T   (B, K)
        -> Sinkhorn(cos) làm target, softmax(cos / tau) làm prediction
        -> loss_swav (swap)
    loss_total = lambda_tc * loss_tc + lambda_swav * loss_swav (+ lambda_cc * loss_cc)

Đặt file tại: models/ssl/hybrid/tstcc_swav_model.py
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.tstcc_encoder import TSTCCEncoder
from models.ssl.contrastive.tc import TC
from losses.nt_xent import NTXentLoss
from losses.swav_loss import sinkhorn_knopp


class TSTCCSwAVModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module = None,
        in_channels: int = 6,
        feature_dim: int = 128,
        timesteps: int = 6,
        tc_hidden_dim: int = 64,
        tc_depth: int = 4,
        tc_heads: int = 4,
        tc_mlp_dim: int = 64,
        num_prototypes: int = 45,
        tau: float = 0.1,
        epsilon: float = 0.05,
        sinkhorn_iters: int = 3,
        lambda_tc: float = 1.0,
        lambda_swav: float = 1.0,
        lambda_cc: float = 0.0,
        cc_temperature: float = 0.2,
        min_t: int = 0,
    ):
        """
        tau      : nhiệt độ softmax của prediction p (SwAV gốc: 0.1).
        epsilon  : nhiệt độ Sinkhorn, áp lên COSINE THÔ (SwAV gốc: 0.05).
        lambda_cc: > 0 để bật thêm NT-Xent (dùng cho ablation), mặc định tắt.
        min_t    : mốc t nhỏ nhất, tăng lên để context luôn nhìn >= min_t+1 bước.
        """
        super().__init__()
        self.timesteps = timesteps
        self.tau = tau
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.lambda_tc = lambda_tc
        self.lambda_swav = lambda_swav
        self.lambda_cc = lambda_cc
        self.min_t = min_t

        self.encoder = encoder if encoder is not None else TSTCCEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            kernel_size=8,
            dropout=0.35,
        )

        self.tc = TC(
            final_out_channels=feature_dim,
            timesteps=timesteps,
            hidden_dim=tc_hidden_dim,
            depth=tc_depth,
            heads=tc_heads,
            mlp_dim=tc_mlp_dim,
        )

        # Chiều context lấy từ lớp Linear cuối của TC.projection_head (= feature_dim // 4 = 32)
        self.ctx_dim = self.tc.projection_head[-1].out_features

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, self.ctx_dim))
        nn.init.orthogonal_(self.prototypes)

        self.nt_xent = NTXentLoss(temperature=cc_temperature) if lambda_cc > 0 else None

        # Thống kê để theo dõi collapse (không tham gia gradient)
        self.last_stats = {}

    def forward(self, x_weak: torch.Tensor, x_strong: torch.Tensor):
        z_w = self.encoder(x_weak)      # (B, 128, 18)
        z_s = self.encoder(x_strong)    # (B, 128, 18)

        # Một mốc t dùng chung cho cả hai chiều (giống TSTCCModel)
        seq_len = z_w.shape[2]
        t_samples = torch.randint(self.min_t, seq_len - self.timesteps, size=(1,)).item()

        loss_tc1, c_w, _ = self.tc(z_w, z_s, t_samples=t_samples)
        loss_tc2, c_s, _ = self.tc(z_s, z_w, t_samples=t_samples)
        loss_tc = loss_tc1 + loss_tc2

        # ---- SwAV trên context vector ----
        ec_w = F.normalize(c_w, dim=-1, p=2)                  # (B, 32)
        ec_s = F.normalize(c_s, dim=-1, p=2)
        protos = F.normalize(self.prototypes, dim=-1, p=2)    # (K, 32), CÓ gradient

        cos_w = ec_w @ protos.t()                             # (B, K), cosine thô
        cos_s = ec_s @ protos.t()

        # Target: Sinkhorn trên cosine thô (epsilon áp 1 lần duy nhất)
        with torch.no_grad():
            q_w = sinkhorn_knopp(cos_w, epsilon=self.epsilon, niters=self.sinkhorn_iters)
            q_s = sinkhorn_knopp(cos_s, epsilon=self.epsilon, niters=self.sinkhorn_iters)

        # Prediction: softmax(cos / tau); swap: q của view này dạy p của view kia
        logp_w = F.log_softmax(cos_w / self.tau, dim=1)
        logp_s = F.log_softmax(cos_s / self.tau, dim=1)
        loss_swav = -0.5 * (
            torch.sum(q_s * logp_w, dim=1).mean() + torch.sum(q_w * logp_s, dim=1).mean()
        )

        loss_total = self.lambda_tc * loss_tc + self.lambda_swav * loss_swav

        if self.nt_xent is not None:
            loss_cc = self.nt_xent(c_w, c_s)
            loss_total = loss_total + self.lambda_cc * loss_cc

        # Thống kê giữ dạng tensor trên GPU (không .item()/unique() -> không ép sync mỗi step)
        with torch.no_grad():
            usage = q_w.mean(dim=0)
            entropy = -(usage * torch.log(usage + 1e-12)).sum()          # tối đa = ln(K)
            counts = torch.bincount(cos_w.argmax(dim=1), minlength=self.prototypes.shape[0])
            self.last_stats = {
                "t_samples": t_samples,
                "proto_entropy": entropy,
                "proto_used": (counts > 0).sum().float(),
            }

        return loss_total, loss_tc, loss_swav