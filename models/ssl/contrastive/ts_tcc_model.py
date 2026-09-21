"""
===============================================================================
MODULE: TS-TCC SIAMESE MODEL WRAPPER
===============================================================================
Mục đích:
    Đóng gói Encoder + Temporal Contrasting (TC) + Contextual Contrasting (NT-Xent)
    vào cùng một class để trainer chỉ cần gọi một lần.

Tham chiếu logic:
    Cách gọi trong `trainer.py` của repo gốc emadeldeen24/TS-TCC:
        z_weak   = model(x_weak)
        z_strong = model(x_strong)
        loss_tc, context = temporal_contr_model(z_weak, z_strong)
        loss_cc = NTXentLoss(context_weak, context_strong)
        loss_total = loss_tc + 0.7 * loss_cc

Cấu trúc luồng:
    x_weak, x_strong  (B, 6, 128)
        -> Encoder (TSTCCEncoder)
        -> z_weak, z_strong  (B, 128, 18)
        -> TC 2 chiều với cùng t_samples
        -> loss_tc = loss_tc_w2s + loss_tc_s2w
        -> context_w, context_s  (B, 32)
        -> NT-Xent  ->  loss_cc
        -> loss_total = 1.0 * loss_tc + 0.7 * loss_cc
===============================================================================
"""

import torch
import torch.nn as nn

from models.encoders.tstcc_encoder import TSTCCEncoder
from models.ssl.contrastive.tc import TC
from losses.nt_xent import NTXentLoss


class TSTCCModel(nn.Module):
    """
    Wrapper gộp Encoder + Temporal Contrasting + Contextual Contrasting.

    Không có gì mới về mặt kiến trúc — chỉ gom logic mà repo gốc viết rải rác
    trong trainer.py vào một chỗ cho gọn.
    """

    def __init__(
        self,
        encoder: nn.Module = None,
        in_channels: int = 6,
        feature_dim: int = 128,
        timesteps: int = 3,
        tc_hidden_dim: int = 64,
        tc_depth: int = 4,
        tc_heads: int = 4,
        tc_mlp_dim: int = 64,
        lambda1: float = 1.0,
        lambda2: float = 0.7,
        temperature: float = 0.2,
    ):
        """
        Args:
            encoder       : Backbone (mặc định TSTCCEncoder 3-block).
            in_channels   : Số kênh cảm biến (6).
            feature_dim   : Số kênh đặc trưng encoder (128).
            timesteps     : K bước tương lai cần dự đoán trong TC (3 theo config HAR).
            tc_hidden_dim : Chiều ẩn Transformer trong TC (64).
            tc_depth      : Số layer Transformer trong TC (4).
            tc_heads      : Số head attention (4).
            tc_mlp_dim    : Chiều FFN trong TC (64).
            lambda1       : Trọng số L_TC (1.0 theo bài báo).
            lambda2       : Trọng số L_CC (0.7 theo bài báo).
            temperature   : Nhiệt độ NT-Xent (0.2 theo bài báo).
        """
        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.timesteps = timesteps

        # 1. Encoder
        self.encoder = encoder if encoder is not None else TSTCCEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            kernel_size=8,
            dropout=0.35,
        )

        # 2. Temporal Contrasting
        self.tc = TC(
            final_out_channels=feature_dim,
            timesteps=timesteps,
            hidden_dim=tc_hidden_dim,
            depth=tc_depth,
            heads=tc_heads,
            mlp_dim=tc_mlp_dim,
        )

        # 3. Contextual Contrasting
        self.nt_xent = NTXentLoss(temperature=temperature)

    def forward(self, x_weak: torch.Tensor, x_strong: torch.Tensor):
        """
        Args:
            x_weak   : (B, in_channels, seq_len)
            x_strong : (B, in_channels, seq_len)

        Returns:
            loss_total : scalar
            loss_tc    : scalar (chỉ để log)
            loss_cc    : scalar (chỉ để log)
        """
        device = x_weak.device

        # ---------------------------------------------------------------
        # 1. Encoder: (B, C, T_in) -> (B, feature_dim, T_out)
        # ---------------------------------------------------------------
        z_weak = self.encoder(x_weak)      # (B, 128, 18)
        z_strong = self.encoder(x_strong)  # (B, 128, 18)

        # ---------------------------------------------------------------
        # 2. Bốc 1 mốc t_samples duy nhất, dùng chung cho cả 2 chiều
        # ---------------------------------------------------------------
        seq_len = z_weak.shape[2]
        # t_samples là scalar int, nằm trong [0, seq_len - timesteps - 1]
        t_samples = torch.randint(
            0, seq_len - self.timesteps, size=(1,)
        ).item()

        # ---------------------------------------------------------------
        # 3. Temporal Contrasting 2 chiều (dùng cùng t_samples)
        # ---------------------------------------------------------------
        # Chiều 1: Weak -> Strong
        loss_tc1, context_w, _ = self.tc(z_weak, z_strong, t_samples=t_samples)

        # Chiều 2: Strong -> Weak
        loss_tc2, context_s, _ = self.tc(z_strong, z_weak, t_samples=t_samples)

        loss_tc = loss_tc1 + loss_tc2

        # ---------------------------------------------------------------
        # 4. Contextual Contrasting (NT-Xent)
        # ---------------------------------------------------------------
        loss_cc = self.nt_xent(context_w, context_s)

        # ---------------------------------------------------------------
        # 5. Tổng loss
        # ---------------------------------------------------------------
        loss_total = self.lambda1 * loss_tc + self.lambda2 * loss_cc

        return loss_total, loss_tc, loss_cc