"""
===============================================================================
MODULE: TEMPORAL CONTRASTING (TC) DÙNG LSTM THAY CHO TRANSFORMER
===============================================================================
Đặt file tại: models/ssl/tstcc/tc_lstm.py

Giữ NGUYÊN so với tc.py: Wk, projection_head, cách lấy encode_samples, InfoNCE
(LogSoftmax + đường chéo), giao diện forward(f1, f2, t_samples).
CHỈ THAY phần sinh vector ngữ cảnh c_t:

    tc.py     : Seq_Transformer (linear embed + CLS token + 4 layer attention)
    tc_lstm.py: nn.LSTM trên z_aug1[:, :t+1], c_t = hidden state cuối (h_n[-1])

Vì LSTM một chiều nên h_n của chuỗi quá khứ [0..t] chính là "tóm tắt tới thời
điểm t" — không cần CLS token.
===============================================================================
"""

import torch
import torch.nn as nn


class TCLSTM(nn.Module):
    def __init__(
            self,
            final_out_channels: int = 128,
            timesteps: int = 3,
            hidden_dim: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
    ):
        """
        Args:
            final_out_channels: Số kênh đầu ra của backbone (128).
            timesteps         : Số bước tương lai cần dự đoán K.
            hidden_dim        : Hidden size của LSTM = chiều c_t (64, bằng bản Transformer).
            num_layers        : Số lớp LSTM xếp chồng.
            dropout           : Dropout giữa các lớp LSTM (chỉ có tác dụng khi num_layers > 1).
        """
        super().__init__()
        self.num_channels = final_out_channels
        self.timestep = timesteps

        self.Wk = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_channels) for _ in range(self.timestep)
        ])
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        self.seq_lstm = nn.LSTM(
            input_size=final_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # BẮT BUỘC một chiều: không nhìn tương lai
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self,
                features_aug1: torch.Tensor,
                features_aug2: torch.Tensor,
                t_samples=None):
        """
        Args:
            features_aug1: (B, channels, seq_len) - view cung cấp quá khứ.
            features_aug2: (B, channels, seq_len) - view cung cấp tương lai (target).
        Returns:
            nce, proj_context (B, channels//4), t_samples
        """
        device = features_aug1.device

        z_aug1 = features_aug1.transpose(1, 2)  # (B, T, C)
        z_aug2 = features_aug2.transpose(1, 2)

        batch_size = z_aug1.shape[0]
        seq_len = z_aug1.shape[1]

        if t_samples is None:
            t_samples = torch.randint(0, seq_len - self.timestep, size=(1,)).item()

        # Target: K bước tương lai của view 2
        encode_samples = torch.empty((self.timestep, batch_size, self.num_channels), device=device)
        for i in range(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch_size, self.num_channels)

        # Quá khứ của view 1 -> LSTM -> c_t = hidden state cuối của lớp trên cùng
        forward_seq = z_aug1[:, :t_samples + 1, :]
        _, (h_n, _) = self.seq_lstm(forward_seq)
        c_t = h_n[-1]  # (B, hidden_dim)

        pred = torch.empty((self.timestep, batch_size, self.num_channels), device=device)
        for i in range(self.timestep):
            pred[i] = self.Wk[i](c_t)

        nce = 0.0
        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1.0 * batch_size * self.timestep

        return nce, self.projection_head(c_t), t_samples