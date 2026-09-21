"""
===============================================================================
MODULE: TEMPORAL CONTRASTING (TC) - REPRODUCTION TS-TCC
===============================================================================
Mã nguồn gốc từ file `models/TC.py` của repo `emadeldeen24/TS-TCC`.
Thực hiện:
    - Bốc ngẫu nhiên mốc thời gian t_samples.
    - Cắt chuỗi quá khứ [:t_samples + 1] đưa qua Seq_Transformer trích xuất c_t.
    - Dùng ModuleList Wk để dự đoán timestep tương lai z_{t+k}.
    - Tính hàm mất mát InfoNCE qua LogSoftmax và ma trận đường chéo.
===============================================================================
"""

import torch
import torch.nn as nn
from models.ssl.contrastive.attention import Seq_Transformer


class TC(nn.Module):
    def __init__(
            self,
            final_out_channels: int = 128,
            timesteps: int = 3,
            hidden_dim: int = 64,
            depth: int = 4,
            heads: int = 4,
            mlp_dim: int = 64
    ):
        """
        Args:
            final_out_channels: Số kênh đầu ra của Backbone (128).
            timesteps: Số bước tương lai cần dự đoán (mặc định 3 theo bài báo cho HAR).
            hidden_dim: Chiều không gian ẩn của Transformer trong TC (mặc định 64).
            depth: Số tầng Transformer layer (mặc định 4).
            heads: Số head attention (mặc định 4).
            mlp_dim: Chiều feed-forward trong Attention (mặc định 64).
        """
        super(TC, self).__init__()
        self.num_channels = final_out_channels
        self.timestep = timesteps

        # K ma trận chiếu dự đoán K bước tương lai W_k
        self.Wk = nn.ModuleList([
            nn.Linear(hidden_dim, self.num_channels) for _ in range(self.timestep)
        ])

        # LogSoftmax tính ma trận xác suất InfoNCE
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        # Projection Head nén c_t phục vụ Contextual Contrasting (NT-Xent)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )

        # Mạng Seq_Transformer gắn kèm CLS token
        self.seq_transformer = Seq_Transformer(
            patch_size=self.num_channels,
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self,
                features_aug1: torch.Tensor,
                features_aug2: torch.Tensor,
                t_samples=None
                ):
        """
        Args:
            features_aug1: Feature map từ view 1 (Weak), shape (B, channels, seq_len) -> (B, 128, 18)
            features_aug2: Feature map từ view 2 (Strong), shape (B, channels, seq_len) -> (B, 128, 18)

        Returns:
            nce: Hàm mất mát InfoNCE của Temporal Contrasting (scalar).
            proj_context: Vector ngữ cảnh sau khi qua Projection Head để đưa vào NT-Xent.
        """
        device = features_aug1.device

        # Chuyển đổi tensor sang dạng (B, seq_len, channels)
        z_aug1 = features_aug1.transpose(1, 2)
        z_aug2 = features_aug2.transpose(1, 2)

        batch_size = z_aug1.shape[0]
        seq_len = z_aug1.shape[1]

        if t_samples is None:
            t_samples = torch.randint(0, seq_len - self.timestep, size=(1,)).item()

        # 2. Lấy mẫu thực tế của view 2 tại các bước tương lai (Target)
        encode_samples = torch.empty((self.timestep, batch_size, self.num_channels), device=device)
        for i in range(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch_size, self.num_channels)

        # 3. Chuỗi quá khứ của view 1 từ đầu đến t_samples
        forward_seq = z_aug1[:, :t_samples + 1, :]

        # 4. Trích xuất vector ngữ cảnh c_t từ Seq_Transformer
        c_t = self.seq_transformer(forward_seq)  # (B, hidden_dim)

        # 5. Dự đoán các bước tương lai bằng các tầng Wk
        pred = torch.empty((self.timestep, batch_size, self.num_channels), device=device)
        for i in range(self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        # 6. Tính InfoNCE Loss qua LogSoftmax
        nce = 0.0
        for i in range(self.timestep):
            # Tính tích vô hướng giữa Target và Pred: (B, B)
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            # Lấy các giá trị trên đường chéo (cùng mẫu i)
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        # Trung bình hóa trên cả batch và số bước timestep (đổi dấu do tối đa hóa log-likelihood)
        nce /= -1.0 * batch_size * self.timestep

        return nce, self.projection_head(c_t), t_samples
