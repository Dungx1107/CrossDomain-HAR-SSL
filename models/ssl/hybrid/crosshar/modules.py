import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    """
    Áp dụng một module (ví dụ nn.Linear) lên từng lát cắt theo chiều thời gian.
    Nhận vào (B, Channels, Length), áp dụng module lên Channels rồi trả về cùng định dạng.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, Feat, Seq_len) -> chuyển thành (B * Seq_len, Feat)
        b, feat, seq_len = x.shape
        x_reshaped = x.permute(0, 2, 1).contiguous().view(b * seq_len, feat)
        out = self.module(x_reshaped)
        # Chuyển ngược lại (B, New_Feat, Seq_len)
        new_feat = out.shape[-1]
        out = out.view(b, seq_len, new_feat).permute(0, 2, 1).contiguous()
        return out