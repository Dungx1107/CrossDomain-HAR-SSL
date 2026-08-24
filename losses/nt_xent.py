import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    """
    Hàm mất mát tương phản đối ngẫu NT-Xent (Normalized Temperature-scaled Cross Entropy)
    Mục đích: Kéo gần 2 views của cùng một cửa sổ, đẩy xa các cửa sổ khác trong Batch.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, h_weak, h_strong):
        """
        Args:
            h_weak (Tensor): Vector chiếu của Weak View, shape (B, projection_dim)
            h_strong (Tensor): Vector chiếu của Strong View, shape (B, projection_dim)
        """
        batch_size = h_weak.shape[0]
        device = h_weak.device

        # Ghép chuỗi các vector đặc trưng lại để tính tương quan chéo
        out = torch.cat([h_weak, h_strong], dim=0)  # Shape: (2*B, projection_dim)

        # Bước 1: Tính ma trận tương đồng Cosine giữa mọi cặp views trong batch
        sim_matrix = torch.mm(out, out.t()) / self.temperature  # Shape: (2*B, 2*B)

        # Bước 2: Tạo mặt nạ xóa bỏ đường chéo (loại bỏ việc mẫu tự so sánh với chính mình)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)  # Gán số cực âm để exp(-9e15) ~ 0

        # Bước 3: Thiết lập nhãn mục tiêu (Target) cho bài toán trắc nghiệm phân biệt thực thể
        # Mẫu i của weak phải khớp với mẫu i của strong nằm ở nửa sau ma trận và ngược lại
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ], dim=0)

        loss = self.criterion(sim_matrix, targets)
        return loss