import torch.nn as nn
import torch.nn.functional as F


class TSTCCModel(nn.Module):
    """
    Kiến trúc Siamese Network chuẩn TS-TCC cho học tự giám sát chuỗi thời gian.
    """

    def __init__(self, encoder: nn.Module, feature_dim=128, projection_dim=64):
        """
        Args:
            encoder (nn.Module): Bộ mã hóa nền tảng (nhận linh hoạt cnn1d, lstm, transformer)
            feature_dim (int): Chiều dài vector đặc trưng phẳng đầu ra của Encoder
            projection_dim (int): Chiều dài vector sau khi qua tầng chiếu để tính toán Loss
        """
        super().__init__()
        self.encoder = encoder

        # Projection Head phi tuyến 2 lớp xóa bỏ thông tin đặc thù của phép tăng cường dữ liệu
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x_weak, x_strong):
        """
        Lan truyền tiến song song cho 2 góc nhìn dữ liệu cảm biến thô
        x_weak, x_strong shape: (B, Channels, Window_Length) -> (B, 12, 256)
        """
        # Luồng 1: Xử lý và chiếu góc nhìn yếu (Weak View)
        z_weak = self.encoder(x_weak)  # Nén về (B, feature_dim)
        h_weak = F.normalize(self.projection_head(z_weak), p=2, dim=1)  # Chuẩn hóa L2 về vector đơn vị

        # Luồng 2: Xử lý và chiếu góc nhìn mạnh (Strong View)
        z_strong = self.encoder(x_strong)
        h_strong = F.normalize(self.projection_head(z_strong), p=2, dim=1)

        return h_weak, h_strong