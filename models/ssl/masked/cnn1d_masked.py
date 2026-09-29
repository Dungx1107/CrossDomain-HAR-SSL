import torch
import torch.nn as nn
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.ssl.masked.mask_generator import SegmentMaskGenerator


class Conv1DDecoder(nn.Module):
    """
    Decoder tái tạo tín hiệu chuẩn đối xứng với StandardSensorEncoder1D (từ chiều dài 8 lên 128).
    """
    def __init__(self, in_channels: int = 128, out_channels: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 128, 8) -> (B, 96, 16)
            nn.ConvTranspose1d(in_channels, 96, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),

            # (B, 96, 16) -> (B, 64, 32)
            nn.ConvTranspose1d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # (B, 64, 32) -> (B, 32, 64)
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # (B, 32, 64) -> (B, 6, 128)
            nn.ConvTranspose1d(32, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StandardCNNMaskedAutoEncoder(nn.Module):
    """
    Autoencoder học mask sử dụng Encoder 1D-CNN chuẩn của dự án.
    """
    def __init__(self, in_channels: int = 6, feature_dim: int = 128, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_generator = SegmentMaskGenerator(mask_ratio=mask_ratio)
        self.encoder = StandardSensorEncoder1D(in_channels=in_channels, feature_dim=feature_dim)
        self.decoder = Conv1DDecoder(in_channels=feature_dim, out_channels=in_channels)

    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        mask = self.mask_generator(x)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        z = self.encoder(x_masked)
        x_recon = self.decoder(z)

        # Tính Loss MSE chỉ tại các vị trí mask
        mask_f = mask.float()
        loss = torch.sum(((x_recon - x) ** 2) * mask_f) / (mask_f.sum() + 1e-8)

        return {"loss": loss, "x_recon": x_recon, "mask": mask, "latent": z}