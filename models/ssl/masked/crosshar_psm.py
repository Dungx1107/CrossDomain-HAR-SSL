import torch
import torch.nn as nn
from models.ssl.masked.modules import TimeDistributed


class PSMEncoder(nn.Module):
    def __init__(self, k: int = 2, fr: int = 50, num_feat_map: int = 64, p: float = 0.3, shar_channels: int = 3):
        super().__init__()
        self.k = k  # k = 2 cụm cảm biến (Acc: 3 trục, Gyro: 3 trục)
        self.fr = fr
        self.channels = num_feat_map

        # Tự động tính số kênh: k * 64 = 128 khi k=2
        conv2_in_channels = self.k * self.channels

        self.conv1 = nn.Conv2d(1, self.channels, kernel_size=(1, 1), padding=0)
        self.bnd1 = nn.BatchNorm2d(self.channels)
        self.mp_100 = nn.MaxPool2d(kernel_size=(2, 1))
        self.drop = nn.Dropout(p)

        self.conv2 = nn.Conv2d(conv2_in_channels, self.channels, kernel_size=(3, 1), padding=(1, 0))
        self.bnd2 = nn.BatchNorm2d(self.channels)
        self.mp_75 = nn.MaxPool2d(kernel_size=(1, 3))
        self.relu = nn.ReLU(inplace=True)

        # 3 * self.channels = 3 * 64 = 192 (chiều gom sau khối Conv2)
        self.time_dist = TimeDistributed(nn.Linear(3 * self.channels, shar_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        # Input nhận vào: (B, C, L, 1) với C=6 kênh
        x = x.permute(0, 3, 2, 1)  # -> (B, 1, L, 6)
        x = self.bnd1(self.relu(self.conv1(x)))  # -> (B, 64, L, 6)
        if self.fr in [50, 100]:
            x = self.mp_100(x)
        x = self.drop(x)

        x = x.permute(0, 1, 3, 2)  # -> (B, 64, 6, L')
        # Tách 6 kênh thành k=2 cụm cảm biến x 3 trục không gian
        x = x.reshape(x.shape[0], self.channels, self.k, 3, -1)
        x = x.permute(0, 2, 1, 3, 4)  # -> (B, k, 64, 3, L')
        x = x.reshape(x.shape[0], self.k * self.channels, 3, -1)  # -> (B, 128, 3, L')

        x = self.bnd2(self.relu(self.conv2(x)))  # -> (B, 64, 3, L')
        x = self.drop(x)
        x = x.permute(0, 1, 3, 2)  # -> (B, 64, L', 3)
        x = x.reshape(x.shape[0], 3 * self.channels, -1)  # -> (B, 192, L')
        x = self.time_dist(x)  # -> (B, shar_channels, L')
        return x


class PSMDecoder(nn.Module):
    def __init__(self, out_dim: int, p: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, num_layers=2, batch_first=True)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, Seq_len) -> chuyển trục thành (B, Seq_len, 3)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = self.drop(self.tanh(x[-1]))
        x = self.fc(x)
        return x


class PSM(nn.Module):
    def __init__(self, out_dim: int, k: int = 2, fr: int = 50, num_feat_map: int = 64, p: float = 0.3,
                 shar_channels: int = 3):
        super().__init__()
        self.shar_channels = shar_channels
        self.num_feat_map = num_feat_map
        # Đã đồng bộ k=2 ở cả PSM và PSMEncoder
        self.encoder = PSMEncoder(k=k, fr=fr, num_feat_map=num_feat_map, p=p, shar_channels=shar_channels)
        self.decoder = PSMDecoder(out_dim, p)

    def forward(self, x_list: list) -> torch.Tensor:
        encodes = []
        outputs = []
        device = next(self.parameters()).device

        for dev_input in x_list:
            dev_input = dev_input.to(device)
            # Tự động ép về dạng 4D: (B, C, L) -> (B, C, L, 1)
            if dev_input.dim() == 3:
                dev_input = dev_input.unsqueeze(-1)
            encode = self.encoder(dev_input)
            outputs.append(self.decoder(encode))
            encodes.append(encode)

        shared_encode = torch.mean(torch.stack(encodes), 2).permute(1, 0, 2).to(device)
        outputs.append(self.decoder(shared_encode))
        return torch.mean(torch.stack(outputs), 0)