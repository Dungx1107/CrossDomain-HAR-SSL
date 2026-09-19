"""
Script: Trực quan hóa không gian đặc trưng t-SNE hoàn toàn độc lập (Standalone)
Vị trí: utils/plot_clusters_standalone.py
Mục đích: Tự đọc trực tiếp checkpoint .pt trong ssl_pretrain/prototype mà không phụ thuộc module ngoài.
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================================
# 1. TỰ ĐỊNH NGHĨA CÁC KIẾN TRÚC BACKBONE (ĐỘC LẬP HOÀN TOÀN)
# ==============================================================================

class StandardSensorEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 6, feature_dim: int = 128):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.projection(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class CNNTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, d_model, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=64)
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn_stem(x)
        feat = feat.transpose(1, 2)
        feat = self.pos_encoder(feat)
        feat = self.dropout(feat)
        out = self.transformer_encoder(feat)
        out = self.norm(out)
        return out.transpose(1, 2)


class ViT1DEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, patch_size: int = 16, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = nn.Conv1d(
            in_channels=in_channels, out_channels=d_model,
            kernel_size=patch_size, stride=patch_size
        )
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=32)
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x)
        tokens = tokens.transpose(1, 2)
        tokens = self.pos_encoder(tokens)
        tokens = self.dropout(tokens)
        out = self.transformer_encoder(tokens)
        out = self.norm(out)
        return out.transpose(1, 2)


def get_model(backbone_type: str, in_channels: int = 6) -> nn.Module:
    if backbone_type == "standard":
        return StandardSensorEncoder1D(in_channels=in_channels, feature_dim=128)
    elif backbone_type == "cnn_transformer":
        return CNNTransformerEncoder(in_channels=in_channels, d_model=128)
    elif backbone_type == "vit_1d":
        return ViT1DEncoder(in_channels=in_channels, patch_size=16, d_model=128)
    raise ValueError(f"Backbone không hợp lệ: {backbone_type}")


# ==============================================================================
# 2. CẤU HÌNH THỰC THI & TỰ ĐỘNG QUÉT ĐƯỜNG DẪN
# ==============================================================================
# Tùy chỉnh trực tiếp 2 biến dưới đây:
DOMAIN = "motionsense"  # "motionsense" hoặc "uci_har"
BACKBONE_TYPE = "standard"  # "standard" | "cnn_transformer" | "vit_1d"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMMON_CLASSES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']

# Tự động nhận diện ROOT từ thư mục chứa file utils này
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Đường dẫn file checkpoint .pt
CKPT_PATH = (
        PROJECT_ROOT
        / "checkpoints"
        / "ssl_pretrain"
        / "prototype"
        / DOMAIN
        / BACKBONE_TYPE
        / f"prototype_{BACKBONE_TYPE}_encoder_pretrained_{DOMAIN}.pt"
)

# Đường dẫn file test .pt (tự tìm trong data/processed)
TEST_PATH = PROJECT_ROOT / "data" / "processed" / DOMAIN / "test.pt"

if not CKPT_PATH.exists():
    raise FileNotFoundError(f"❌ Không tìm thấy checkpoint: {CKPT_PATH}")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu test: {TEST_PATH}")

print("=" * 70)
print(f"🚀 VẼ CỤM t-SNE CHO MÔ HÌNH PROTOTYPE")
print(f"📦 Checkpoint : {CKPT_PATH}")
print(f"📂 Dữ liệu test: {TEST_PATH}")
print(f"🧠 Thiết bị   : {DEVICE}")
print("=" * 70)

# ==============================================================================
# 3. NẠP DỮ LIỆU & RÚT TRÍCH ĐẶC TRƯNG
# ==============================================================================
# Đọc trực tiếp dictionary từ file .pt
raw_test = torch.load(TEST_PATH, map_location="cpu", weights_only=True)
samples = raw_test["samples"]
labels = raw_test["labels"].squeeze()

# Ép kiểu tensor nếu cần
if not isinstance(samples, torch.Tensor):
    samples = torch.tensor(samples, dtype=torch.float32)
if not isinstance(labels, torch.Tensor):
    labels = torch.tensor(labels, dtype=torch.long)

# Lọc đúng 5 lớp hành động chung (0 đến 4)
mask = (labels >= 0) & (labels < 5)
x_test, y_test = samples[mask], labels[mask]

loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128, shuffle=False)

# Nạp model
model = get_model(BACKBONE_TYPE, in_channels=6)
state_dict = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

features_list = []
labels_list = []

with torch.no_grad():
    for x_b, y_b in loader:
        x_b = x_b.to(DEVICE)
        feat = model(x_b)

        # Nếu output là tensor 3D (như cnn_transformer hay vit_1d), nén trung bình theo chiều thời gian
        if feat.dim() == 3:
            feat = feat.mean(dim=-1)

        # Chuẩn hóa L2 vector đặc trưng
        feat = F.normalize(feat, p=2, dim=-1)

        features_list.append(feat.cpu().numpy())
        labels_list.append(y_b.numpy())

all_features = np.concatenate(features_list, axis=0)
all_labels = np.concatenate(labels_list, axis=0)

# ==============================================================================
# 4. CHẠY t-SNE & XUẤT ẢNH
# ==============================================================================
print(f"⏳ Đang chạy t-SNE cho {len(all_features)} mẫu đặc trưng...")
tsne = TSNE(n_components=2, perplexity=35, random_state=42, max_iter=1000)
embedded_2d = tsne.fit_transform(all_features)

plt.figure(figsize=(9, 7), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for cls_idx, cls_name in enumerate(COMMON_CLASSES):
    idx = np.where(all_labels == cls_idx)[0]
    plt.scatter(
        embedded_2d[idx, 0],
        embedded_2d[idx, 1],
        c=colors[cls_idx],
        label=cls_name,
        alpha=0.6,
        s=16,
        edgecolors='none'
    )

plt.title(f"Prototype Feature Clusters (t-SNE) | {BACKBONE_TYPE.upper()} - {DOMAIN.upper()}", fontsize=12,
          fontweight='bold')
plt.xlabel("t-SNE Dimension 1", fontsize=11)
plt.ylabel("t-SNE Dimension 2", fontsize=11)
plt.legend(markerscale=2.5, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

