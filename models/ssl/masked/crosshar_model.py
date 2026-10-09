"""
===============================================================================
MODULE: CROSSHAR ARCHITECTURE (PRETRAIN HIERARCHICAL MODEL & CLASSIFIER - 5 LỚP)
===============================================================================
"""

import sys
import math
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ssl.masked.mask_generator import SegmentMaskGenerator
from models.encoders.builder import build_encoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"seq_len={seq_len} vượt quá max_len={self.pe.size(1)}")
        return x + self.pe[:, :seq_len, :]


class TransformerContextModule(nn.Module):
    def __init__(
            self,
            d_model: int = 128,
            nhead: int = 4,
            dim_feedforward: int = 256,
            dropout: float = 0.1,
            max_len: int = 32
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_t = z.permute(0, 2, 1)
        z_t = self.pos_encoder(z_t)
        out = self.transformer(z_t)
        return out.permute(0, 2, 1)


class TransformerSensorDecoder(nn.Module):
    def __init__(self, feature_dim: int = 128, out_channels: int = 6, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # Positional Encoding để khôi phục nhận thức về thứ tự thời gian
        self.pos_encoder = PositionalEncoding(d_model=feature_dim, max_len=128)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        # Sử dụng TransformerEncoder làm Decoder (theo phong cách Masked Autoencoder / BERT)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Lớp tuyến tính để ánh xạ từ chiều feature (128) về lại tín hiệu vật lý (6 kênh)
        self.output_projection = nn.Linear(feature_dim, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, feature_dim, Seq_Len) -> Chuyển thành (B, Seq_Len, feature_dim) cho Transformer
        z_t = z.permute(0, 2, 1)
        z_t = self.pos_encoder(z_t)

        decoded_features = self.transformer_decoder(z_t)

        x_recon = self.output_projection(decoded_features)

        # Trả về shape ban đầu (B, out_channels, Seq_Len)
        return x_recon.permute(0, 2, 1)


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class FeatureMixerClassificationHead(nn.Module):
    def __init__(self, in_features: int = 128, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, in_features)
        self.bn = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.flatten(self.pool(x))
        h = self.drop(self.relu(self.bn(self.fc1(h))))
        return self.classifier(h)


class CrossHARPretrainModel(nn.Module):
    def __init__(
            self,
            backbone_type: str = "cnn_transformer",
            in_channels: int = 6,
            feature_dim: int = 128,
            feature_length: int = 32,
            mask_ratio: float = 0.15,
            temperature: float = 0.2
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_length = feature_length
        self.temperature = temperature

        self.encoder = build_encoder(backbone_type=backbone_type, in_channels=in_channels)
        self.mask_generator = SegmentMaskGenerator(mask_ratio=mask_ratio)
        self.decoder = TransformerSensorDecoder(
            feature_dim=feature_dim,
            out_channels=in_channels,
            nhead=4,
            num_layers=2
        )
        self.reg_head = TransformerContextModule(d_model=feature_dim, nhead=4, max_len=feature_length)
        self.proj_head = ContrastiveProjectionHead(in_dim=feature_dim, out_dim=64)

    def forward_msm(self, x_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask_generator(x_raw)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        if mask.shape[1] != x_raw.shape[1]:
            mask = mask.expand(-1, x_raw.shape[1], -1)

        x_masked = x_raw.clone()
        x_masked[mask] = 0.0

        z = self.encoder(x_masked)
        x_recon = self.decoder(z)

        # TỰ ĐỘNG ĐỒNG BỘ CHIỀU DÀI: Khớp mọi loại backbone (1D-CNN, Transformer, ViT...)
        if x_recon.shape[-1] != x_raw.shape[-1]:
            x_recon = F.interpolate(x_recon, size=x_raw.shape[-1], mode='linear', align_corners=False)

        mask_f = mask.float()
        loss_m = torch.sum(((x_recon - x_raw) ** 2) * mask_f) / (mask_f.sum() + 1e-8)
        return loss_m, x_recon

    def forward_contrastive(self, x_neg: torch.Tensor, x_pos: torch.Tensor) -> torch.Tensor:
        batch_size = x_neg.size(0)
        z_neg = self.encoder(x_neg)
        z_pos = self.encoder(x_pos)

        r_neg = self.reg_head(z_neg)
        r_pos = self.reg_head(z_pos)

        s_neg = self.proj_head(r_neg)
        s_pos = self.proj_head(r_pos)

        representations = torch.cat([s_neg, s_pos], dim=0)
        sim_matrix = torch.matmul(representations, representations.T) / self.temperature

        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)

        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=sim_matrix.device)
        negatives = sim_matrix[~mask_self].view(2 * batch_size, -1)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=sim_matrix.device)
        return F.cross_entropy(logits, labels)

    def forward(
            self,
            x_raw: torch.Tensor,
            x_neg: torch.Tensor,
            x_pos: torch.Tensor,
            alpha: float = 6.0,
            beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        loss_m, x_recon = self.forward_msm(x_raw)
        loss_r = self.forward_contrastive(x_neg, x_pos) if beta > 0.0 else torch.tensor(0.0, device=x_raw.device)
        total_loss = alpha * loss_m + beta * loss_r
        return {
            "loss_total": total_loss,
            "loss_m": loss_m,
            "loss_r": loss_r,
            "x_recon": x_recon
        }


class CrossHARClassifier(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            feature_dim: int = 128,
            feature_length: int = 32,
            num_classes: int = 5,
            dropout: float = 0.3
    ):
        super().__init__()
        self.encoder = encoder
        self.context_module = TransformerContextModule(
            d_model=feature_dim, nhead=4, dropout=dropout, max_len=feature_length
        )
        self.classifier = FeatureMixerClassificationHead(
            in_features=feature_dim, num_classes=num_classes, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        c = self.context_module(z)
        return self.classifier(c)
