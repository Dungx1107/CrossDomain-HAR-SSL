"""
===============================================================================
ENGINE: HUẤN LUYỆN SELF-SUPERVISED LEARNING (TS-TCC)
===============================================================================
- Hỗ trợ kiến trúc TS-TCC chuẩn hóa (Temporal & Contextual Contrasting).
- Tích hợp đo đạc độ phức tạp mô hình (Complexity), kiểm tra GPU và ghi log CSV.

Bố cục:
    tstcc_trainer.py   : luồng huấn luyện (file này)
    tstcc_reporting.py : config, CSV/JSON, biểu đồ, complexity, in ấn
===============================================================================
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from models.ssl.contrastive.ts_tcc_model import TSTCCModel
from models.encoders.builder import build_encoder
from config.methods.tstcc_config import TSTCCConfig
from utils.hardware import print_gpu_status
from engines.pretrain_ssl.tstcc_reporting import (
    build_config, save_json, save_training_outputs,
    print_header, log_epoch, print_summary,
)


# =============================================================================
# HELPERS — SETUP
# =============================================================================

def _build_loader(data_path, batch_size: int, num_workers: int, device: str) -> DataLoader:
    """Tạo DataLoader cho cặp view (weak, strong) của TS-TCC."""
    dataset = ContrastiveDatasetWrapper(data_path)
    use_workers = num_workers if torch.cuda.is_available() else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
        num_workers=use_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(use_workers > 0),
    )


def _build_ssl_model(
        model: Optional[nn.Module], backbone_type: str, in_channels: int,
        feature_dim: int, temperature: float, device: str
) -> nn.Module:
    """Dùng model truyền vào, hoặc dựng TSTCCModel mới từ TSTCCConfig."""
    if model is not None:
        return model.to(device)
    return TSTCCModel(
        encoder=build_encoder(backbone_type=backbone_type, in_channels=in_channels),
        in_channels=in_channels,
        feature_dim=feature_dim,
        timesteps=TSTCCConfig.TIMESTEPS,
        lambda1=TSTCCConfig.LAMBDA1,
        lambda2=TSTCCConfig.LAMBDA2,
        temperature=temperature,
        tc_hidden_dim=TSTCCConfig.TC_HIDDEN_DIM,
        tc_depth=TSTCCConfig.TC_DEPTH,
        tc_heads=TSTCCConfig.TC_HEADS,
        tc_mlp_dim=TSTCCConfig.TC_MLP_DIM,
    ).to(device)


# =============================================================================
# HELPERS — TRAIN
# =============================================================================

def _train_one_epoch(
        ssl_model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str
) -> Dict[str, float]:
    """Chạy 1 epoch, trả về loss/grad-norm trung bình theo batch."""
    ssl_model.train()
    totals = {"loss_total": 0.0, "loss_tc": 0.0, "loss_cc": 0.0, "grad_norm": 0.0}

    for x_w, x_s in loader:
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, loss_tc, loss_cc = ssl_model(x_w, x_s)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(
            ssl_model.parameters(), max_norm=TSTCCConfig.GRAD_CLIP_MAX_NORM
        )
        optimizer.step()

        totals["loss_total"] += loss.item()
        totals["loss_tc"] += loss_tc.item()
        totals["loss_cc"] += loss_cc.item()
        totals["grad_norm"] += gnorm.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# =============================================================================
# MAIN
# =============================================================================

def train_tstcc_encoder(
        data_path: Union[str, Path],
        save_dir: Union[str, Path],
        checkpoint_name: str = "best_encoder.pt",
        backbone_type: str = "standard",
        model: Optional[nn.Module] = None,
        in_channels: int = 6,
        feature_dim: int = 128,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        epochs: int = 60,
        temperature: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        measure_complexity: bool = True,
        seq_len: int = 128,
        num_workers: int = 2
) -> None:
    # ---- Đường dẫn ----
    save_dir = Path(save_dir)
    target_dir = save_dir / backbone_type
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / checkpoint_name
    dataset_name = Path(data_path).parent.name

    # ---- 1. Dữ liệu + 2. Model ----
    loader = _build_loader(data_path, batch_size, num_workers, device)
    total_samples = len(loader.dataset)
    ssl_model = _build_ssl_model(model, backbone_type, in_channels, feature_dim, temperature, device)

    # ---- 3. Config + 4. Optimizer ----
    save_json(
        build_config(
            data_path=data_path, total_samples=total_samples, in_channels=in_channels,
            seq_len=seq_len, backbone_type=backbone_type, feature_dim=feature_dim,
            temperature=temperature, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, epochs=epochs, num_workers=num_workers,
            save_dir=save_dir, target_dir=target_dir, checkpoint_name=checkpoint_name,
            device=device,
        ),
        target_dir / "config.json",
    )
    optimizer = AdamW(ssl_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ---- 5. Vòng lặp huấn luyện (lưu encoder có loss thấp nhất) ----
    print_header(dataset_name, data_path, target_dir, backbone_type, epochs, batch_size)
    best_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        metrics = _train_one_epoch(ssl_model, loader, optimizer, device)
        history.append({"epoch": epoch, **metrics})

        if metrics["loss_total"] < best_loss:
            best_loss = metrics["loss_total"]
            torch.save(ssl_model.encoder.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            log_epoch(epoch, epochs, metrics, best_loss)
            print_gpu_status()

    # ---- 6. Lưu kết quả + 7. Tóm tắt ----
    save_training_outputs(
        history, target_dir, checkpoint_path, backbone_type, dataset_name,
        in_channels, seq_len, device, measure_complexity,
    )
    print_summary(checkpoint_path, best_loss, total_samples)