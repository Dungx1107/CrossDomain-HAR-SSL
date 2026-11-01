"""
===============================================================================
ENGINE: HUẤN LUYỆN SSL LAI TS-TCC x SwAV
===============================================================================
Đặt file tại: engines/pretrain_ssl/tstcc_swav/tstcc_swav_trainer.py

Khác trainer TS-TCC:
    - Model: TSTCCSwAVModel (loss_tc + loss_swav), trả (loss_total, loss_tc, loss_swav).
    - Prototype không weight decay (vì luôn được L2-normalize).
    - Không lưu "best" trong WARMUP_EPOCHS đầu; luôn lưu thêm encoder epoch cuối.
    - Theo dõi collapse: proto_entropy (tối đa ln K) và proto_used mỗi epoch.
    - Kiểm tra encoder trước khi train (3D, đúng số kênh, đủ độ dài cho TC).
    - Ghi log/config tự chứa (CSV + JSON), chưa dùng tstcc_reporting.py vì các key
      loss khác nhau (loss_swav thay cho loss_cc).
===============================================================================
"""

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config.methods.tstcc_config import TSTCCConfig
from config.methods.tstcc_swav_config import TSTCCSwAVConfig as SwCfg
from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from models.encoders.builder import build_encoder
from models.ssl.hybrid.tstcc_swav_model import TSTCCSwAVModel
from utils.hardware import print_gpu_status


# =============================================================================
# HELPERS — SETUP
# =============================================================================

def _build_loader(data_path, batch_size: int, num_workers: int, device: str) -> DataLoader:
    dataset = ContrastiveDatasetWrapper(data_path)
    use_workers = num_workers if torch.cuda.is_available() else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,   # Sinkhorn + BatchNorm cần batch đầy
        num_workers=use_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(use_workers > 0),
    )


def _check_encoder(encoder: nn.Module, in_channels: int, seq_len: int,
                   feature_dim: int, device: str) -> int:
    """Chạy thử 1 batch giả; trả về độ dài chuỗi sau encoder (T_out)."""
    encoder.eval()
    with torch.no_grad():
        out = encoder(torch.zeros(2, in_channels, seq_len, device=device))
    if out.dim() != 3 or out.shape[1] != feature_dim:
        raise ValueError(
            f"TC cần encoder trả (B, {feature_dim}, T); nhận được {tuple(out.shape)}."
        )
    t_out = out.shape[2]
    if t_out - TSTCCConfig.TIMESTEPS <= SwCfg.MIN_T:
        raise ValueError(
            f"T_out={t_out} quá ngắn cho TIMESTEPS={TSTCCConfig.TIMESTEPS}, MIN_T={SwCfg.MIN_T}."
        )
    return t_out


def _build_ssl_model(model: Optional[nn.Module], backbone_type: str, in_channels: int,
                     feature_dim: int, device: str) -> nn.Module:
    if model is not None:
        return model.to(device)
    return TSTCCSwAVModel(
        encoder=build_encoder(backbone_type=backbone_type, in_channels=in_channels),
        in_channels=in_channels,
        feature_dim=feature_dim,
        timesteps=TSTCCConfig.TIMESTEPS,
        tc_hidden_dim=TSTCCConfig.TC_HIDDEN_DIM,
        tc_depth=TSTCCConfig.TC_DEPTH,
        tc_heads=TSTCCConfig.TC_HEADS,
        tc_mlp_dim=TSTCCConfig.TC_MLP_DIM,
        num_prototypes=SwCfg.NUM_PROTOTYPES,
        tau=SwCfg.TAU,
        epsilon=SwCfg.EPSILON,
        sinkhorn_iters=SwCfg.SINKHORN_ITERS,
        lambda_tc=SwCfg.LAMBDA_TC,
        lambda_swav=SwCfg.LAMBDA_SWAV,
        lambda_cc=SwCfg.LAMBDA_CC,
        cc_temperature=SwCfg.CC_TEMPERATURE,
        min_t=SwCfg.MIN_T,
    ).to(device)


def _build_optimizer(ssl_model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    others = [p for n, p in ssl_model.named_parameters() if n != "prototypes"]
    return AdamW(
        [
            {"params": others, "weight_decay": weight_decay},
            {"params": [ssl_model.prototypes], "weight_decay": 0.0},
        ],
        lr=lr,
    )


# =============================================================================
# HELPERS — TRAIN
# =============================================================================

def _train_one_epoch(ssl_model: nn.Module, loader: DataLoader,
                     optimizer: torch.optim.Optimizer, device: str) -> Dict[str, float]:
    ssl_model.train()
    keys = ["loss_total", "loss_tc", "loss_swav", "grad_norm", "proto_entropy", "proto_used"]

    # Mọi metric tích lũy ngay trên GPU, chỉ đồng bộ 1 lần cuối epoch
    # (tránh .item() mỗi step làm GPU phải chờ CPU).
    totals = torch.zeros(len(keys), device=device)
    all_finite = torch.ones((), dtype=torch.bool, device=device)

    data_wait = 0.0                      # thời gian CPU chờ DataLoader trả batch
    t_prev = time.perf_counter()
    for x_w, x_s in loader:
        data_wait += time.perf_counter() - t_prev

        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_tc, loss_swav = ssl_model(x_w, x_s)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(
            ssl_model.parameters(), max_norm=TSTCCConfig.GRAD_CLIP_MAX_NORM
        )
        optimizer.step()

        st = ssl_model.last_stats
        totals += torch.stack([
            loss.detach(), loss_tc.detach(), loss_swav.detach(), gnorm.detach(),
            st["proto_entropy"], st["proto_used"],
        ])
        all_finite &= torch.isfinite(loss.detach())
        t_prev = time.perf_counter()

    if not all_finite.item():
        raise RuntimeError("Loss không hữu hạn (NaN/Inf) trong epoch — dừng huấn luyện.")

    n = len(loader)
    out = {k: v / n for k, v in zip(keys, totals.tolist())}
    out["data_time"] = data_wait
    return out


def _save_csv(history, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# =============================================================================
# MAIN
# =============================================================================

def train_tstcc_swav_encoder(
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seq_len: int = 128,
        num_workers: int = 2,
) -> None:
    save_dir = Path(save_dir)
    target_dir = save_dir / backbone_type
    target_dir.mkdir(parents=True, exist_ok=True)
    best_path = target_dir / checkpoint_name
    last_path = target_dir / f"last_{checkpoint_name}"
    dataset_name = Path(data_path).parent.name

    loader = _build_loader(data_path, batch_size, num_workers, device)
    ssl_model = _build_ssl_model(model, backbone_type, in_channels, feature_dim, device)
    t_out = _check_encoder(ssl_model.encoder, in_channels, seq_len, feature_dim, device)
    t_max = t_out - TSTCCConfig.TIMESTEPS - 1

    with open(target_dir / "config.json", "w") as f:
        json.dump({
            "dataset": dataset_name, "data_path": str(data_path),
            "total_samples": len(loader.dataset), "backbone": backbone_type,
            "encoder_seq_len": t_out, "t_samples_range": [SwCfg.MIN_T, t_max],
            "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay,
            "epochs": epochs, "timesteps": TSTCCConfig.TIMESTEPS,
            "num_prototypes": SwCfg.NUM_PROTOTYPES, "tau": SwCfg.TAU,
            "epsilon": SwCfg.EPSILON, "lambda_tc": SwCfg.LAMBDA_TC,
            "lambda_swav": SwCfg.LAMBDA_SWAV, "lambda_cc": SwCfg.LAMBDA_CC,
            "warmup_epochs": SwCfg.WARMUP_EPOCHS,
        }, f, indent=2)

    optimizer = _build_optimizer(ssl_model, learning_rate, weight_decay)

    print(f"[TS-TCC x SwAV] {dataset_name} | backbone={backbone_type} | T_out={t_out} "
          f"| t in [{SwCfg.MIN_T}, {t_max}] | K={SwCfg.NUM_PROTOTYPES} "
          f"| epochs={epochs} | batch={batch_size}")

    best_loss = float("inf")
    history = []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t_ep = time.perf_counter()
        m = _train_one_epoch(ssl_model, loader, optimizer, device)   # đã sync cuối epoch
        m["epoch_time"] = time.perf_counter() - t_ep
        history.append({"epoch": epoch, **m})

        if epoch > SwCfg.WARMUP_EPOCHS and m["loss_total"] < best_loss:
            best_loss = m["loss_total"]
            torch.save(ssl_model.encoder.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | total {m['loss_total']:.4f} "
                  f"(tc {m['loss_tc']:.4f}, swav {m['loss_swav']:.4f}) "
                  f"| entropy {m['proto_entropy']:.2f}/{math.log(SwCfg.NUM_PROTOTYPES):.2f} "
                  f"| used {m['proto_used']:.1f}/{SwCfg.NUM_PROTOTYPES} "
                  f"| gnorm {m['grad_norm']:.2f} "
                  f"| {m['epoch_time']:.1f}s/epoch (chờ data {m['data_time']:.1f}s) "
                  f"| ETA {(epochs - epoch) * m['epoch_time'] / 60:.1f} phút")
            print_gpu_status()

    torch.save(ssl_model.encoder.state_dict(), last_path)
    if not best_path.exists():       # epochs <= warmup
        torch.save(ssl_model.encoder.state_dict(), best_path)
    _save_csv(history, target_dir / "training_log.csv")

    total_sec = time.perf_counter() - t_start
    with open(target_dir / "pretrain_summary.json", "w") as f:
        json.dump({
            "total_pretrain_sec": round(total_sec, 1),
            "avg_epoch_sec": round(total_sec / epochs, 2),
            "avg_data_wait_sec_per_epoch": round(sum(h["data_time"] for h in history) / epochs, 2),
            "best_loss_after_warmup": None if best_loss == float("inf") else round(best_loss, 5),
            "final_proto_entropy": round(history[-1]["proto_entropy"], 3),
            "final_proto_used": round(history[-1]["proto_used"], 1),
        }, f, indent=2)
    print(f"Xong trong {total_sec / 60:.1f} phút ({total_sec / epochs:.1f}s/epoch). "
          f"best loss {best_loss:.4f} | {best_path} | {last_path}")