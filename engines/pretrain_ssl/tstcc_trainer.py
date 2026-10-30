"""
===============================================================================
ENGINE: HUẤN LUYỆN SELF-SUPERVISED LEARNING (TS-TCC)
===============================================================================
- Hỗ trợ kiến trúc TS-TCC chuẩn hóa (Temporal & Contextual Contrasting).
- Tích hợp đo đạc độ phức tạp mô hình (Complexity), kiểm tra GPU và ghi log CSV.
===============================================================================
"""

import csv
import json
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from models.ssl.contrastive.ts_tcc_model import TSTCCModel
from utils.hardware import print_gpu_status
from utils.complexity import measure_encoder_from_checkpoint
from models.encoders.builder import build_encoder
from config.methods.tstcc_config import TSTCCConfig
from utils.plotting import PlottingUtils


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
    # ---- Chuẩn bị đường dẫn ----
    save_dir = Path(save_dir)
    target_dir = save_dir / backbone_type
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / checkpoint_name

    # ---- 1. NẠP DỮ LIỆU ----
    dataset = ContrastiveDatasetWrapper(data_path)
    use_workers = num_workers if torch.cuda.is_available() else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
        num_workers=use_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(use_workers > 0),
    )

    # ---- 2. KHỞI TẠO MODEL ----
    if model is None:
        ssl_model = TSTCCModel(
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
    else:
        ssl_model = model.to(device)

    # ---- 3. LƯU CONFIG ----
    config = _build_config(
        data_path=data_path, total_samples=len(dataset), in_channels=in_channels,
        seq_len=seq_len, backbone_type=backbone_type, feature_dim=feature_dim,
        temperature=temperature, batch_size=batch_size, learning_rate=learning_rate,
        weight_decay=weight_decay, epochs=epochs, num_workers=num_workers,
        save_dir=save_dir, target_dir=target_dir, checkpoint_name=checkpoint_name,
        device=device,
    )
    _save_json(config, target_dir / "config.json")

    # ---- 4. OPTIMIZER ----
    optimizer = AdamW(ssl_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ---- 5. VÒNG LẶP HUẤN LUYỆN ----
    best_loss = float("inf")
    history = []

    dataset_name = Path(data_path).parent.name
    print("=" * 80)
    print(f"🚀 PRETRAIN SSL (TS-TCC) | Dataset: {dataset_name.upper()}")
    print(f"   📂 Nguồn      : {data_path}")
    print(f"   📁 Lưu tại    : {target_dir}")
    print(f"   🧠 Backbone: {backbone_type.upper()} | Epochs     : {epochs} | Batch: {batch_size}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss = total_tc = total_cc = total_gnorm = 0.0

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

            total_loss += loss.item()
            total_tc += loss_tc.item()
            total_cc += loss_cc.item()
            total_gnorm += gnorm.item()

        n = len(loader)
        avg_loss, avg_tc, avg_cc, avg_gnorm = (
            total_loss / n, total_tc / n, total_cc / n, total_gnorm / n
        )
        history.append({
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_tc": avg_tc,
            "loss_cc": avg_cc,
            "grad_norm": avg_gnorm,
        })

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ssl_model.encoder.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Loss Total: {avg_loss:.5f} (TC: {avg_tc:.4f}, CC: {avg_cc:.4f}) | "
                f"Best: {best_loss:.5f}"
            )
            print_gpu_status()

    # ---- 6. LƯU KẾT QUẢ ----
    _save_history_csv(history, target_dir / "loss_history.csv")

    dataset_name = Path(data_path).parent.name
    PlottingUtils.plot_tstcc_loss_history(
        history, target_dir / "loss_curve.png",
        title=f"{backbone_type.upper()} on {dataset_name}",
    )

    if measure_complexity:
        complexity_info = measure_encoder_from_checkpoint(
            checkpoint_path=checkpoint_path,
            backbone_type=backbone_type,
            in_channels=in_channels,
            input_size=(1, in_channels, seq_len),
            device=device,
            build_fn=build_encoder,
        )
        _save_json(complexity_info, target_dir / "complexity.json")
        print(f"📊 Đã lưu độ phức tạp tại: {target_dir / 'complexity.json'}")

    # ---- 7. IN TÓM TẮT ----
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN TS-TCC")
    print(f"   📁 Checkpoint : {checkpoint_path}")
    print(f"   📉 Best Loss  : {best_loss:.5f}")
    print(f"   📊 Tổng số mẫu: {len(dataset):,}")
    print("=" * 80)


# =============================================================================
# HELPERS — SETUP / LƯU TRỮ
# =============================================================================

def _build_config(
        data_path, total_samples, in_channels, seq_len, backbone_type,
        feature_dim, temperature, batch_size, learning_rate, weight_decay,
        epochs, num_workers, save_dir, target_dir, checkpoint_name, device,
) -> dict:
    """Tạo dict config đầy đủ để tái lập + audit."""
    return {
        "data_path": str(data_path),
        "total_samples": total_samples,
        "in_channels": in_channels,
        "seq_len": seq_len,
        "backbone_type": backbone_type,
        "feature_dim": feature_dim,
        "timesteps": TSTCCConfig.TIMESTEPS,
        "lambda1": TSTCCConfig.LAMBDA1,
        "lambda2": TSTCCConfig.LAMBDA2,
        "temperature": temperature,
        "tc_hidden_dim": TSTCCConfig.TC_HIDDEN_DIM,
        "tc_depth": TSTCCConfig.TC_DEPTH,
        "tc_heads": TSTCCConfig.TC_HEADS,
        "tc_mlp_dim": TSTCCConfig.TC_MLP_DIM,
        "grad_clip_max_norm": TSTCCConfig.GRAD_CLIP_MAX_NORM,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "num_workers": num_workers,
        "save_dir": str(save_dir),
        "target_dir": str(target_dir),
        "checkpoint_name": checkpoint_name,
        "device": device,
        "torch_version": torch.__version__,
    }


def _save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def _save_history_csv(history: list, path: Path):
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "loss_tc", "loss_cc", "grad_norm"])
        for row in history:
            writer.writerow([
                row["epoch"],
                f"{row['loss_total']:.6f}",
                f"{row['loss_tc']:.6f}",
                f"{row['loss_cc']:.6f}",
                f"{row['grad_norm']:.4f}",
            ])
    print(f"📊 Đã lưu lịch sử loss tại: {path}")
