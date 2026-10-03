"""
===============================================================================
ENGINE: HUẤN LUYỆN PRETRAIN PHÂN CẤP TUẦN TỰ (CROSSHAR SEQUENTIAL TRAINER)
===============================================================================
- Tự động lấy động feature_dim và feature_length từ encoder backbone.
- Reset best_val_loss khi chuyển từ Stage A (MSM) sang Stage B (Joint 6*MSM + 1*CR).
- Tương thích hoàn toàn với AdaptiveConv1DDecoder.
===============================================================================
"""

import sys
import csv
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.builder import build_encoder
from models.ssl.masked.crosshar_model import CrossHARPretrainModel


def train_crosshar_sequential(
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Union[str, Path],
        checkpoint_name: str = "best_crosshar_encoder.pt",
        backbone_type: str = "cnn_transformer",
        in_channels: int = 6,
        epochs: int = 60,
        warmup_msm_epochs: int = 15,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        mask_ratio: float = 0.15,
        temperature: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / checkpoint_name
    csv_history_path = save_dir / "pretrain_loss_history.csv"
    loss_plot_path = save_dir / "pretrain_loss_curve.png"

    # 1. XÁC ĐỊNH ĐỘNG FEATURE_DIM VÀ FEATURE_LENGTH TỪ BACKBONE
    temp_encoder = build_encoder(backbone_type=backbone_type, in_channels=in_channels)
    with torch.no_grad():
        dummy_in = torch.randn(2, in_channels, 128)
        dummy_out = temp_encoder(dummy_in)
        detected_feat_dim = dummy_out.shape[1]
        detected_feat_len = dummy_out.shape[2]
    del temp_encoder

    print(f"🔍 [Pretrain Auto-Config] Backbone: {backbone_type.upper()} -> Output shape: (B, {detected_feat_dim}, {detected_feat_len})")

    # 2. KHỞI TẠO MÔ HÌNH PRETRAIN VỚI KÍCH THƯỚC CHUẨN XÁC
    model = CrossHARPretrainModel(
        backbone_type=backbone_type,
        in_channels=in_channels,
        feature_dim=detected_feat_dim,
        feature_length=detected_feat_len,
        mask_ratio=mask_ratio,
        temperature=temperature
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = []

    print("=" * 85)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN PRETRAIN CHUẨN CROSSHAR (SEQUENTIAL UPDATING)")
    print(f"📦 Backbone: {backbone_type.upper()} | Kênh: {in_channels} | Thiết bị: {device.upper()}")
    print(f"⏱️ Tổng: {epochs} epochs (Giai đoạn A - Chỉ MSM: {warmup_msm_epochs} | Giai đoạn B: {epochs - warmup_msm_epochs})")
    print("=" * 85)

    for epoch in range(1, epochs + 1):
        if epoch <= warmup_msm_epochs:
            current_stage = "Stage A (MSM Only)"
            alpha = 6.0
            beta = 0.0
        else:
            if epoch == warmup_msm_epochs + 1:
                print("\n" + "#" * 80)
                print("🔄 BƯỚC SANG GIAI ĐOẠN B (JOINT: 6*MSM + 1*CR) -> RESET BEST VAL LOSS")
                print("#" * 80)
                best_val_loss = float("inf")
            current_stage = "Stage B (Joint: 6*MSM + 1*CR)"
            alpha = 6.0
            beta = 1.0

        model.train()
        train_total, train_m, train_r = 0.0, 0.0, 0.0
        batches = 0

        for x_raw, x_neg, x_pos in train_loader:
            x_raw = x_raw.to(device, non_blocking=True)
            x_neg = x_neg.to(device, non_blocking=True)
            x_pos = x_pos.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(x_raw, x_neg, x_pos, alpha=alpha, beta=beta)
            loss = out["loss_total"]
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_total += loss.item()
            train_m += out["loss_m"].item()
            train_r += out["loss_r"].item()
            batches += 1

        avg_train_total = train_total / max(batches, 1)
        avg_train_m = train_m / max(batches, 1)
        avg_train_r = train_r / max(batches, 1)

        model.eval()
        val_total, val_m, val_r = 0.0, 0.0, 0.0
        val_batches = 0

        with torch.no_grad():
            for v_raw, v_neg, v_pos in val_loader:
                v_raw = v_raw.to(device, non_blocking=True)
                v_neg = v_neg.to(device, non_blocking=True)
                v_pos = v_pos.to(device, non_blocking=True)

                v_out = model(v_raw, v_neg, v_pos, alpha=alpha, beta=beta)
                val_total += v_out["loss_total"].item()
                val_m += v_out["loss_m"].item()
                val_r += v_out["loss_r"].item()
                val_batches += 1

        avg_val_total = val_total / max(val_batches, 1)
        avg_val_m = val_m / max(val_batches, 1)
        avg_val_r = val_r / max(val_batches, 1)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "stage": current_stage,
            "train_loss_total": avg_train_total,
            "train_loss_m": avg_train_m,
            "train_loss_r": avg_train_r,
            "val_loss_total": avg_val_total,
            "val_loss_m": avg_val_m,
            "val_loss_r": avg_val_r,
            "lr": curr_lr
        })

        eval_criterion = avg_val_m if beta == 0.0 else avg_val_total
        if eval_criterion < best_val_loss:
            best_val_loss = eval_criterion
            torch.save(model.encoder.state_dict(), checkpoint_path)
            saved_msg = f" ⭐ [ĐÃ LƯU CHECKPOINT EPOCH {epoch}]"
        else:
            saved_msg = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == warmup_msm_epochs + 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({current_stage}) | "
                  f"Train: {avg_train_total:.4f} (Lm: {avg_train_m:.4f}, Lr: {avg_train_r:.4f}) | "
                  f"Val: {avg_val_total:.4f} (Lm: {avg_val_m:.4f}, Lr: {avg_val_r:.4f}) | Best: {best_val_loss:.4f}{saved_msg}")

    with open(csv_history_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "stage", "train_total", "train_lm", "train_lr", "val_total", "val_lm", "val_lr"])
        for r in history:
            writer.writerow([
                r["epoch"], r["stage"],
                f"{r['train_loss_total']:.5f}", f"{r['train_loss_m']:.5f}", f"{r['train_loss_r']:.5f}",
                f"{r['val_loss_total']:.5f}", f"{r['val_loss_m']:.5f}", f"{r['val_loss_r']:.5f}"
            ])

    epochs_idx = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(epochs_idx, [h["train_loss_total"] for h in history], label="Train Total", color="blue", lw=2)
    ax1.plot(epochs_idx, [h["val_loss_total"] for h in history], label="Val Total", color="red", linestyle="--", lw=2)
    ax1.axvline(x=warmup_msm_epochs, color="gray", linestyle="-.", label="Stage B Start")
    ax1.set_title(f"CrossHAR Sequential Pretraining Convergence ({backbone_type})", fontweight="bold")
    ax1.set_ylabel("Total Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs_idx, [h["val_loss_m"] for h in history], label="Val L_m (Reconstruction)", color="green", lw=1.8)
    ax2.plot(epochs_idx, [h["val_loss_r"] for h in history], label="Val L_r (Contrastive)", color="purple", lw=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Component Losses")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()

    print(f"\n💾 Trọng số Encoder tối ưu nhất đã lưu tại: {checkpoint_path}")
    print(f"📈 Biểu đồ lịch sử Loss đã lưu tại: {loss_plot_path}")

    return {
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
        "history": history
    }
