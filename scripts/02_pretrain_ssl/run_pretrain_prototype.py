"""
===============================================================================
SCRIPT: PRETRAIN PROTOTYPE SSL ENCODER (SwAV-INSPIRED)
Chuẩn hóa đối số và pipeline tương thích hoàn toàn với run_pretrain_contrastive.py
===============================================================================
"""

import sys
import math
import argparse
import csv
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from config.hhar_config import HHARConfig

from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from models.encoders.builder import build_encoder
from models.ssl.prototype.cluster_model import PrototypicalHARModel
from losses.swav_loss import SwAVPrototypeLoss

# Tự động điều hướng lưu Checkpoint về working directory của Kaggle
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

# ================== ARGUMENT PARSER ==================
parser = argparse.ArgumentParser(description="Pretrain Prototype SSL (SwAV-inspired) trên UCI-HAR & MotionSense")
parser.add_argument(
    "--backbone",
    type=str,
    default="standard",
    choices=["tstcc", "standard", "cnn_transformer", "vit_1d"],
    help="Loại kiến trúc backbone (mặc định: standard)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=40,
    help="Số epoch pretrain (mặc định: 40)"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size (mặc định: 64)"
)
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["uci_har", "motionsense"],
    help="Danh sách dataset cần pretrain"
)
parser.add_argument(
    "--lr",
    type=float,
    default=3e-4,
    help="Base learning rate (mặc định: 3e-4)"
)
parser.add_argument(
    "--num_prototypes",
    type=int,
    default=45,
    help="Số lượng prototype cluster K (mặc định: 45)"
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_MAP = {
    "motionsense": (
        Path(MotionSenseConfig.DATA_ALL_PATH),
        int(MotionSenseConfig.IN_CHANNELS)
    ),
    "uci_har": (
        Path(UCIHARConfig.DATA_ALL_PATH),
        int(UCIHARConfig.IN_CHANNELS)
    ),
    "hhar_phone": (
        HHARConfig.PROCESSED_DIR_PHONE / "dataset_all.pt",
        int(HHARConfig.IN_CHANNELS),
    ),
    "hhar_watch": (
        HHARConfig.PROCESSED_DIR_WATCH / "dataset_all.pt",
        int(HHARConfig.IN_CHANNELS),
    ),
}

# Tham số SwAV
FEATURE_DIM = 128
PROJECTION_DIM = 64
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-4
TAU_S = 0.1  # Softmax temperature
EPSILON = 0.05  # Sinkhorn temperature


def adjust_lr(optimizer, epoch, total_epochs, base_lr, warmup_epochs):
    """Linear warmup + Cosine annealing scheduler."""
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def plot_prototype_history(history: list, save_path: Path, title: str, max_entropy: float):
    """Vẽ 2 subplots: (1) Đường cong Loss và (2) Diễn biến Entropy."""
    epochs = [h["epoch"] for h in history]
    losses = [h["loss_total"] for h in history]
    entropies = [h["entropy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(epochs, losses, label="SwAV Loss", color="#d62728", linewidth=2.0)
    ax1.set_title(f"Prototype Pretraining Convergence: {title}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss Value", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(epochs, entropies, label="Cluster Entropy", color="#1f77b4", linewidth=1.5)
    ax2.axhline(y=max_entropy, color="gray", linestyle="-.", alpha=0.6, label=f"Max Entropy ({max_entropy:.2f})")
    ax2.axhline(y=max_entropy * 0.3, color="red", linestyle=":", alpha=0.6, label="Collapse Warning Threshold")
    ax2.set_title("Cluster Assignment Entropy", fontsize=11)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Entropy", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📈 Đã lưu biểu đồ phân tích loss và entropy tại: {save_path}")


def train_prototype_single_domain(domain_name: str, data_path: Path, in_channels: int, backbone_type: str):
    save_dir = OUTPUT_ROOT / "checkpoints" / "ssl_pretrain" / "prototype" / domain_name / backbone_type
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / f"prototype_{backbone_type}_encoder_pretrained_{domain_name}.pt"
    csv_history_path = save_dir / "loss_history.csv"
    loss_plot_path = save_dir / "loss_curve.png"

    print(f"\n🚀 ĐANG PRETRAIN PROTOTYPE: {domain_name.upper()}")
    print(f"📂 Dữ liệu: {data_path}")
    print(f"🧠 Backbone: {backbone_type.upper()} | Feature Dim: {FEATURE_DIM} | Prototypes K: {args.num_prototypes}")
    print(f"💾 Thư mục lưu: {save_dir}")
    print("-" * 65)

    if not data_path.exists():
        print(f"❌ File dữ liệu không tồn tại: {data_path}. Bỏ qua domain này!")
        return

    # 1. Dataset & DataLoader
    dataset = ContrastiveDatasetWrapper(data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 2. Xây dựng Backbone & Mô hình Prototype
    backbone = build_encoder(backbone_type=backbone_type, in_channels=in_channels).to(DEVICE)
    model = PrototypicalHARModel(
        encoder=backbone,
        feature_dim=FEATURE_DIM,
        projection_dim=PROJECTION_DIM,
        num_prototypes=args.num_prototypes,
        temperature=TAU_S
    ).to(DEVICE)

    criterion = SwAVPrototypeLoss(epsilon=EPSILON).to(DEVICE)

    # 3. Tách biệt hai optimizer: network vs prototypes
    network_params = [p for n, p in model.named_parameters() if n != "prototypes"]
    opt_network = torch.optim.AdamW(network_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    opt_proto = torch.optim.AdamW([model.prototypes], lr=args.lr, weight_decay=WEIGHT_DECAY)

    max_entropy = math.log(args.num_prototypes)
    best_loss = float("inf")
    history = []

    # 4. Vòng lặp huấn luyện
    for epoch in range(1, args.epochs + 1):
        model.train()
        cur_lr = adjust_lr(opt_network, epoch, args.epochs, args.lr, WARMUP_EPOCHS)
        adjust_lr(opt_proto, epoch, args.epochs, args.lr, WARMUP_EPOCHS)

        total_loss, entropy_acc, n_batches = 0.0, 0.0, 0

        for x_w, x_s in loader:
            x_w, x_s = x_w.to(DEVICE), x_s.to(DEVICE)

            opt_network.zero_grad()
            opt_proto.zero_grad()

            scores_w, scores_s = model(x_w, x_s)
            loss, q_w = criterion(scores_w, scores_s, return_q=True)

            loss.backward()

            opt_network.step()
            opt_proto.step()

            # Giám sát cluster entropy để phát hiện sụp cụm
            with torch.no_grad():
                avg_q = q_w.mean(dim=0)
                batch_entropy = -(avg_q * torch.log(avg_q + 1e-8)).sum().item()
                entropy_acc += batch_entropy

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_entropy = entropy_acc / max(1, n_batches)

        history.append({
            "epoch": epoch,
            "loss_total": avg_loss,
            "entropy": avg_entropy,
            "lr": cur_lr
        })

        if avg_entropy < (0.3 * max_entropy):
            print(f"⚠️  Cảnh báo sụp cụm: Entropy={avg_entropy:.2f} < {0.3 * max_entropy:.2f} tại epoch {epoch}")

        # Lưu checkpoint theo Loss tốt nhất sau warmup
        if epoch > WARMUP_EPOCHS and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] | LR: {cur_lr:.2e} | "
                f"Loss: {avg_loss:.4f} (Best: {best_loss:.4f}) | "
                f"Entropy: {avg_entropy:.2f}/{max_entropy:.2f}"
            )

    # 5. Fallback nếu chưa lưu được
    if not ckpt_path.exists():
        torch.save(model.encoder.state_dict(), ckpt_path)
        print(f"   ⚠️ Fallback: Đã lưu checkpoint tại epoch cuối.")

    # 6. Ghi log lịch sử ra CSV
    with open(csv_history_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "entropy", "lr"])
        for row in history:
            writer.writerow([
                row["epoch"], f"{row['loss_total']:.6f}", f"{row['entropy']:.6f}", f"{row['lr']:.6e}"
            ])

    print(f"📊 Đã lưu lịch sử loss tại: {csv_history_path}")

    # 7. Vẽ và lưu Plot
    plot_prototype_history(history, loss_plot_path, title=f"{backbone_type.upper()} on {domain_name}", max_entropy=max_entropy)

    print(f"   📁 Checkpoint : {ckpt_path}")
    print(f"   📉 Best Loss  : {best_loss:.5f}")
    print(f"   📊 Tổng số mẫu: {len(dataset):,}")

    del model, backbone, criterion, opt_network, opt_proto, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 80)
    print(f"🌟 BẮT ĐẦU PRETRAIN PROTOTYPE SSL TRÊN: {args.datasets} | Thiết bị: {DEVICE.upper()}")
    print(f"🧠 Backbone: {args.backbone.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | K: {args.num_prototypes}")
    print(f"📁 Output Root: {OUTPUT_ROOT}")
    print("=" * 80)

    for name in args.datasets:
        if name not in DATASET_MAP:
            print(f"⚠️ Bỏ qua dataset không hợp lệ: {name}")
            continue

        data_path, in_channels = DATASET_MAP[name]
        train_prototype_single_domain(
            domain_name=name,
            data_path=data_path,
            in_channels=in_channels,
            backbone_type=args.backbone
        )

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN TẤT CẢ DATASET CHO PROTOTYPE!")
    print("=" * 80)


if __name__ == "__main__":
    main()