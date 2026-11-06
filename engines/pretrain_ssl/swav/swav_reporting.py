"""
===============================================================================
MODULE: SWAV REPORTING (CONFIG, LOG, BIỂU ĐỒ, COMPLEXITY, TÓM TẮT)
===============================================================================
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import matplotlib.pyplot as plt

from config.methods.swav_config import SwAVConfig
from models.encoders.builder import build_encoder
from utils.complexity import ComplexityAnalyzer


def save_json(obj: dict, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def build_config(
        data_path, total_samples, in_channels, seq_len, backbone_type,
        batch_size, learning_rate, epochs, num_workers,
        save_dir, target_dir, checkpoint_name, device,
) -> dict:
    return {
        "data_path": str(data_path),
        "total_samples": total_samples,
        "in_channels": in_channels,
        "seq_len": seq_len,
        "backbone_type": backbone_type,
        "feature_dim": SwAVConfig.FEATURE_DIM,
        "projection_dim": SwAVConfig.PROJECTION_DIM,
        "num_prototypes": SwAVConfig.NUM_PROTOTYPES,
        "tau_s": SwAVConfig.TAU_S,
        "epsilon": SwAVConfig.EPSILON,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": SwAVConfig.WEIGHT_DECAY,
        "epochs": epochs,
        "warmup_epochs": SwAVConfig.WARMUP_EPOCHS,
        "freeze_prototypes_epochs": SwAVConfig.FREEZE_PROTOTYPES_EPOCHS,
        "num_workers": num_workers,
        "save_dir": str(save_dir),
        "target_dir": str(target_dir),
        "checkpoint_name": checkpoint_name,
        "device": device,
        "torch_version": torch.__version__,
    }


def save_history_csv(history: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "entropy", "lr"])
        for row in history:
            writer.writerow([
                row["epoch"],
                f"{row['loss_total']:.6f}",
                f"{row['entropy']:.6f}",
                f"{row['lr']:.6e}",
            ])
    print(f"📊 Đã lưu lịch sử loss tại: {path}")


def plot_swav_history(history: List[Dict[str, Any]], save_path: Union[str, Path], title: str):
    epochs = [h["epoch"] for h in history]
    losses = [h["loss_total"] for h in history]
    entropies = [h["entropy"] for h in history]
    max_entropy = math.log(SwAVConfig.NUM_PROTOTYPES)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Subplot 1: SwAV Loss
    ax1.plot(epochs, losses, label="SwAV Loss", color="#d62728", linewidth=2.0)
    ax1.set_title(f"SwAV Pretraining Convergence: {title}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss Value", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Subplot 2: Cluster Entropy
    ax2.plot(epochs, entropies, label="Cluster Entropy", color="#1f77b4", linewidth=1.5)
    ax2.axhline(y=max_entropy, color="gray", linestyle="-.", alpha=0.6, label=f"Max Entropy ({max_entropy:.2f})")
    ax2.axhline(y=max_entropy * SwAVConfig.MAX_ENTROPY_COLLAPSE_THRESHOLD,
                color="red", linestyle=":", alpha=0.6, label="Collapse Warning")
    ax2.set_title("Cluster Assignment Entropy", fontsize=11)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Entropy", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📈 Đã lưu biểu đồ phân tích tại: {save_path}")


def save_training_outputs(
        history: List[Dict[str, Any]],
        target_dir: Union[str, Path],
        checkpoint_path: Union[str, Path],
        backbone_type: str,
        dataset_name: str,
        in_channels: int,
        seq_len: int,
        device: str,
        measure_complexity: bool,
) -> None:
    target_dir = Path(target_dir)

    save_history_csv(history, target_dir / "loss_history.csv")
    plot_swav_history(
        history, target_dir / "loss_curve.png",
        title=f"{backbone_type.upper()} on {dataset_name}",
    )

    if measure_complexity:
        complexity_info = ComplexityAnalyzer.measure_from_checkpoint(
            checkpoint_path=checkpoint_path,
            backbone_type=backbone_type,
            in_channels=in_channels,
            input_size=(1, in_channels, seq_len),
            device=device,
            build_fn=build_encoder,
        )
        save_json(complexity_info, target_dir / "complexity.json")
        print(f"📊 Đã lưu độ phức tạp tại: {target_dir / 'complexity.json'}")


# =============================================================================
# CONSOLE LOGGING
# =============================================================================

def print_header(dataset_name, data_path, target_dir, backbone_type, epochs, batch_size) -> None:
    print("=" * 80)
    print(f"🚀 PRETRAIN SSL (SwAV) | Dataset: {dataset_name.upper()}")
    print(f"   📂 Nguồn      : {data_path}")
    print(f"   📁 Lưu tại    : {target_dir}")
    print(f"   🧠 Backbone: {backbone_type.upper()} | K: {SwAVConfig.NUM_PROTOTYPES} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 80)


def log_epoch(epoch: int, epochs: int, metrics: Dict[str, float], best_loss: float, max_entropy: float) -> None:
    print(
        f"Epoch [{epoch:03d}/{epochs:03d}] | LR: {metrics['lr']:.2e} | "
        f"Loss: {metrics['loss_total']:.5f} (Best: {best_loss:.5f}) | "
        f"Entropy: {metrics['entropy']:.2f}/{max_entropy:.2f}"
    )


def print_summary(checkpoint_path, best_loss: float, total_samples: int) -> None:
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN SWAV")
    print(f"   📁 Checkpoint : {checkpoint_path}")
    print(f"   📉 Best Loss  : {best_loss:.5f}")
    print(f"   📊 Tổng số mẫu: {total_samples:,}")
    print("=" * 80)