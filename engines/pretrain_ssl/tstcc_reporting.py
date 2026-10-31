"""
===============================================================================
MODULE: TS-TCC REPORTING (CONFIG, LOG, BIỂU ĐỒ, COMPLEXITY, TÓM TẮT)
===============================================================================
Mọi thứ liên quan đến GHI / IN kết quả của quá trình pretrain TS-TCC nằm ở đây,
để `tstcc_trainer.py` chỉ giữ luồng: nạp dữ liệu -> khởi tạo -> huấn luyện.
===============================================================================
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch

from config.methods.tstcc_config import TSTCCConfig
from models.encoders.builder import build_encoder
from utils.complexity import ComplexityAnalyzer
from utils.plotting import PlottingUtils


# =============================================================================
# CONFIG / FILE
# =============================================================================

def save_json(obj: dict, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def build_config(
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


def save_history_csv(history: List[Dict[str, Any]], path: Union[str, Path]) -> None:
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
    """Sau khi train xong: lưu CSV loss, vẽ đường cong loss, đo độ phức tạp encoder."""
    target_dir = Path(target_dir)

    save_history_csv(history, target_dir / "loss_history.csv")
    PlottingUtils.plot_tstcc_loss_history(
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
# CONSOLE
# =============================================================================

def print_header(dataset_name, data_path, target_dir, backbone_type, epochs, batch_size) -> None:
    print("=" * 80)
    print(f"🚀 PRETRAIN SSL (TS-TCC) | Dataset: {dataset_name.upper()}")
    print(f"   📂 Nguồn      : {data_path}")
    print(f"   📁 Lưu tại    : {target_dir}")
    print(f"   🧠 Backbone: {backbone_type.upper()} | Epochs     : {epochs} | Batch: {batch_size}")
    print("=" * 80)


def log_epoch(epoch: int, epochs: int, metrics: Dict[str, float], best_loss: float) -> None:
    print(
        f"Epoch [{epoch:03d}/{epochs:03d}] | "
        f"Loss Total: {metrics['loss_total']:.5f} "
        f"(TC: {metrics['loss_tc']:.4f}, CC: {metrics['loss_cc']:.4f}) | "
        f"Best: {best_loss:.5f}"
    )


def print_summary(checkpoint_path, best_loss: float, total_samples: int) -> None:
    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH PRETRAIN TS-TCC")
    print(f"   📁 Checkpoint : {checkpoint_path}")
    print(f"   📉 Best Loss  : {best_loss:.5f}")
    print(f"   📊 Tổng số mẫu: {total_samples:,}")
    print("=" * 80)