"""
===============================================================================
MODULE: FINE-TUNE REPORTING
===============================================================================
Mọi thứ liên quan đến IN log và LƯU kết quả của quá trình fine-tune:
    - log_epoch / print_setup / print_test_report : in ra console
    - build_finetune_metadata                      : gom metadata để lưu .json
    - save_checkpoint_and_metadata                 : ghi .pt + _metadata.json
===============================================================================
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from engines.transfer.model_builder import FREEZE_FULL, FREEZE_PARTIAL


# =============================================================================
# CONSOLE
# =============================================================================

def print_setup(protocol_name: str, ckpt_name: str, num_classes: int, in_channels: int) -> None:
    print("=" * 80)
    print(f"🚀 THIẾT LẬP THỰC NGHIỆM: {protocol_name}")
    print(f"📦 Checkpoint SSL nguồn: {ckpt_name}")
    print(f"🎯 Số lớp đích (Target Classes): {num_classes} | Kênh vào (Channels): {in_channels}")


def log_epoch(
        epoch: int, epochs: int, train_loss: float,
        val_metrics: Optional[Dict[str, float]] = None, best_val_f1: float = -1.0
) -> None:
    if val_metrics is None:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (No Val)")
    else:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:5.2f}% | "
              f"Val Macro F1: {val_metrics['macro_f1']:5.2f}% (Best: {best_val_f1:5.2f}%)")


def print_test_report(protocol_name: str, m: Dict[str, float]) -> None:
    print("-" * 80)
    print(f"🎯 KẾT QUẢ ĐÁNH GIÁ TẬP TEST ({protocol_name}):")
    print(f"   - Test Accuracy : {m['accuracy']:6.2f}%")
    print(f"   - Test Precision : {m['macro_precision']:6.2f}%")
    print(f"   - Test Recall    : {m['macro_recall']:6.2f}%")
    print(f"   - Test Macro F1 : {m['macro_f1']:6.2f}% (Chỉ số nghiên cứu chính)")
    print("-" * 80)


# =============================================================================
# METADATA / LƯU TRỮ
# =============================================================================

def build_finetune_metadata(
        protocol_name: str, freeze_mode: str, num_unfrozen_blocks: int,
        encoder_ckpt_path: Union[str, Path], save_model_path: Optional[Union[str, Path]],
        complexity_info: Dict[str, Any], best_val_f1: float, test_metrics: Dict[str, float],
        total_train_time_sec: float, epochs: int, inference_time_info: Dict[str, Any],
        training_history: list
) -> dict:
    """Tạo dict metadata tổng hợp để lưu trữ."""
    return {
        "protocol": protocol_name,
        "freeze_mode": freeze_mode,
        "freeze_backbone": freeze_mode == FREEZE_FULL,  # giữ key cũ cho script tổng hợp
        "num_unfrozen_blocks": num_unfrozen_blocks if freeze_mode == FREEZE_PARTIAL else None,
        "source_checkpoint": str(encoder_ckpt_path),
        "saved_model_path": str(save_model_path) if save_model_path else None,
        "model_complexity": {
            "total_params": complexity_info["total_params"],
            "trainable_params": complexity_info["trainable_params"],
            "trainable_percent": round(
                (complexity_info["trainable_params"] / max(complexity_info["total_params"], 1)) * 100, 2),
            "macs": complexity_info["macs"],
            "flops": complexity_info["flops"],
        },
        "performance": {
            "best_val_f1": round(best_val_f1, 2) if best_val_f1 != -1.0 else "N/A",
            "test_accuracy": round(test_metrics["accuracy"], 2),
            "test_macro_f1": round(test_metrics["macro_f1"], 2),
            "test_macro_precision": round(test_metrics["macro_precision"], 2),
            "test_macro_recall": round(test_metrics["macro_recall"], 2)
        },
        "timing": {
            "total_train_time_sec": round(total_train_time_sec, 2),
            "avg_epoch_time_sec": round(total_train_time_sec / epochs, 2),
            "total_test_inference_sec": inference_time_info["total_inference_time_sec"],
            "avg_inference_ms_per_sample": inference_time_info["avg_inference_ms_per_sample_full_test"]
        },
        "history": training_history
    }


def save_checkpoint_and_metadata(
        save_model_path: Optional[Union[str, Path]],
        model_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any]
) -> float:
    """Lưu file trọng số .pt và file metadata .json. Trả về dung lượng (KB)."""
    checkpoint_file_size_kb = 0.0

    if save_model_path is not None and model_state is not None:
        save_path = Path(save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model_state, save_path)
        checkpoint_file_size_kb = round(save_path.stat().st_size / 1024, 2)
        print(f"💾 Đã lưu Checkpoint mô hình tại: {save_path} ({checkpoint_file_size_kb:.1f} KB)")

        meta_path = save_path.parent / f"{save_path.stem}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"📄 Đã lưu Metadata (báo cáo) tại: {meta_path}")

    return checkpoint_file_size_kb