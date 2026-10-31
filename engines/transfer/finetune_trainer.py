"""
===============================================================================
MODULE: FINE-TUNE & ADAPTATION ENGINE (CROSS-DOMAIN & FEW-LABEL HAR)
===============================================================================
Bố cục (engines/transfer/):
    model_builder.py     : lắp ráp model, nạp trọng số SSL, đóng băng + optimizer
    finetune_trainer.py  : luồng train -> (validate) -> test (file này)
    (engines/evaluation/evaluator.py: predict + compute_metrics, dùng chung val/test)
    finetune_reporting.py: in log, build metadata, lưu checkpoint/json

Hỗ trợ 2 chế độ:
    - Có val_loader  : chọn epoch tốt nhất theo Val Macro F1 (scheduler theo Val F1).
    - val_loader=None: Fixed-Epoch, lấy trọng số epoch cuối (scheduler theo Train Loss).
===============================================================================
"""

import sys
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.har_classifier import HARClassifier
from engines.transfer.model_builder import (
    build_transfer_model,
    configure_optimizer,
    resolve_freeze_mode,
    FREEZE_NONE,
    FREEZE_FULL,
    FREEZE_PARTIAL,
)
from engines.evaluation.evaluator import predict, compute_metrics
from engines.transfer.finetune_reporting import (
    print_setup, log_epoch, print_test_report,
    build_finetune_metadata, save_checkpoint_and_metadata,
)
from utils.complexity import ComplexityAnalyzer

LINEAR_PROBING = "LINEAR PROBING"
FULL_FINE_TUNING = "FULL FINE-TUNING"
PARTIAL_FINE_TUNING = "PARTIAL FINE-TUNING"

_PROTOCOL_NAMES = {
    FREEZE_NONE: FULL_FINE_TUNING,
    FREEZE_FULL: LINEAR_PROBING,
    FREEZE_PARTIAL: PARTIAL_FINE_TUNING,
}


# =============================================================================
# HELPERS — TRAIN
# =============================================================================

def _build_scheduler(optimizer: torch.optim.Optimizer, has_val: bool) -> ReduceLROnPlateau:
    """Có Val: theo dõi Val F1 (max). Không Val: theo dõi Train Loss (min)."""
    return ReduceLROnPlateau(optimizer, mode="max" if has_val else "min", factor=0.5, patience=5)


def _train_one_epoch(
        model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
        criterion: nn.Module, device: str
) -> float:
    """Chạy 1 epoch, trả về train loss trung bình theo batch."""
    model.train()
    total_loss, num_batches = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def _snapshot_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Bản sao state_dict trên CPU."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


# =============================================================================
# MAIN
# =============================================================================

def train_and_eval_finetune(
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: DataLoader,
        encoder_checkpoint_path: Union[str, Path],
        encoder: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
        num_classes: int = 5,
        in_channels: int = 6,
        feature_dim: int = 128,
        seq_len: int = 128,
        epochs: int = 40,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,  # Giữ lại để tương thích code cũ
        freeze_mode: Optional[str] = None,  # "none" | "full" | "partial"
        num_unfrozen_blocks: int = 1,  # Chỉ dùng khi freeze_mode="partial"
        save_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
) -> Tuple[float, float, HARClassifier, Dict[str, Any]]:
    """
    Thực thi quá trình huấn luyện chuyển giao (Transfer Learning) và đánh giá mô hình.
    """
    target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    encoder_ckpt_path = Path(encoder_checkpoint_path)
    mode = resolve_freeze_mode(freeze_mode, freeze_backbone)
    protocol_name = _PROTOCOL_NAMES[mode]
    has_val = val_loader is not None

    # 1. LẮP RÁP MÔ HÌNH + NẠP TRỌNG SỐ SSL
    model = build_transfer_model(
        encoder_checkpoint_path=encoder_ckpt_path, encoder=encoder, classifier=classifier,
        num_classes=num_classes, in_channels=in_channels, feature_dim=feature_dim,
        device=target_device
    )
    print_setup(protocol_name, encoder_ckpt_path.name, num_classes, in_channels)

    # 2. ĐÓNG BĂNG & OPTIMIZER
    optimizer = configure_optimizer(
        model, freeze_mode=mode, backbone_lr=backbone_lr, head_lr=head_lr,
        weight_decay=weight_decay, num_unfrozen_blocks=num_unfrozen_blocks
    )

    # 3. ĐỘ PHỨC TẠP (đo sau khi đóng băng để trainable_params đúng)
    complexity_info = ComplexityAnalyzer.measure_safely(
        model, input_size=(1, in_channels, seq_len), device=target_device
    )
    ComplexityAnalyzer.print_report(complexity_info)

    # 4. VÒNG LẶP HUẤN LUYỆN
    criterion = nn.CrossEntropyLoss()
    scheduler = _build_scheduler(optimizer, has_val)

    best_val_f1 = -1.0
    best_model_state = None
    training_history = []
    start_total_train_time = time.time()

    for epoch in range(1, epochs + 1):
        start_epoch_time = time.time()
        avg_train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, target_device)
        epoch_time_sec = time.time() - start_epoch_time

        row = {"epoch": epoch, "train_loss": round(avg_train_loss, 5)}
        val_metrics = None

        if has_val:
            val_metrics = compute_metrics(*predict(model, val_loader, target_device))
            scheduler.step(val_metrics["macro_f1"])
            row.update({
                "val_acc": round(val_metrics["accuracy"], 2),
                "val_f1": round(val_metrics["macro_f1"], 2),
                "val_precision": round(val_metrics["macro_precision"], 2),
                "val_recall": round(val_metrics["macro_recall"], 2),
            })
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_model_state = _snapshot_state(model)
        else:
            scheduler.step(avg_train_loss)

        row["epoch_time_sec"] = round(epoch_time_sec, 3)
        training_history.append(row)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            log_epoch(epoch, epochs, avg_train_loss, val_metrics if has_val else None, best_val_f1)

    total_train_time_sec = time.time() - start_total_train_time

    # 5. CHỌN TRỌNG SỐ CUỐI CÙNG
    #    Có Val -> nạp lại epoch tốt nhất. Không Val -> giữ epoch cuối (chụp 1 lần, không chụp mỗi epoch).
    if has_val and best_model_state is not None:
        model.load_state_dict({k: v.to(target_device) for k, v in best_model_state.items()})
    else:
        best_model_state = _snapshot_state(model)

    # 6. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP
    model.eval()
    inference_time_info = ComplexityAnalyzer.measure_total_inference_time(
        model, test_loader, device=target_device
    )
    print(f"⏱️ Đo thời gian chạy Test: {inference_time_info['total_inference_time_sec']} giây "
          f"({inference_time_info['avg_inference_ms_per_sample_full_test']} ms/mẫu)")

    test_targets, test_preds = predict(model, test_loader, target_device)
    test_metrics = compute_metrics(test_targets, test_preds)
    print_test_report(protocol_name, test_metrics)

    # 7. LƯU TRỮ KẾT QUẢ
    metadata = build_finetune_metadata(
        protocol_name, mode, num_unfrozen_blocks, encoder_ckpt_path, save_model_path,
        complexity_info, best_val_f1, test_metrics, total_train_time_sec, epochs,
        inference_time_info, training_history
    )
    ckpt_size = save_checkpoint_and_metadata(save_model_path, best_model_state, metadata)
    metadata["model_complexity"]["checkpoint_size_kb"] = ckpt_size

    return test_metrics["accuracy"], test_metrics["macro_f1"], model, metadata