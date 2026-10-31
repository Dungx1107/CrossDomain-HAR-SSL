"""
===============================================================================
MODULE: TRANSFER MODEL BUILDER
===============================================================================
Lắp ráp mô hình cho bài toán Transfer (Backbone + Classifier Head), nạp trọng
số SSL pretrain và cấu hình Optimizer theo chế độ đóng băng (freeze_mode):

    - "none"    : Full fine-tuning  (mở khóa toàn bộ encoder)
    - "full"    : Linear probing    (đóng băng toàn bộ encoder)
    - "partial" : Partial fine-tuning (chỉ mở khóa N khối cuối của encoder)
===============================================================================
"""

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.har_classifier import HARClassifier
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead

FREEZE_NONE = "none"
FREEZE_FULL = "full"
FREEZE_PARTIAL = "partial"
VALID_FREEZE_MODES = (FREEZE_NONE, FREEZE_FULL, FREEZE_PARTIAL)


def resolve_freeze_mode(freeze_mode: Optional[str], freeze_backbone: bool = False) -> str:
    """
    Chuẩn hóa chế độ đóng băng. Tương thích ngược với cờ `freeze_backbone`:
    nếu không truyền freeze_mode thì freeze_backbone=True -> "full", False -> "none".
    """
    if freeze_mode is None:
        return FREEZE_FULL if freeze_backbone else FREEZE_NONE
    if freeze_mode not in VALID_FREEZE_MODES:
        raise ValueError(f"freeze_mode phải thuộc {VALID_FREEZE_MODES}, nhận được: '{freeze_mode}'")
    return freeze_mode


def _load_encoder_weights(
        model: HARClassifier,
        checkpoint_path: Path,
        device: str
) -> None:
    """Đọc checkpoint SSL và nạp vào model.encoder (hỗ trợ nhiều định dạng key)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict):
        if "encoder_state_dict" in checkpoint:
            encoder_dict = checkpoint["encoder_state_dict"]
        elif "encoder" in checkpoint:
            encoder_dict = checkpoint["encoder"]
        else:
            encoder_dict = checkpoint
    else:
        encoder_dict = checkpoint

    model.encoder.load_state_dict(encoder_dict, strict=True)


def build_transfer_model(
        encoder_checkpoint_path: Union[str, Path],
        encoder: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
        num_classes: int = 5,
        in_channels: int = 6,
        feature_dim: int = 128,
        device: str = "cpu"
) -> HARClassifier:
    """
    Khởi tạo HARClassifier (Encoder + Classifier Head) và nạp trọng số pretrain
    vào Encoder. Classifier Head luôn được khởi tạo mới (ngẫu nhiên).
    """
    ckpt_path = Path(encoder_checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số SSL tại: {ckpt_path}")

    used_encoder = encoder if encoder is not None else StandardSensorEncoder1D(
        in_channels=in_channels,
        feature_dim=feature_dim
    )
    used_classifier = classifier if classifier is not None else ClassifierHead(
        feature_dim=feature_dim,
        num_classes=num_classes
    )

    model = HARClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        encoder=used_encoder,
        classifier=used_classifier
    ).to(device)

    _load_encoder_weights(model, ckpt_path, device)
    return model


def _set_encoder_trainable(
        encoder: nn.Module,
        freeze_mode: str,
        num_unfrozen_blocks: int
) -> None:
    """Đặt requires_grad cho các tham số của encoder theo freeze_mode."""
    if freeze_mode == FREEZE_NONE:
        for p in encoder.parameters():
            p.requires_grad = True

    elif freeze_mode == FREEZE_FULL:
        for p in encoder.parameters():
            p.requires_grad = False

    else:  # FREEZE_PARTIAL: đóng băng hết, rồi mở N khối con cuối cùng
        blocks = [m for _, m in encoder.named_children() if any(True for _ in m.parameters())]
        if num_unfrozen_blocks < 1 or num_unfrozen_blocks > len(blocks):
            raise ValueError(
                f"num_unfrozen_blocks={num_unfrozen_blocks} không hợp lệ: "
                f"encoder có {len(blocks)} khối con có tham số."
            )
        for p in encoder.parameters():
            p.requires_grad = False
        for block in blocks[-num_unfrozen_blocks:]:
            for p in block.parameters():
                p.requires_grad = True


def configure_optimizer(
        model: HARClassifier,
        freeze_mode: str = FREEZE_NONE,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_unfrozen_blocks: int = 1
) -> Adam:
    """
    Áp dụng chế độ đóng băng lên encoder và trả về Optimizer Adam tương ứng.
    Classifier Head luôn được huấn luyện.
    """
    _set_encoder_trainable(model.encoder, freeze_mode, num_unfrozen_blocks)

    head_group = {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": weight_decay}

    if freeze_mode == FREEZE_FULL:
        print("🔒 Trạng thái Backbone: ĐÃ ĐÓNG BĂNG (Chỉ cập nhật Classifier Head)")
        return Adam([head_group])

    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    backbone_group = {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}

    if freeze_mode == FREEZE_PARTIAL:
        print(f"🔓 Trạng thái Backbone: MỞ KHÓA MỘT PHẦN ({num_unfrozen_blocks} khối cuối) "
              f"(LR: Backbone={backbone_lr:.1e}, Head={head_lr:.1e})")
    else:
        print(f"🔓 Trạng thái Backbone: MỞ KHÓA (Layer-wise LR: Backbone={backbone_lr:.1e}, Head={head_lr:.1e})")

    return Adam([backbone_group, head_group])