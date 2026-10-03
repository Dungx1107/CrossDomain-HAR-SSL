"""
===============================================================================
ENGINE: HUẤN LUYỆN CLASSIFIER HEAD TRÊN MIỀN NGUỒN (SOURCE SUPERVISED TRAINING)
===============================================================================
Mục đích:
    - Nạp trọng số Encoder từ bước SSL Pretrain.
    - Gắn Classifier Head mới (5 lớp chung).
    - Huấn luyện trên 10% dữ liệu có nhãn của tập Train nguồn.
    - Dùng tập Val nguồn để Early Stopping và lưu file `best_source_model.pt`
      (Bao gồm toàn bộ Encoder + Head hoàn chỉnh để sẵn sàng mang đi Zero-Shot).
===============================================================================
"""

import sys
from pathlib import Path
from typing import Union, Dict, Any, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.har_classifier import HARClassifier
from models.encoders.builder import build_encoder
from models.heads.classifier import ClassifierHead


def train_source_classifier(
        train_loader: DataLoader,
        val_loader: DataLoader,
        encoder_checkpoint_path: Union[str, Path],
        save_model_path: Union[str, Path],
        backbone_type: str = "standard",
        num_classes: int = 5,
        in_channels: int = 6,
        feature_dim: int = 128,
        epochs: int = 40,
        backbone_lr: float = 1e-4,
        head_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[float, float, HARClassifier, Dict[str, Any]]:
    """
    Huấn luyện bộ phân loại trên nhãn nguồn và lưu lại toàn bộ model tốt nhất.
    """
    save_path = Path(save_model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path = Path(encoder_checkpoint_path)

    if not encoder_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy checkpoint SSL tại: {encoder_path}")

    # 1. KHỞI TẠO MÔ HÌNH TOÀN PHẦN (ENCODER + CLASSIFIER HEAD)
    encoder = build_encoder(backbone_type=backbone_type, in_channels=in_channels)
    classifier = ClassifierHead(feature_dim=feature_dim, num_classes=num_classes)

    model = HARClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        feature_dim=feature_dim,
        encoder=encoder,
        classifier=classifier
    ).to(device)

    # 2. NẠP TRỌNG SỐ PRETRAINED VÀO BACKBONE
    encoder_weights = torch.load(encoder_path, map_location=device, weights_only=True)
    # Xử lý trường hợp state_dict được bọc trong key
    if isinstance(encoder_weights, dict) and "encoder_state_dict" in encoder_weights:
        encoder_weights = encoder_weights["encoder_state_dict"]
    model.encoder.load_state_dict(encoder_weights, strict=True)
    print(f"📦 Đã nạp thành công trọng số SSL từ: {encoder_path.name}")

    # 3. THIẾT LẬP CHIẾN LƯỢC ĐÓNG BĂNG & OPTIMIZER
    if freeze_backbone:
        # Linear Probing: Khóa cứng Backbone, chỉ cập nhật Head
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer = Adam(model.classifier.parameters(), lr=head_lr, weight_decay=weight_decay)
        proto_str = "LINEAR PROBING (Freeze Backbone)"
    else:
        # Full Fine-Tuning: Mở khóa cả 2 với Layer-wise LR
        for p in model.encoder.parameters():
            p.requires_grad = True
        optimizer = Adam([
            {"params": model.encoder.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": weight_decay}
        ])
        proto_str = "FULL FINE-TUNING (Layer-wise LR)"

    print(f"⚙️ Giao thức: {proto_str} | Số lớp: {num_classes}")

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_weights = None
    history = []

    # 4. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, batches = 0.0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        avg_train_loss = train_loss / max(batches, 1)

        # Đánh giá trên tập Validation của nguồn
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(device)
                preds = torch.argmax(model(x_v), dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_v.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average="macro") * 100

        scheduler.step(val_f1)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        # Cập nhật checkpoint tốt nhất dựa trên Val Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc:5.2f}% | Val Macro F1: {val_f1:5.2f}% (Best: {best_val_f1:5.2f}%)")

    # 5. NẠP LẠI TRỌNG SỐ TỐT NHẤT VÀ LƯU RA ĐĨA
    if best_weights is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        torch.save(best_weights, save_path)
        print(f"💾 Checkpoint Full Model tối ưu nhất đã lưu tại: {save_path}")

    # Vẽ biểu đồ F1 / Loss trên nguồn
    plot_path = save_path.parent / "source_training_curve.png"
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], color="blue")
    plt.title("Source Train Loss")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot([h["epoch"] for h in history], [h["val_f1"] for h in history], color="green")
    plt.title("Source Val Macro F1 (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return best_val_f1, model, history