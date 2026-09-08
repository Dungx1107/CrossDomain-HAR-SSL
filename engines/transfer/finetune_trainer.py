"""
===============================================================================
MODULE: FINE-TUNE & ADAPTATION ENGINE (CROSS-DOMAIN & FEW-LABEL HAR)
===============================================================================
VAI TRÒ TRONG HỆ THỐNG:
    - Động cơ cốt lõi thực hiện quá trình tinh chỉnh (Fine-tuning) hoặc thăm dò
      tuyến tính (Linear Probing) mô hình đã học tự giám sát (SSL Pre-trained).
    - Hỗ trợ chuyển giao mô hình từ miền nguồn (Source Domain) sang miền đích
      (Target Domain) với lượng dữ liệu có nhãn hạn chế (Few-label Regime).
    - Tự động đo lường độ phức tạp tính toán (FLOPs, Trainable Params, Checkpoint Size)
      và lưu lại checkpoint tối ưu nhất cho bài toán kiểm định.

NGUYÊN LÝ HOẠT ĐỘNG:
    1. Khởi tạo mô hình tổng hợp HARClassifier (Backbone Encoder + Classifier Head).
    2. Nạp trọng số biểu diễn đã pre-train (SSL weights) vào Backbone Encoder.
    3. Tùy biến chiến lược tối ưu hóa theo tham số `freeze_backbone`:
       - Linear Probing (`freeze_backbone=True`): Đóng băng hoàn toàn Backbone,
         chỉ cập nhật duy nhất Classifier Head với tốc độ học lớn.
       - Full Fine-Tuning (`freeze_backbone=False`): Mở khóa toàn bộ mạng nhưng
         sử dụng Layer-wise Learning Rate (Backbone học chậm, Head học nhanh)
         để bảo toàn tri thức SSL và tránh hiện tượng quên thảm khốc (Catastrophic Forgetting).
    4. Huấn luyện, kiểm định qua từng Epoch bằng Macro F1, lưu trữ trạng thái tốt nhất
       và trích xuất báo cáo đo lường phần cứng khoa học.
===============================================================================
"""

import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.har_classifier import HARClassifier
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from utils.complexity import measure_model_complexity


def train_and_eval_finetune(
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        freeze_backbone: bool = False,
        save_model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
) -> Tuple[float, float, HARClassifier, Dict[str, Any]]:
    """
    Thực thi quá trình huấn luyện chuyển giao (Transfer Learning) và đánh giá mô hình.
    """
    target_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    encoder_ckpt_path = Path(encoder_checkpoint_path)

    # 1. KIỂM TRA ĐƯỜNG DẪN CHECKPOINT SSL NGUỒN
    if not encoder_ckpt_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số SSL tại: {encoder_ckpt_path}")

    # 2. KHỞI TẠO MÔ HÌNH TOÀN PHẦN (BACKBONE + CLASSIFIER HEAD)
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
    ).to(target_device)

    # 3. NẠP TRỌNG SỐ PRETRAINED VÀO BACKBONE
    checkpoint = torch.load(encoder_ckpt_path, map_location=target_device, weights_only=True)
    encoder_dict = checkpoint["encoder"] if isinstance(checkpoint, dict) and "encoder" in checkpoint else checkpoint

    model.encoder.load_state_dict(encoder_dict, strict=True)
    protocol_name = "LINEAR PROBING" if freeze_backbone else "FULL FINE-TUNING"

    print("=" * 80)
    print(f"🚀 THIẾT LẬP THỰC NGHIỆM: {protocol_name}")
    print(f"📦 Checkpoint SSL nguồn: {encoder_ckpt_path.name}")
    print(f"🎯 Số lớp đích (Target Classes): {num_classes} | Kênh vào (Channels): {in_channels}")

    # 4. CẤU HÌNH ĐÓNG BĂNG & THIẾT LẬP BỘ TỐI ƯU (OPTIMIZER)
    if freeze_backbone:
        # KỊCH BẢN 1: LINEAR PROBING (Khóa gradient của Backbone)
        for param in model.encoder.parameters():
            param.requires_grad = False

        optimizer = Adam(
            model.classifier.parameters(),
            lr=head_lr,
            weight_decay=weight_decay
        )
        print("🔒 Trạng thái Backbone: ĐÃ ĐÓNG BĂNG (Chỉ cập nhật Classifier Head)")
    else:
        # KỊCH BẢN 2: FULL FINE-TUNING (Mở toàn bộ mạng với Layer-wise Learning Rate)
        for param in model.encoder.parameters():
            param.requires_grad = True

        optimizer = Adam([
            {"params": model.encoder.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ])
        print(f"🔓 Trạng thái Backbone: MỞ KHÓA (Layer-wise LR: Backbone={backbone_lr:.1e}, Head={head_lr:.1e})")

    # 5. ĐO LƯỜNG ĐỘ PHỨC TẠP TÍNH TOÁN
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs_val = "N/A"
    flops_val = "N/A"
    model_size_mb = round((total_params * 4) / (1024 ** 2), 4)

    try:
        input_shape = (1, in_channels, seq_len)
        complexity_info = measure_model_complexity(
            model,
            input_size=input_shape,
            device=torch.device(target_device)
        )
        total_params = complexity_info.get("total_params", total_params)[cite: 7]
        trainable_params = complexity_info.get("trainable_params", trainable_params)[cite: 7]
        macs_val = complexity_info.get("macs", "N/A")[cite: 7]
        flops_val = complexity_info.get("flops", "N/A")[cite: 7]
        model_size_mb = complexity_info.get("model_size_mb", model_size_mb)[cite: 7]
    except Exception:
        pass

    print(f"📊 Độ phức tạp mô hình:")
    print(f"   - Tổng số tham số (Total Params): {total_params:,}")
    print(f"   - Tham số cập nhật (Trainable)  : {trainable_params:,} ({(trainable_params / max(total_params, 1)) * 100:.2f}%)")
    print(f"   - Khối lượng tính toán          : {macs_val} MACs | {flops_val} FLOPs")
    print(f"   - Dung lượng RAM/VRAM           : {model_size_mb} MB")
    print("=" * 80)

    # 6. VÒNG LẶP HUẤN LUYỆN & THEO DÕI HỘI TỤ
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_model_state = None
    training_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(target_device)
            y_batch = y_batch.to(target_device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(num_batches, 1)

        # Đánh giá trên tập Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(target_device)
                preds = torch.argmax(model(x_val), dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average="macro") * 100

        scheduler.step(val_f1)

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_acc": round(val_acc, 2),
            "val_f1": round(val_f1, 2)
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc:5.2f}% | Val Macro F1: {val_f1:5.2f}% (Best: {best_val_f1:5.2f}%)")

    # 7. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP BẰNG TRỌNG SỐ TỐI ƯU NHẤT
    if best_model_state is not None:
        model.load_state_dict({k: v.to(target_device) for k, v in best_model_state.items()})

    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(target_device)
            preds = torch.argmax(model(x_test), dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y_test.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds, average="macro") * 100

    print("-" * 80)
    print(f"🎯 KẾT QUẢ ĐÁNH GIÁ TẬP TEST ({protocol_name}):")
    print(f"   - Test Accuracy : {test_acc:6.2f}%")
    print(f"   - Test Macro F1 : {test_f1:6.2f}% (Chỉ số nghiên cứu chính)")
    print("-" * 80)

    # 8. LƯU MÔ HÌNH VÀ SIÊU DỮ LIỆU METADATA RA ĐĨA
    checkpoint_file_size_kb = 0.0
    if save_model_path is not None and best_model_state is not None:
        save_path = Path(save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(best_model_state, save_path)
        checkpoint_file_size_kb = round(save_path.stat().st_size / 1024, 2)
        print(f"💾 Đã lưu Checkpoint mô hình tối ưu tại: {save_path} ({checkpoint_file_size_kb:.1f} KB)")

    metadata = {
        "protocol": protocol_name,
        "freeze_backbone": freeze_backbone,
        "source_checkpoint": str(encoder_ckpt_path),
        "saved_model_path": str(save_model_path) if save_model_path else None,
        "model_complexity": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percent": round((trainable_params / max(total_params, 1)) * 100, 2),
            "macs": macs_val,
            "flops": flops_val,
            "checkpoint_size_kb": checkpoint_file_size_kb
        },
        "performance": {
            "best_val_f1": round(best_val_f1, 2),
            "test_accuracy": round(test_acc, 2),
            "test_macro_f1": round(test_f1, 2)
        },
        "history": training_history
    }

    return test_acc, test_f1, model, metadata