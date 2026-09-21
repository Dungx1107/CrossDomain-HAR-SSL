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
from utils.complexity import measure_model_complexity, print_complexity_report
from models.encoders.builder import build_encoder


def train_contrastive_encoder(
        data_path: Union[str, Path],
        save_dir: Union[str, Path],
        checkpoint_name: str = "best_encoder.pt",
        backbone_type: str = "tstcc",
        model: Optional[nn.Module] = None,
        in_channels: int = 6,
        feature_dim: int = 128,
        projection_dim: int = 64,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        epochs: int = 40,
        temperature: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        measure_complexity: bool = True,
        seq_len: int = 128
) -> dict:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / backbone_type / checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    csv_history_path = save_dir / "loss_history.csv"
    complexity_path = save_dir / "complexity.json"
    config_path = save_dir / "config.json"

    # 1. NẠP DỮ LIỆU
    dataset = ContrastiveDatasetWrapper(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(dataset) >= batch_size else False,
        num_workers=4,  # Dùng 4 luồng CPU để chuẩn bị dữ liệu song song
        pin_memory=(device == "cuda"),
        persistent_workers=True  # Giữ luồng CPU hoạt động liên tục giữa các epoch
    )

    # 2. KHỞI TẠO MODEL
    if model is None:
        encoder_backbone = build_encoder(
            backbone_type=backbone_type,
            in_channels=in_channels
        )
        ssl_model = TSTCCModel(
            encoder=encoder_backbone,
            in_channels=in_channels,
            feature_dim=feature_dim,
            timesteps=3,
            lambda1=1.0,
            lambda2=0.7,
            temperature=temperature
        ).to(device)
    else:
        ssl_model = model.to(device)

    # 3. LƯU CẤU HÌNH
    config = {
        "data_path": str(data_path),
        "save_dir": str(save_dir),
        "checkpoint_name": checkpoint_name,
        "backbone_type": backbone_type,
        "in_channels": in_channels,
        "feature_dim": feature_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "temperature": temperature,
        "device": device,
        "total_samples": len(dataset)
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"📝 Đã lưu cấu hình tại: {config_path}")

    # 4. KHỞI TẠO OPTIMIZER
    optimizer = AdamW(
        ssl_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_loss = float("inf")
    history = []

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN SSL (TS-TCC) | Nguồn: {data_path}")
    print(f"📁 Thư mục lưu: {save_dir} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 80)

    # 5. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss = 0.0
        total_tc = 0.0
        total_cc = 0.0

        for x_w, x_s in loader:
            x_w = x_w.to(device, non_blocking=True)
            x_s = x_s.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Nhận 3 giá trị trực tiếp từ TSTCCModel
            loss, loss_tc, loss_cc = ssl_model(x_w, x_s)

            loss.backward()

            # Giữ an toàn gradient cho khối Attention Transformer
            nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=2.0)

            optimizer.step()

            total_loss += loss.item()
            total_tc += loss_tc.item()
            total_cc += loss_cc.item()

        avg_loss = total_loss / len(loader)
        avg_tc = total_tc / len(loader)
        avg_cc = total_cc / len(loader)

        history.append({
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_tc": avg_tc,
            "loss_cc": avg_cc
        })

        # Lưu trọng số encoder của mô hình có loss thấp nhất
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

    # 6. GHI LOG LỊCH SỬ
    with open(csv_history_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "loss_tc", "loss_cc"])
        for row in history:
            writer.writerow([
                row["epoch"],
                f"{row['loss_total']:.6f}",
                f"{row['loss_tc']:.6f}",
                f"{row['loss_cc']:.6f}"
            ])

    print(f"📊 Đã lưu lịch sử loss tại: {csv_history_path}")
    print(f"💾 Checkpoint tốt nhất lưu tại: {checkpoint_path}")

    # 7. ĐO ĐỘ PHỨC TẠP
    complexity_info = None
    if measure_complexity:
        try:
            encoder_backbone = build_encoder(backbone_type=backbone_type, in_channels=in_channels)
            encoder_backbone.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True))

            input_shape = (1, in_channels, seq_len)
            complexity_info = measure_model_complexity(
                encoder_backbone,
                input_size=input_shape,
                device=torch.device(device)
            )
            print_complexity_report(complexity_info)

            with open(complexity_path, "w", encoding="utf-8") as f:
                json.dump(complexity_info, f, indent=4)
            print(f"📊 Đã lưu độ phức tạp tại: {complexity_path}")

        except Exception as e:
            print(f"⚠️ Không thể đo độ phức tạp: {e}")
            complexity_info = {"error": str(e)}

    # 8. TÓM TẮT KẾT QUẢ
    summary = {
        "best_loss": best_loss,
        "best_ckpt": str(checkpoint_path),
        "history_csv": str(csv_history_path),
        "config": config,
        "total_samples": len(dataset)
    }

    with open(save_dir / "pretrain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary
