"""
Engine huấn luyện chính cho phương pháp Self-Supervised Learning (SSL) Contrastive Learning.
    Mục đích: Huấn luyện encoder (bộ mã hóa) để học biểu diễn đặc trưng từ dữ liệu cảm biến 1D (gia tốc kế, con quay hồi chuyển) mà không cần nhãn
    Kỹ thuật: TS-TCC (Time-Series Transformer with Contrastive Coding) - một phương pháp contrastive learning cho dữ liệu chuỗi thời gian
    Loss function: NT-Xent (Normalized Temperature-scaled Cross Entropy) - loss phổ biến trong SimCLR
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
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.ssl.contrastive.ts_tcc_model import TSTCCModel
from losses.nt_xent import NTXentLoss
from utils.hardware import print_gpu_status
from utils.complexity import measure_model_complexity, print_complexity_report


def train_contrastive_encoder(
        data_path: Union[str, Path],  # Đường dẫn đến file dữ liệu đã xử lý (.pt)
        save_dir: Union[str, Path],  # Thư mục lưu kết quả huấn luyện
        checkpoint_name: str = "best_encoder.pt",  # Đường dẫn đầy đủ đến file checkpoint sẽ lưu.
        model: Optional[nn.Module] = None,
        in_channels: int = 6,  # Số kênh đầu vào (6 cho gia tốc 3 trục + quay 3 trục)
        feature_dim: int = 128,  # Kích thước vector đặc trưng từ encoder
        projection_dim: int = 64,  # Kích thước vector sau projection head
        batch_size: int = 64,  # Số mẫu trong 1 batch
        learning_rate: float = 1e-3,  # Tốc độ học
        weight_decay: float = 1e-4,  # Hệ số điều chỉnh (regularization)
        epochs: int = 40,  # Số epoch huấn luyện
        temperature: float = 0.2,  # Nhiệt độ trong NT-Xent loss (kiểm soát độ phân biệt)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        measure_complexity: bool = True,
        seq_len: int = 128
) -> dict:
    """
    Engine huấn luyện Contrastive SSL.
    """
    # Tạo cấu trúc thư mục lưu trữ kết quả
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Tất cả file đều lưu trực tiếp trong save_dir
    checkpoint_path = save_dir / checkpoint_name
    csv_history_path = save_dir / "loss_history.csv"
    complexity_path = save_dir / "complexity.json"
    config_path = save_dir / "config.json"

    # 1. NẠP DỮ LIỆU
    dataset = ContrastiveDatasetWrapper(data_path)  # Wrapper xử lý dữ liệu contrastive
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Xáo trộn dữ liệu để tránh overfitting
        drop_last=True if len(dataset) >= batch_size else False  # Bỏ batch cuối nếu không đủ
    )

    # 2. KHỞI TẠO MODEL
    if model is None:
        # encoder_backbone: CNN 1D để trích xuất đặc trưng từ chuỗi thời gian
        encoder_backbone = StandardSensorEncoder1D(
            in_channels=in_channels,
            feature_dim=feature_dim
        )
        # ssl_model: Model chính với projection head (MLP) ánh xạ sang không gian nhúng
        ssl_model = TSTCCModel(
            encoder=encoder_backbone,
            feature_dim=feature_dim,
            projection_dim=projection_dim
        ).to(device)
    else:
        ssl_model = model.to(device)

    # 4. LƯU CẤU HÌNH
    config = {
        "data_path": str(data_path),
        "save_dir": str(save_dir),
        "checkpoint_name": checkpoint_name,
        "in_channels": in_channels,
        "feature_dim": feature_dim,
        "projection_dim": projection_dim,
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

    # 5. HUẤN LUYỆN
    # Khởi tạo loss function và optimizer
    criterion = NTXentLoss(temperature=temperature).to(device)  # Loss contrastive chính
    optimizer = AdamW(  # AdamW: phiên bản cải tiến của Adam, giúp regularization tốt hơn
        ssl_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    best_loss = float("inf")  # Khởi tạo loss tốt nhất ở mức vô cùng lớn
    history = []  # Lưu lịch sử loss theo từng epoch

    print("=" * 80)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN SSL (TS-TCC) | Nguồn: {data_path}")
    print(f"📁 Thư mục lưu: {save_dir} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 80)

    # 3. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(1, epochs + 1):
        ssl_model.train()  # Chuyển sang chế độ train (bật dropout, batch norm...)
        total_loss = 0.0

        # Lặp qua từng batch
        for x_w, x_s in loader:  # x_w: view "weak" (tăng cường ít), x_s: view "strong" (tăng cường nhiều)
            x_w, x_s = x_w.to(device), x_s.to(device)

            optimizer.zero_grad()  # Xóa gradient từ batch trước

            # Forward pass: tính vector đặc trưng của 2 views
            h_w, h_s = ssl_model(x_w, x_s)  # h_w, h_s: projections sau MLP

            # Tính NT-Xent loss: so sánh độ tương đồng giữa các cặp cùng class
            loss = criterion(h_w, h_s)

            # Backward pass: tính gradient
            loss.backward()

            # Cập nhật trọng số model
            optimizer.step()

            total_loss += loss.item()

        # Tính loss trung bình của epoch
        avg_loss = total_loss / len(loader)
        history.append({"epoch": epoch, "loss": avg_loss})

        # CHIẾN LƯỢC LƯU MODEL TỐT NHẤT
        # Lưu encoder (không bao gồm projection head) vì encoder là thành phần chính
        # để sử dụng cho downstream tasks sau
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ssl_model.encoder.state_dict(), checkpoint_path)

        # In log mỗi 5 epoch + epoch đầu/cuối
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | 📉 NT-Xent Loss: {avg_loss:.5f} (Best: {best_loss:.5f})")
            print_gpu_status()  # In trạng thái GPU: load %, VRAM, nhiệt độ

    # 6. GHI LOG LỊCH SỬ
    with open(csv_history_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for row in history:
            writer.writerow([row["epoch"], f"{row['loss']:.6f}"])

    print(f"📊 Đã lưu lịch sử loss tại: {csv_history_path}")
    print(f"💾 Checkpoint tốt nhất lưu tại: {checkpoint_path}")

    # ✅ 6. ĐO ĐỘ PHỨC TẠP SAU KHI TRAIN XONG
    # ============================================================
    complexity_info = None
    if measure_complexity:
        try:
            # Tạo model hoàn chỉnh từ encoder đã train
            encoder_backbone = StandardSensorEncoder1D(in_channels=in_channels, feature_dim=feature_dim)
            encoder_backbone.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

            # Đo độ phức tạp của encoder (không bao gồm projection head)
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

    # 7. TÓM TẮT KẾT QUẢ
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
