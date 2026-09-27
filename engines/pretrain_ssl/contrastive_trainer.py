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
import matplotlib.pyplot as plt
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
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        weight_decay: float = 3e-4,
        epochs: int = 40,
        temperature: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        measure_complexity: bool = True,
        seq_len: int = 128,
        num_workers: int = 2
) -> dict:
    save_dir = Path(save_dir)
    target_dir = save_dir / backbone_type
    target_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = target_dir / checkpoint_name
    csv_history_path = target_dir / "loss_history.csv"
    loss_plot_path = target_dir / "loss_curve.png"
    complexity_path = target_dir / "complexity.json"
    config_path = target_dir / "config.json"

    # 1. NẠP DỮ LIỆU
    dataset = ContrastiveDatasetWrapper(data_path)
    use_workers = num_workers if torch.cuda.is_available() else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(dataset) >= batch_size else False,
        num_workers=use_workers,  # Dùng đa luồng CPU để chuẩn bị dữ liệu song song
        pin_memory=(device == "cuda"),
        persistent_workers=(use_workers > 0)
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
    print(f"📁 Thư mục lưu: {target_dir} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 80)

    # 5. VÒNG LẶP HUẤN LUYỆN
    for epoch in range(1, epochs + 1):
        ssl_model.train()
        total_loss, total_tc, total_cc, total_gnorm = 0.0, 0.0, 0.0, 0.0

        for x_w, x_s in loader:
            x_w = x_w.to(device, non_blocking=True)
            x_s = x_s.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss, loss_tc, loss_cc = ssl_model(x_w, x_s)  # Nhận 3 giá trị trực tiếp từ TSTCCModel
            loss.backward()

            # Giữ an toàn gradient cho khối Attention Transformer
            gnorm = nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            total_tc += loss_tc.item()
            total_cc += loss_cc.item()
            total_gnorm += gnorm.item()

        avg_loss = total_loss / len(loader)
        avg_tc = total_tc / len(loader)
        avg_cc = total_cc / len(loader)
        avg_gnorm = total_gnorm / len(loader)

        history.append({
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_tc": avg_tc,
            "loss_cc": avg_cc,
            "grad_norm": avg_gnorm
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
        writer.writerow(["epoch", "loss_total", "loss_tc", "loss_cc", "grad_norm"])
        for row in history:
            writer.writerow([
                row["epoch"],
                f"{row['loss_total']:.6f}",
                f"{row['loss_tc']:.6f}",
                f"{row['loss_cc']:.6f}",
                f"{row['grad_norm']:.4f}"
            ])

    print(f"📊 Đã lưu lịch sử loss tại: {csv_history_path}")

    # 2. Lưu Plot
    dataset_name = Path(data_path).parent.name
    plot_loss_history(history, loss_plot_path, title=f"{backbone_type.upper()} on {dataset_name}")

    # 7. ĐO ĐỘ PHỨC TẠP
    complexity_info = None
    if measure_complexity:
        try:
            encoder_eval = build_encoder(backbone_type=backbone_type, in_channels=in_channels)
            state_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
            missing, unexpected = encoder_eval.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"⚠️ [LoadState Warning] Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")

            encoder_eval = encoder_eval.to(device)
            input_shape = (1, in_channels, seq_len)
            complexity_info = measure_model_complexity(
                encoder_eval,
                input_size=input_shape,
                device=torch.device(device)
            )
            print_complexity_report(complexity_info)


        except Exception as e:

            print(f"⚠️ Đo trên {device} thất bại ({type(e).__name__}: {e})")

            traceback.print_exc()

            print("🔄 Đang thử đo lại trên CPU...")

            try:
                encoder_cpu = build_encoder(backbone_type=backbone_type, in_channels=in_channels).to("cpu")
                state_dict_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                encoder_cpu.load_state_dict(state_dict_cpu, strict=False)
                complexity_info = measure_model_complexity(
                    encoder_cpu,
                    input_size=(1, in_channels, seq_len),
                    device=torch.device("cpu")
                )
                print_complexity_report(complexity_info)
            except Exception as e_cpu:
                print(f"❌ Không thể đo độ phức tạp trên CPU: {e_cpu}")
                complexity_info = {"error": f"CUDA error: {str(e)} | CPU error: {str(e_cpu)}"}

        if complexity_info:
            with open(complexity_path, "w", encoding="utf-8") as f:
                json.dump(complexity_info, f, indent=4)
            print(f"📊 Đã lưu độ phức tạp tại: {complexity_path}")

    # 8. TÓM TẮT KẾT QUẢ
    summary = {
        "best_loss": best_loss,
        "best_ckpt": str(checkpoint_path),
        "history_csv": str(csv_history_path),
        "loss_curve_plot": str(loss_plot_path),
        "config": config,
        "total_samples": len(dataset)
    }

    with open(save_dir / "pretrain_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def plot_loss_history(history: list, save_path: Path, title: str):
    """Vẽ 2 subplots: (1) Đường cong Loss thành phần và (2) Tỷ lệ cân bằng TC/CC."""
    epochs = [h["epoch"] for h in history]
    loss_total = [h["loss_total"] for h in history]
    loss_tc = [h["loss_tc"] for h in history]
    loss_cc = [h["loss_cc"] for h in history]
    tc_cc_ratio = [tc / (cc + 1e-8) for tc, cc in zip(loss_tc, loss_cc)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Subplot 1: Convergence
    ax1.plot(epochs, loss_total, label="Total Loss", color="#1f77b4", linewidth=2.0)
    ax1.plot(epochs, loss_tc, label="Temporal Contrasting (TC)", color="#ff7f0e", linestyle="--")
    ax1.plot(epochs, loss_cc, label="Contextual Contrasting (CC)", color="#2ca02c", linestyle=":")
    ax1.set_title(f"Pretraining Convergence: {title}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss Value", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Subplot 2: Loss Dynamics Ratio (TC / CC)
    ax2.plot(epochs, tc_cc_ratio, label="Ratio (TC / CC)", color="#9467bd", linewidth=1.5)
    ax2.axhline(y=1.0, color="gray", linestyle="-.", alpha=0.6)
    ax2.set_title("Balance Dynamics: TC / CC Ratio", fontsize=11)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Ratio", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"📈 Đã lưu biểu đồ phân tích loss tại: {save_path}")
