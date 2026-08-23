"""
===============================================================================
    Đo lường các chỉ số độ phức tạp tính toán và hiệu năng của mô hình:
      1. Tổng số tham số (Total Parameters) & Tham số huấn luyện (Trainable Params).
      2. Dung lượng mô hình trên bộ nhớ (Model Size in MB).
      3. Khối lượng tính toán (FLOPs / MACs) trên 1 cửa sổ tín hiệu (Window).
      4. Độ trễ suy luận trung bình (Inference Latency in ms/sample).
===============================================================================
"""

import time
import torch
import torch.nn as nn

try:
    from thop import profile, clever_format

    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def measure_model_complexity(
        model: nn.Module,
        input_size=(1, 9, 128),
        device=None,
        num_warmup=20,
        num_runs=100
):
    """
    Đo toàn diện độ phức tạp và độ trễ suy luận của mô hình.

    Args:
        model (nn.Module): Mô hình nơ-ron cần đo.
        input_size (tuple): Kích thước của 1 mẫu đơn lẻ (Batch=1, In_Channels, Window_Size).
        device (torch.device): CPU hoặc CUDA GPU.
        num_warmup (int): Số lượt chạy làm nóng bộ nhớ cache trước khi bấm giờ.
        num_runs (int): Số lượt đo để lấy trung bình thời gian suy luận.

    Returns:
        dict: Chứa toàn bộ các thông số đo đạc để lưu vào config.json và in báo cáo.
    """

    model.eval()

    # 1. Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Ước lượng dung lượng mô hình trên RAM/VRAM (theo định dạng float32 = 4 bytes)
    model_size_mb = (total_params * 4) / (1024 ** 2)

    # 3. Tính FLOPs / MACs bằng thop
    dummy_input = torch.randn(*input_size, device=device)
    flops_str, macs_str = "N/A", "N/A"
    flops_count, macs_count = 0, 0

    if HAS_THOP:
        try:
            macs_count, _ = profile(model, inputs=(dummy_input,), verbose=False)
            flops_count = macs_count * 2  # 1 MAC xấp xỉ 2 FLOPs
            macs_str, _ = clever_format([macs_count, total_params], "%.2f")
            flops_str, _ = clever_format([flops_count, total_params], "%.2f")
        except Exception:
            pass

    # 4. Đo độ trễ suy luận (Inference Latency - mili-giây / mẫu)
    with torch.no_grad():
        # Warm-up (Khởi động GPU/CPU)
        for _ in range(num_warmup):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

        end_time = time.time()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000  # Chuyển sang ms

    complexity_info = {
        "input_shape": list(input_size),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 4),
        "macs": macs_str,
        "flops": flops_str,
        "latency_ms_per_sample": round(avg_latency_ms, 3),
        "device": str(device)
    }

    return complexity_info


def print_complexity_report(info: dict):
    """In bảng độ phức tạp ra Terminal."""
    print("\n" + "=" * 75)
    print(f"{'BÁO CÁO ĐỘ PHỨC TẠP TÍNH TOÁN & HIỆU NĂNG MÔ HÌNH':^75}")
    print("=" * 75)
    print(f"📦 Kích thước đầu vào (1 sample) : {info['input_shape']}")
    print(f"⚙️ Tổng số tham số (Params)      : {info['total_params']:,}")
    print(f"🔥 Tham số huấn luyện (Trainable) : {info['trainable_params']:,}")
    print(f"💾 Dung lượng trọng số (Size)     : {info['model_size_mb']:.4f} MB")
    print(f"⚡ Khối lượng tính toán (MACs)   : {info['macs']}")
    print(f"🚀 Khối lượng tính toán (FLOPs)  : {info['flops']}")
    print(f"⏱️ Độ trễ suy luận (Latency)     : {info['latency_ms_per_sample']} ms / sample (trên {info['device']})")
    print("=" * 75 + "\n")
