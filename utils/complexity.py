"""
===============================================================================
MODULE: ĐO LƯỜNG ĐỘ PHỨC TẠP VÀ HIỆU NĂNG MÔ HÌNH (COMPLEXITY ANALYZER)
===============================================================================
VAI TRÒ TRONG HỆ THỐNG:
    - Cung cấp các công cụ đo lường phần cứng (Hardware Profiling) cho cả quá trình
      Pretrain (đo Encoder) và Fine-tune (đo mô hình phân loại hoàn chỉnh).
    - Đo các chỉ số: Parameters, Model Size, MACs/FLOPs, Latency (độ trễ 1 mẫu)
      và Total Inference Time (tổng thời gian chạy trên tập Test).

CƠ CHẾ HOẠT ĐỘNG:
    1. Đếm tham số: Duyệt qua các lớp (layers) của mô hình (Params, Trainable).
    2. Ước lượng dung lượng: Params * 4 bytes (float32).
    3. MACs/FLOPs: Dùng thư viện `thop` chạy 1 lượt forward với input giả.
    4. Latency: Chạy mô hình 100 lần (có warmup, có đồng bộ CUDA) để tính thời gian trung bình 1 mẫu.
    5. Total Inference Time: Đo thời gian thực tế khi duyệt qua toàn bộ DataLoader.

CÁCH SỬ DỤNG:
    Tất cả các phương thức đều là `@staticmethod`, gọi trực tiếp qua Class.
    Ví dụ: ComplexityAnalyzer.measure_model(model, (1, 6, 128), device)
===============================================================================
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


class ComplexityAnalyzer:
    """Class tiện ích (Utility Class) đo lường độ phức tạp của mô hình PyTorch."""

    @staticmethod
    def measure_model(
            model: nn.Module,
            input_size=(1, 6, 128),
            device=None,
            num_warmup=20,
            num_runs=100
    ) -> dict:
        """
        Đo độ phức tạp tĩnh (Params, MACs, Latency 1 mẫu) của một mô hình.
        """
        if device is None:
            device = next(model.parameters()).device
        else:
            model = model.to(device)

        model.eval()

        # 1. Đếm số lượng tham số
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = (total_params * 4) / (1024 ** 2)

        # 2. Đo MACs / FLOPs (thop)
        dummy_input = torch.randn(*input_size, device=device)
        flops_str, macs_str = "N/A", "N/A"
        thop_error = None

        if HAS_THOP:
            try:
                macs_count, _ = profile(model, inputs=(dummy_input,), verbose=False)
                flops_count = macs_count * 2
                formatted = clever_format([macs_count, flops_count], "%.2f")
                macs_str, flops_str = formatted[0], formatted[1]
            except Exception as e:
                thop_error = f"{type(e).__name__}: {str(e)}"
                print(f"⚠️ [THOP WARNING] Không đo được MACs/FLOPs: {thop_error}")
        else:
            thop_error = "Thư viện 'thop' chưa được cài đặt"

        # 3. Đo Inference Latency (1 mẫu)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)

            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(num_runs):
                _ = model(dummy_input)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize()
            end_time = time.time()

        avg_latency_ms = ((end_time - start_time) / num_runs) * 1000

        return {
            "input_shape": list(input_size),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_size_mb": round(model_size_mb, 4),
            "macs": macs_str,
            "flops": flops_str,
            "latency_ms_per_sample": round(avg_latency_ms, 3),
            "device": str(device),
            "thop_status": "Success" if thop_error is None else thop_error
        }

    @staticmethod
    def print_report(info: dict):
        """In báo cáo độ phức tạp ra Terminal."""
        print("\n" + "=" * 75)
        print(f"{'BÁO CÁO ĐỘ PHỨC TẠP TÍNH TOÁN & HIỆU NĂNG MÔ HÌNH':^75}")
        print("=" * 75)
        print(f"📦 Kích thước đầu vào (1 sample) : {info.get('input_shape', 'N/A')}")
        print(f"⚙️ Tổng số tham số (Params)      : {info.get('total_params', 0):,}")
        print(f"🔥 Tham số huấn luyện (Trainable) : {info.get('trainable_params', 0):,}")
        print(f"💾 Dung lượng trọng số (Size)     : {info.get('model_size_mb', 0):.4f} MB")
        print(f"⚡ Khối lượng tính toán (MACs)   : {info.get('macs', 'N/A')}")
        print(f"🚀 Khối lượng tính toán (FLOPs)  : {info.get('flops', 'N/A')}")
        print(f"⏱️ Độ trễ suy luận (Latency)     : {info.get('latency_ms_per_sample', -1)} ms / sample (trên {info.get('device', 'Unknown')})")
        if info.get("thop_status") != "Success":
            print(f"⚠️ Ghi chú FLOPs/MACs           : {info.get('thop_status')}")
        print("=" * 75 + "\n")

    @staticmethod
    def measure_from_checkpoint(
            checkpoint_path,
            backbone_type: str,
            in_channels: int,
            input_size=(1, 6, 128),
            device="cuda",
            build_fn=None,
    ) -> dict:
        """Đo complexity của Encoder từ checkpoint (Dùng cho Pretrain)."""
        def _try_measure(dev):
            encoder = build_fn(backbone_type=backbone_type, in_channels=in_channels)
            state = torch.load(checkpoint_path, map_location=torch.device(dev), weights_only=True)
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"⚠️ [LoadState] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            encoder = encoder.to(dev)
            info = ComplexityAnalyzer.measure_model(encoder, input_size=input_size, device=torch.device(dev))
            ComplexityAnalyzer.print_report(info)
            return info

        try:
            return _try_measure(device)
        except Exception as e:
            print(f"⚠️ Đo trên {device} thất bại ({type(e).__name__}: {e})")
            try:
                return _try_measure("cpu")
            except Exception as e_cpu:
                print(f"❌ Không thể đo complexity trên CPU: {e_cpu}")
                return {"error": f"{device}: {e} | cpu: {e_cpu}"}

    @staticmethod
    def measure_safely(model: nn.Module, input_size=(1, 6, 128), device="cuda") -> dict:
        """Hàm wrapper (dùng cho Fine-tune): Tự động bắt lỗi, trả về giá trị mặc định an toàn."""
        try:
            return ComplexityAnalyzer.measure_model(model, input_size=input_size, device=device)
        except Exception as e:
            print(f"⚠️ Lỗi đo complexity: {e}")
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_size_mb": round((total_params * 4) / (1024 ** 2), 4),
                "macs": "N/A", "flops": "N/A",
                "latency_ms_per_sample": -1.0,
                "device": str(device), "thop_status": "Error"
            }

    @staticmethod
    def measure_total_inference_time(
            model: nn.Module,
            test_loader: DataLoader,
            device: str = "cuda"
    ) -> dict:
        """
        Đo tổng thời gian thực tế để mô hình dự đoán toàn bộ tập Test.
        Rất hữu ích để báo cáo chi phí thực tế khi triển khai.
        """
        model.eval()
        model.to(device)

        # Warmup (chạy thử 1 batch nhỏ để GPU/CPU làm nóng)
        try:
            first_batch = next(iter(test_loader))[0].to(device)
            with torch.no_grad():
                _ = model(first_batch)
        except StopIteration:
            pass

        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()
        total_samples = 0

        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                _ = model(x_batch)
                total_samples += x_batch.size(0)

        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()

        total_time_sec = time.time() - start_time
        # Tính trung bình ms/mẫu dựa trên toàn bộ tập test
        avg_ms_per_sample = (total_time_sec / total_samples) * 1000 if total_samples > 0 else 0

        return {
            "total_test_samples": total_samples,
            "total_inference_time_sec": round(total_time_sec, 4),
            "avg_inference_ms_per_sample_full_test": round(avg_ms_per_sample, 4),
            "batch_count": len(test_loader),
            "device": str(device)
        }