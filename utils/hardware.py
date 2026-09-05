"""
Tiện ích giám sát phần cứng GPU: Đo VRAM, % Tải (Load), và Nhiệt độ khi huấn luyện.
"""
from typing import Optional, Dict
import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def get_gpu_status(device_index: int = 0) -> Optional[Dict[str, float]]:
    """
    Lấy thông số tải thực tế của GPU thông qua pynvml.
    Trả về Dict gồm: gpu_load (%), vram_used (MB), vram_total (MB), temp (C).
    """
    if not torch.cuda.is_available() or not PYNVML_AVAILABLE:
        return None

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        status = {
            "name": pynvml.nvmlDeviceGetName(handle),
            "gpu_load": float(util.gpu),
            "vram_used_mb": float(mem.used / (1024 ** 2)),
            "vram_total_mb": float(mem.total / (1024 ** 2)),
            "vram_percent": float((mem.used / mem.total) * 100),
            "temp_c": float(temp),
        }
        pynvml.nvmlShutdown()
        return status
    except Exception:
        return None


def print_gpu_status(device_index: int = 0) -> None:
    """In trực tiếp tình trạng GPU ra terminal."""
    status = get_gpu_status(device_index)
    if status is None:
        if torch.cuda.is_available():
            # Fallback nếu chưa cài pynvml: chỉ đọc VRAM qua PyTorch
            allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            print(f"⚡ [GPU] VRAM PyTorch đang dùng: {allocated:.1f} MB")
        return

    print(
        f"⚡ [{status['name']}] "
        f"Load: {status['gpu_load']:.0f}% | "
        f"VRAM: {status['vram_used_mb']:.1f}/{status['vram_total_mb']:.1f} MB ({status['vram_percent']:.1f}%) | "
        f"Nhiệt độ: {status['temp_c']:.0f}°C"
    )