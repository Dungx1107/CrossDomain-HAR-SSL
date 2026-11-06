"""
===============================================================================
MODULE: SSL CHECKPOINT PATHS (NGUỒN DUY NHẤT CHO ĐƯỜNG DẪN CHECKPOINT)
===============================================================================
Quy ước (khớp với run_few_shot.py):
    <output_root>/checkpoints/ssl_pretrain/<method>/<dataset>/<backbone>/
        <method>_<backbone>_encoder_pretrained_<dataset>.pt

    method  : tstcc | swav | crosshar | tstcc_swav ...
    dataset : key trong config/datasets/registry.py (vd "motion_sense")
===============================================================================
"""

import os
from pathlib import Path
from typing import Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_output_root() -> Path:
    """Kaggle -> /kaggle/working; còn lại -> thư mục gốc dự án."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return Path("/kaggle/working")
    return PROJECT_ROOT


def ssl_checkpoint_name(
        method: str,
        backbone: str,
        dataset: str,
) -> str:
    return f"{method}_{backbone}_encoder_pretrained_{dataset}.pt"


def ssl_checkpoint_dir(
    output_root: Union[str, Path],
    method: str,
    dataset: str,
    backbone: str,
    tag: str = "",
) -> Path:
    """
    Thư mục checkpoint. Tag rỗng -> giữ đường dẫn cũ (backward-compatible).
    Tag không rỗng -> backbone_<tag> (vd 'standard_bs128').
    """
    suffix = f"_{tag}" if tag else ""
    return Path(output_root) / "checkpoints" / "ssl_pretrain" / method / dataset / f"{backbone}{suffix}"

def ssl_checkpoint_path(
    output_root: Union[str, Path],
    method: str,
    dataset: str,
    backbone: str,
    tag: str = "",
) -> Path:
    return ssl_checkpoint_dir(output_root, method, dataset, backbone, tag=tag) / \
           ssl_checkpoint_name(method, backbone, dataset)