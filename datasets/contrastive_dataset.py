"""
===============================================================================
MODULE: CONTRASTIVE DATASET WRAPPER (CHO HỌC TỰ GIÁM SÁT TƯƠNG PHẢN)
===============================================================================
VAI TRÒ:
    - Bọc dữ liệu chuỗi thời gian thô từ file .pt hoặc Tensor.
    - Loại bỏ nhãn hoạt động (Unlabeled).
    - Tự động sinh cặp góc nhìn tăng cường (x_weak, x_strong) phục vụ
      các mô hình Siamese Contrastive Learning (TS-TCC, SimCLR...).
===============================================================================
"""

from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset

from models.ssl.contrastive.augmentations import TS_TCC_Augmentation


class ContrastiveDatasetWrapper(Dataset):
    """
    Dataset Wrapper sinh 2 views (Weak, Strong) phục vụ Contrastive Learning.
    """
    def __init__(self, data_source: Union[str, Path, torch.Tensor], aug_pipeline=None):
        self.aug_pipeline = aug_pipeline if aug_pipeline is not None else TS_TCC_Augmentation()

        if isinstance(data_source, (str, Path)):
            file_path = Path(data_source)
            if not file_path.exists():
                raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {file_path}")
            loaded_data = torch.load(file_path, map_location="cpu", weights_only=True)
            self.samples = loaded_data["samples"] if isinstance(loaded_data, dict) else loaded_data
        elif isinstance(data_source, torch.Tensor):
            self.samples = data_source
        else:
            raise TypeError("❌ data_source phải là đường dẫn file .pt hoặc torch.Tensor")

        if not isinstance(self.samples, torch.Tensor):
            self.samples = torch.tensor(self.samples, dtype=torch.float32)
        else:
            self.samples = self.samples.float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x_raw = self.samples[idx]
        return self.aug_pipeline(x_raw)