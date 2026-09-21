"""
===============================================================================
MODULE: CONTRASTIVE DATASET WRAPPER (CHO HỌC TỰ GIÁM SÁT TƯƠNG PHẢN)
===============================================================================
VAI TRÒ:
    - Bọc dữ liệu chuỗi thời gian thô từ file .pt hoặc Tensor.
    - Loại bỏ nhãn hoạt động (Unlabeled) vì SSL không cần nhãn.
    - Tự động sinh cặp góc nhìn tăng cường (x_weak, x_strong) phục vụ
      các mô hình Siamese Contrastive Learning (TS-TCC, SimCLR...).

Ý TƯỞNG CHÍNH:
    Contrastive Learning cần ít nhất 2 "góc nhìn" (views) khác nhau của
    cùng một mẫu dữ liệu. Ví dụ:
        - Weak Augmentation: Jitter nhẹ (nhiễu ít)
        - Strong Augmentation: Jitter + Scaling + Permutation (nhiễu nhiều)

    Model sẽ học cách kéo 2 views này lại gần nhau trong không gian đặc trưng.
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

    Args:
        data_source:
            - Đường dẫn đến file .pt (str/Path): Ví dụ "data/processed/uci_har/dataset_all.pt"
            - Hoặc torch.Tensor: Dữ liệu đã load sẵn
        aug_pipeline:
            - Pipeline tăng cường dữ liệu (Data Augmentation)
            - Mặc định dùng TS_TCC_Augmentation (jitter, scaling, permutation...)

    Returns:
        Tuple (x_weak, x_strong):
            - x_weak: Góc nhìn yếu (ít biến đổi)
            - x_strong: Góc nhìn mạnh (biến đổi nhiều)
            Cả 2 đều có shape (6, 128)
    """

    def __init__(self,
                 data_source: Union[str, Path, torch.Tensor],
                 aug_pipeline=None
        ):
        # Nếu không truyền augmentation pipeline, dùng mặc định của TS-TCC
        self.aug_pipeline = aug_pipeline if aug_pipeline is not None else TS_TCC_Augmentation()

        # ============================================================
        # 1. XỬ LÝ DATA SOURCE: Hỗ trợ cả file path và tensor trực tiếp
        # ============================================================
        if isinstance(data_source, (str, Path)):
            # Trường hợp 1: data_source là đường dẫn file
            file_path = Path(data_source)
            if not file_path.exists():
                raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {file_path}")

            # Load file .pt (chỉ lấy samples, không cần labels vì SSL unsupervised)
            # weights_only=True: Tăng security, chỉ load tensor thuần túy
            loaded_data = torch.load(file_path, map_location="cpu", weights_only=True)

            # Nếu file là dict (chứa samples, labels, subjects) -> lấy samples
            # Nếu file chỉ là tensor (dataset_all.pt của MotionSense) -> lấy luôn
            self.samples = loaded_data["samples"] if isinstance(loaded_data, dict) else loaded_data

        elif isinstance(data_source, torch.Tensor):
            # Trường hợp 2: data_source đã là tensor sẵn (dùng cho debugging)
            self.samples = data_source
        else:
            raise TypeError("❌ data_source phải là đường dẫn file .pt hoặc torch.Tensor")

        # ============================================================
        # 2. ÉP KIỂU DỮ LIỆU VỀ FLOAT32 (chuẩn cho neural network)
        # ============================================================
        if not isinstance(self.samples, torch.Tensor):
            self.samples = torch.tensor(self.samples, dtype=torch.float32)
        else:
            self.samples = self.samples.float()

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Lấy mẫu thứ idx và sinh 2 views.

        Returns:
            Tuple (x_weak, x_strong):
                - x_weak:  (6, 128) - Góc nhìn yếu (ít thay đổi so với gốc)
                - x_strong: (6, 128) - Góc nhìn mạnh (thay đổi nhiều)
        """
        x_raw = self.samples[idx]  # Lấy mẫu gốc: shape (6, 128)
        return self.aug_pipeline(x_raw)  # Sinh cặp (x_weak, x_strong)

ContrastiveDataset = ContrastiveDatasetWrapper