"""
===============================================================================
MODULE: BASE HAR DATASET & DATALOADER FACTORY
===============================================================================
Mục đích:
    - Cung cấp class BaseHARDataset dùng chung cho toàn bộ dữ liệu HAR dạng .pt.
    - Hỗ trợ slicing tỷ lệ ít nhãn (Few-shot fraction) và phân tầng (stratified) nếu cần.
    - Cung cấp hàm get_har_dataloaders() tự động build 3 loader: Train, Val, Test.
===============================================================================
"""

from pathlib import Path
from typing import Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader


class BaseHARDataset(Dataset):
    """
    Dataset chuẩn hóa cho bài toán HAR từ file .pt đã tiền xử lý.

    Args:
        pt_file_path: Đường dẫn đến file .pt (ví dụ: data/processed/uci_har/train.pt)
        fraction: Tỷ lệ dữ liệu muốn giữ lại (0.01 = 1%, 0.1 = 10%, 1.0 = 100%)
        seed: Seed dùng để random khi cắt fraction, giúp reproducible
    """

    def __init__(self,
                 pt_file_path: Union[str, Path],
                 fraction: float = 1.0,
                 seed: int = 42
    ):
        # Chuyển đường dẫn về Path object để dễ xử lý
        pt_file_path = Path(pt_file_path)
        if not pt_file_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {pt_file_path}")

        # Load dữ liệu từ file .pt (bao gồm samples, labels, subjects)
        data = torch.load(pt_file_path, map_location="cpu")

        # Lấy samples (dữ liệu cảm biến) và labels (nhãn hoạt động)
        self.samples = data["samples"]
        self.labels = data["labels"].squeeze()  # squeeze() để bỏ chiều thừa

        # Lấy subjects nếu có (dùng cho cross-domain validation)
        self.subjects = data.get("subjects", None)
        if self.subjects is not None:
            self.subjects = self.subjects.squeeze()

        # Ép kiểu dữ liệu về đúng chuẩn PyTorch
        # samples -> float32, labels -> long (dùng cho CrossEntropyLoss)
        if not isinstance(self.samples, torch.Tensor):
            self.samples = torch.tensor(self.samples, dtype=torch.float32)
        else:
            self.samples = self.samples.float()

        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = self.labels.long()

        # ============================================================
        # XỬ LÝ FEW-SHOT: Cắt dữ liệu theo tỷ lệ fraction (1%, 5%, 10%)
        # ============================================================
        if 0.0 < fraction < 1.0:
            num_total = len(self.labels)                      # Tổng số mẫu ban đầu
            num_keep = max(1, int(num_total * fraction))     # Số mẫu cần giữ lại (tối thiểu 1)

            # Tạo generator với seed cố định để kết quả reproducible
            g = torch.Generator().manual_seed(seed)

            # Random permutation (xáo trộn) và lấy num_keep mẫu đầu tiên
            indices = torch.randperm(num_total, generator=g)[:num_keep]

            # Cắt samples, labels, subjects theo indices đã chọn
            self.samples = self.samples[indices]
            self.labels = self.labels[indices]
            if self.subjects is not None:
                self.subjects = self.subjects[indices]

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lấy mẫu thứ idx trong dataset.

        Returns:
            Tuple (samples, labels): samples shape (6, 128), labels là int (0-5)
        """
        return self.samples[idx], self.labels[idx]


def get_har_dataloaders(
        data_dir: Union[str, Path],
        batch_size: int = 64,
        train_fraction: float = 1.0,
        num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Hàm tạo DataLoader dùng chung cho Train/Val/Test.

    Args:
        data_dir: Thư mục chứa các file train.pt, val.pt, test.pt
        batch_size: Số mẫu trong 1 batch
        train_fraction: Tỷ lệ dữ liệu train cần lấy (Few-shot)
        num_workers: Số luồng CPU dùng để load dữ liệu

    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    # ============================================================
    # 1. Tạo 3 dataset: Train, Val, Test
    # ============================================================
    # Train: có thể cắt fraction (few-shot)
    train_ds = BaseHARDataset(data_dir / "train.pt", fraction=train_fraction)

    # Val và Test: luôn giữ 100% dữ liệu để đánh giá đúng
    val_ds = BaseHARDataset(data_dir / "val.pt", fraction=1.0)
    test_ds = BaseHARDataset(data_dir / "test.pt", fraction=1.0)

    # ============================================================
    # 2. Tạo DataLoader cho từng tập
    # ============================================================
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,                                    # Xáo trộn để training tốt hơn
        drop_last=True if len(train_ds) >= batch_size else False,  # Bỏ batch cuối nếu không đủ
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,                                   # Validation không cần xáo trộn
        drop_last=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def get_har_all_loader(
        data_dir: Union[str, Path],
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0
) -> DataLoader:
    """
    Nạp toàn bộ dataset_all.pt (gộp cả Train + Val + Test).

    Dùng cho bước Pretrain SSL vì SSL cần nhiều dữ liệu không nhãn.

    Args:
        data_dir: Thư mục chứa file dataset_all.pt
        batch_size: Số mẫu trong 1 batch
        shuffle: Có xáo trộn dữ liệu không (thường là True cho pretrain)
        num_workers: Số luồng CPU

    Returns:
        DataLoader chứa toàn bộ dữ liệu
    """
    data_dir = Path(data_dir)
    all_ds = BaseHARDataset(data_dir / "dataset_all.pt", fraction=1.0)

    return DataLoader(
        all_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True if len(all_ds) >= batch_size else False,
        num_workers=num_workers
    )