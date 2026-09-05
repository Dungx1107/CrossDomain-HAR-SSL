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
    """

    def __init__(self, pt_file_path: Union[str, Path], fraction: float = 1.0, seed: int = 42):
        pt_file_path = Path(pt_file_path)
        if not pt_file_path.exists():
            raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {pt_file_path}")

        data = torch.load(pt_file_path, map_location="cpu")

        self.samples = data["samples"]
        self.labels = data["labels"].squeeze()
        self.subjects = data.get("subjects", None)
        if self.subjects is not None:
            self.subjects = self.subjects.squeeze()

        # Ép kiểu dữ liệu tiêu chuẩn
        if not isinstance(self.samples, torch.Tensor):
            self.samples = torch.tensor(self.samples, dtype=torch.float32)
        else:
            self.samples = self.samples.float()

        if not isinstance(self.labels, torch.Tensor):
            self.labels = torch.tensor(self.labels, dtype=torch.long)
        else:
            self.labels = self.labels.long()

        # Xử lý cắt tỷ lệ Few-shot (1%, 5%, 10%...)
        if 0.0 < fraction < 1.0:
            num_total = len(self.labels)
            num_keep = max(1, int(num_total * fraction))

            g = torch.Generator().manual_seed(seed)
            indices = torch.randperm(num_total, generator=g)[:num_keep]

            self.samples = self.samples[indices]
            self.labels = self.labels[indices]
            if self.subjects is not None:
                self.subjects = self.subjects[indices]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


def get_har_dataloaders(
        data_dir: Union[str, Path],
        batch_size: int = 64,
        train_fraction: float = 1.0,
        num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Hàm Loader dùng chung: Nạp trực tiếp từ thư mục chứa train.pt, val.pt, test.pt.

    Args:
        data_dir: Thư mục chứa các file .pt (ví dụ: data/processed/uci_har)
        batch_size: Kích thước batch
        train_fraction: Tỷ lệ tập train cần lấy (Few-shot)
        num_workers: Số luồng nạp dữ liệu
    """
    data_dir = Path(data_dir)

    train_ds = BaseHARDataset(data_dir / "train.pt", fraction=train_fraction)
    val_ds = BaseHARDataset(data_dir / "val.pt", fraction=1.0)
    test_ds = BaseHARDataset(data_dir / "test.pt", fraction=1.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(train_ds) >= batch_size else False,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
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
        num_workers: int = 0) -> DataLoader:
    """Nạp toàn bộ dataset_all.pt cho bước Pretrain SSL."""
    data_dir = Path(data_dir)
    all_ds = BaseHARDataset(data_dir / "dataset_all.pt", fraction=1.0)

    return DataLoader(
        all_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True if len(all_ds) >= batch_size else False,
        num_workers=num_workers
    )