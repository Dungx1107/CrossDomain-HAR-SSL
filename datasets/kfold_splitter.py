import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def get_kfold_loaders(
        dataset_name="motionsense",
        k=5,
        batch_size=64,
        seed=42,
        num_workers=0
    ):
    """
    Nạp dữ liệu đã xử lý từ data/processed/{dataset_name}/dataset_all.pt
    và phân chia thành K Folds có phân tầng (Stratified K-Fold).
    """
    data_path = PROJECT_ROOT / "data" / "processed" / dataset_name / "dataset_all.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")

    data = torch.load(data_path, map_location="cpu")
    X = data["samples"]
    y = data["labels"].squeeze()

    # Chuyển đổi định dạng an toàn
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    else:
        X = X.float()

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).long()
    else:
        y = y.long()

    # Xác định số kênh cảm biến (in_channels) và số lớp (num_classes)
    in_channels = X.shape[1]
    num_classes = len(torch.unique(y))

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_loaders = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y.numpy())):
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        test_ds = TensorDataset(X[test_idx], y[test_idx])

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        fold_loaders.append((train_loader, test_loader))

    return fold_loaders, in_channels, num_classes