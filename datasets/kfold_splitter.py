import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from datasets.base_dataset import HARDataset


def get_kfold_loaders(dataset_name="motionsense", k=5, batch_size=64, seed=42, num_workers=2):
    """
    Nạp dataset .pt đã chuẩn hóa và chia thành K Folds (Stratified).

    Args:
        dataset_name (str): Tên dataset ('motionsense', 'uci_har',...)
        k (int): Số lượng folds
        batch_size (int): Kích thước batch cho DataLoader
        seed (int): Random seed cố định
        num_workers (int): Số worker nạp dữ liệu

    Returns:
        list of tuples: [(train_loader_1, test_loader_1), ..., (train_loader_k, test_loader_k)]
        int: Số lượng class duy nhất
    """
    data_path = PROJECT_ROOT / "data" / "processed" / dataset_name / "dataset_all.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại: {data_path}")

    data = torch.load(data_path, map_location="cpu")
    X = data["samples"]
    y = data["labels"].squeeze()

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    else:
        X = X.float()

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).long()
    else:
        y = y.long()

    num_classes = len(torch.unique(y))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_loaders = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y.numpy())):
        train_ds = HARDataset(X[train_idx], y[train_idx])
        test_ds = HARDataset(X[test_idx], y[test_idx])

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

    return fold_loaders, num_classes