"""
=============================================================================
VAI TRÒ CỦA FILE (datasets/kfold_splitter.py):
-----------------------------------------------------------------------------
1. Chuẩn bị dữ liệu cho đánh giá K-Fold Cross-Validation:
   - Thay vì chỉ chia dữ liệu 1 lần cố định thành Train/Test (dễ bị may rủi do
     phân bố dữ liệu), file này chia tập dữ liệu Target (như MotionSense hoặc
     UCI-HAR) thành K phần (Folds) độc lập.
   - Mỗi Fold lần lượt đóng vai trò làm tập kiểm thử (Test/Val set), K-1 phần
     còn lại làm tập huấn luyện (Train set), giúp đo lường độ ổn định thực tế
     của mô hình (Mean ± Std).

2. Đảm bảo tính cân bằng nhãn (Stratified Splitting):
   - Dữ liệu HAR có nhiều lớp hành động (Walking, Sitting, Jogging,...).
   - Hàm sử dụng StratifiedKFold để đảm bảo tỷ lệ phân bố giữa các lớp hành vi
     trong tập Train và Test ở mỗi Fold luôn giống hệt nhau, ngăn ngừa hiện
     tượng một lớp nào đó bị thiếu hụt khi đánh giá.

3. Tự động hóa tiền xử lý kiểu dữ liệu và sinh DataLoader:
   - Tự động kiểm tra, ép kiểu tensor chuẩn (Float32 cho tín hiệu, Int64 cho nhãn).
   - Tự động trích xuất metadata (số kênh cảm biến `in_channels`, số lớp `num_classes`).
   - Đóng gói trực tiếp thành PyTorch DataLoader (hỗ trợ pin_memory cho GPU).
=============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

# -------------------------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN (PROJECT ROOT SETUP)
# -------------------------------------------------------------------------
# __file__: Lấy đường dẫn file hiện tại (datasets/kfold_splitter.py)
# .resolve(): Chuyển đổi thành đường dẫn tuyệt đối chuẩn xác trên hệ thống
# .parent.parent: Lùi lại 2 cấp thư mục để trỏ đúng vào thư mục gốc CrossDomain-HAR-SSL
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thêm đường dẫn gốc vào sys.path để code có thể import các module anh em
# (ví dụ: import models, import utils) từ bất kỳ đâu mà không bị lỗi ModuleNotFoundError
sys.path.append(str(PROJECT_ROOT))


# -------------------------------------------------------------------------
# 2. HÀM CHÍNH: NẠP VÀ PHÂN CHIA K-FOLD
# -------------------------------------------------------------------------
def get_kfold_loaders(
        dataset_name="motionsense",  # Tên tập dữ liệu mục tiêu ('motionsense' hoặc 'uci_har')
        k=5,                         # Số lượng Folds cần chia (ví dụ: 3, 5, 7, 10)
        batch_size=64,               # Kích thước batch truyền vào mô hình mỗi bước forward
        seed=42,                     # Hạt giống ngẫu nhiên để cố định cách chia Fold (tái lập kết quả)
        num_workers=0                # Số luồng CPU dùng để load data (để 0 khi chạy trên Kaggle/máy cá nhân để tránh nghẽn)
    ):
    """
    Nạp dữ liệu đã tiền xử lý từ thư mục data/processed/{dataset_name}/dataset_all.pt
    và chia thành K Folds cân bằng lớp (Stratified).

    Returns:
        fold_loaders (list): Danh sách k tuple, mỗi phần tử là (train_loader, test_loader).
        in_channels (int): Số kênh cảm biến (thường là 6: 3 trục Acc + 3 trục Gyro).
        num_classes (int): Số lớp hành động (thường là 6 lớp tương ứng với 6 hành vi).
    """

    # ---------------------------------------------------------------------
    # Bước 1: Kiểm tra an toàn sự tồn tại của file dữ liệu
    # ---------------------------------------------------------------------
    # Tạo đường dẫn động đến file dữ liệu tổng của dataset chỉ định
    data_path = PROJECT_ROOT / "data" / "processed" / dataset_name / "dataset_all.pt"

    # Nếu chưa chạy bước tiền xử lý hoặc sai đường dẫn, báo lỗi ngay lập tức
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {data_path}")

    # ---------------------------------------------------------------------
    # Bước 2: Nạp dữ liệu và ép kiểu chuẩn cho PyTorch
    # ---------------------------------------------------------------------
    # Nạp dict dữ liệu vào bộ nhớ CPU (map_location='cpu' để không chiếm VRAM của GPU)
    data = torch.load(data_path, map_location="cpu")

    # X: Mảng tín hiệu cảm biến, kích thước (N, Channels=6, Seq_len=128)
    X = data["samples"]

    # y: Mảng nhãn hành vi, squeeze() để làm phẳng về 1 chiều (N,)
    y = data["labels"].squeeze()

    # Chuẩn hóa X thành kiểu Float32 (chuẩn bắt buộc khi đưa qua các lớp Conv1D/Linear)
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    else:
        X = X.float()

    # Chuẩn hóa y thành kiểu Int64 / Long (chuẩn bắt buộc khi tính CrossEntropyLoss)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).long()
    else:
        y = y.long()

    # ---------------------------------------------------------------------
    # Bước 3: Tự động trích xuất thông số cấu hình mạng (Metadata)
    # ---------------------------------------------------------------------
    # Lấy số trục cảm biến (chiều channel của tín hiệu, X.shape[1] = 6)
    in_channels = X.shape[1]

    # Tìm các nhãn duy nhất để xác định số lớp phân loại (num_classes = 6)
    num_classes = len(torch.unique(y))

    # ---------------------------------------------------------------------
    # Bước 4: Khởi tạo bộ chia Stratified K-Fold
    # ---------------------------------------------------------------------
    # n_splits=k: Số fold chia ra
    # shuffle=True: Xáo trộn vị trí các mẫu ngẫu nhiên trước khi chia
    # random_state=seed: Đảm bảo cùng seed thì luôn tạo ra các Fold giống hệt nhau
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_loaders = []

    # ---------------------------------------------------------------------
    # Bước 5: Lặp qua từng Fold để đóng gói DataLoader
    # ---------------------------------------------------------------------
    # skf.split nhận X và y (dưới dạng numpy) để phân tầng
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y.numpy())):

        # Cắt dữ liệu theo danh sách chỉ số (indices) của Fold hiện tại
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        test_ds = TensorDataset(X[test_idx], y[test_idx])

        # Loader cho tập Train: Xáo trộn (shuffle=True) sau mỗi epoch để mô hình học đều
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()  # Bật pin_memory nếu có GPU để nạp dữ liệu nhanh hơn
        )

        # Loader cho tập Test: Không xáo trộn (shuffle=False) để thứ tự đánh giá được giữ nguyên
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        # Đưa cặp DataLoader của Fold này vào danh sách trả về
        fold_loaders.append((train_loader, test_loader))

    return fold_loaders, in_channels, num_classes