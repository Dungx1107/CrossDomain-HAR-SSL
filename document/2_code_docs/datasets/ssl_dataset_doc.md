# TÀI LIỆU KỸ THUẬT & GIẢI THÍCH CHI TIẾT: SSL TIME-SERIES DATASET

---

## 1. CƠ SỞ LÝ THUYẾT VÀ Ý NGHĨA TRONG HỆ THỐNG HAR-SSL

### 1.1. Bản Chất Của Học Tự Giám Sát (Self-Supervised Pre-training)
Trong bài toán nhận dạng hoạt động người dùng (HAR), lượng dữ liệu cảm biến thu được từ thiết bị đeo là vô cùng lớn nhưng hầu hết đều **không có nhãn (Unlabeled)**.
- Lớp `SSLTimeSeriesDataset` được thiết kế riêng cho giai đoạn **Pre-training**, loại bỏ hoàn toàn sự phụ thuộc vào nhãn $y$.
- Thay vì dự đoán nhãn hoạt động, mạng nơ-ron được giao nhiệm vụ học biểu diễn bất biến thông qua cơ chế so sánh cặp góc nhìn $(\mathbf{x}^w, \mathbf{x}^s)$ từ cùng một mẫu tín hiệu gốc $\mathbf{x}$.

---

### 1.2. Cơ Chế Xử Lý Định Dạng Chiều Không Gian (Dimensionality Normalization)
Các thư viện xử lý chuỗi thời gian (như Pandas, Scikit-Learn) và PyTorch có quy ước chiều không gian khác nhau:
- **Chuẩn Dữ Liệu Bảng / CSV:** Thường có dạng $(\text{Samples}, \text{Time Steps}, \text{Channels})$ hay $(N, T, C)$ do mỗi dòng là 1 mốc thời gian và các cột là kênh cảm biến.
- **Chuẩn PyTorch Conv1D:** Yêu cầu định dạng $(\text{Batch Size}, \text{Channels}, \text{Length})$ hay $(N, C, T)$.

Đoạn code tự động phát hiện và chuyển đổi:
$$\text{If } \text{dim}_1 > \text{dim}_2 \implies \text{Transpose}(0, 2, 1): (N, T, C) \longrightarrow (N, C, T)$$
Giúp hệ thống linh hoạt, chống lỗi `RuntimeError` khi nhận dữ liệu từ các nguồn khác nhau mà không cần tiền xử lý thủ công bên ngoài.

---

### 1.3. Cơ Chế Sinh Cặp Views Trực Tiếp Khi Huấn Luyện (On-The-Fly Augmentation)
- Thay vì tạo sẵn (offline) các mẫu biến đổi làm tăng dung lượng lưu trữ trên đĩa cứng, `__getitem__` thực hiện biến đổi **trực tiếp (On-The-Fly)** mỗi khi một Mini-batch được gọi.
- Mỗi epoch, một cửa sổ dữ liệu $\mathbf{x}$ sẽ được tạo ra một cặp $(\mathbf{x}^w, \mathbf{x}^s)$ hoàn toàn mới do tính chất ngẫu nhiên của nhiễu Gaussian và đường cong Cubic Spline, giúp chống hiện tượng Overfitting hiệu quả.

---

## 2. NGUỒN TRÍCH DẪN THAM KHẢO HỌC THUẬT (ACADEMIC REFERENCES)

1. **Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021).** *Time-series representation learning via temporal and contextual contrasting.* In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), pp. 2352-2359.
2. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).** *A simple framework for contrastive learning of visual representations.* In International Conference on Machine Learning (ICML 2020), pp. 1597-1607. (Cơ chế Two-View Data Augmentation trong Contrastive Learning).
3. **PyTorch Documentation:** *torch.utils.data.Dataset*. [https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

---

## 3. MÃ NGUỒN ĐẦY ĐỦ KÈM CHÚ THÍCH TỪNG DÒNG (FULL ANNOTATED CODE)

```python

import numpy as np
from torch.utils.data import Dataset
from utils.augmentations import TS_TCC_Augmentation


class SSLTimeSeriesDataset(Dataset):
    """
    Dataset bọc dữ liệu chuỗi thời gian phục vụ tiền huấn luyện Tự giám sát (Self-Supervised Learning).

    Đặc điểm:
        - Không cần nhãn (Unlabeled Data): Tận dụng toàn bộ dữ liệu cảm biến chưa gán nhãn.
        - Mỗi lần lấy mẫu (`__getitem__`), dữ liệu được đi qua bộ `augmentor` để sinh ra
          cặp góc nhìn (x_weak, x_strong) phục vụ Contrastive Loss.
    """

    def __init__(self, data, augmentor=None):
        """
        Khởi tạo và chuẩn hóa định dạng Tensor đầu vào.

        Tham số:
            data (np.ndarray hoặc torch.Tensor):
                Mảng dữ liệu chuỗi thời gian. Chấp nhận 2 định dạng:
                - (N, C, T): N mẫu, C kênh cảm biến (vd: 6), T điểm thời gian (vd: 128).
                - (N, T, C): N mẫu, T điểm thời gian (128), C kênh cảm biến (6).
            augmentor (callable, optional):
                Đối tượng thực hiện biến đổi dữ liệu. Mặc định là `TS_TCC_Augmentation()`.
        """
        super().__init__()

        # Kiểm tra và tự động hoán vị trục ma trận nếu dữ liệu đầu vào là dạng (N, T, C)
        # Trong HAR: Thông thường T (128 điểm mẫu) > C (3 hoặc 6 kênh cảm biến)
        # Mục tiêu: Đưa về chuẩn (N, C, T) để tương thích với PyTorch Conv1D
        if data.shape[1] > data.shape[2]:  # Điều kiện nhận biết shape đang là (N, 128, 6)
            data = np.transpose(data, (0, 2, 1))  # Chuyển trục: 0 giữ nguyên, 1 <-> 2 => (N, 6, 128)

        self.data = data
        # Nếu không truyền augmentor tùy biến thì sử dụng bộ biến đổi mặc định TS-TCC
        self.augmentor = augmentor if augmentor is not None else TS_TCC_Augmentation()

    def __len__(self):
        """
        Trả về tổng số lượng mẫu cửa sổ chuỗi thời gian có trong tập huấn luyện SSL.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Truy xuất một mẫu dữ liệu tại vị trí chỉ mục `idx` và áp dụng Data Augmentation.

        Tham số:
            idx (int): Chỉ mục của mẫu cần lấy (từ 0 đến len(data) - 1).

        Trả về:
            tuple: (x_weak, x_strong)
                - x_weak (torch.Tensor): Góc nhìn biến đổi yếu (Scaling + Jittering nhẹ), shape (C, T).
                - x_strong (torch.Tensor): Góc nhìn biến đổi mạnh (Permutation + Time-Warping), shape (C, T).
        """
        # Trích xuất ma trận cảm biến của 1 cửa sổ thời gian tại index idx: shape (C, T)
        x = self.data[idx]

        # Áp dụng bộ biến đổi tạo ra 2 góc nhìn đối sánh
        x_weak, x_strong = self.augmentor(x)

        return x_weak, x_strong