# TÀI LIỆU KỸ THUẬT & GIẢI THÍCH CHI TIẾT: TIME-SERIES AUGMENTATION TRONG HỌC TỰ GIÁM SÁT (SSL)

---

## 1. CƠ SỞ LÝ THUYẾT: VAI TRÒ CỦA DATA AUGMENTATION TRONG SSL

### 1.1. Tư Duy Trực Giác: Con Người Nhận Diện Vạn Vật & Mạng Nơ-ron Tự Học Không Nhãn
Để hiểu bản chất của Data Augmentation trong Self-Supervised Learning (SSL), hãy liên hệ với cách con người nhận thức thế giới thực:
* **Cách con người nhận diện:** Nếu gặp một người bạn thân ngoài đời, dù họ thay đổi kiểu tóc, mặc chiếc áo màu khác, đeo khẩu trang hay đứng ở một góc nghiêng trong bóng râm, bạn vẫn lập tức nhận ra đó chính là họ. Bộ não con người đã tự động loại bỏ các yếu tố ngoại cảnh biến đổi (màu áo, ánh sáng, kiểu tóc) để trích xuất các **đặc trưng cốt lõi bất biến** (khung xương mặt, dáng đi, giọng nói).
* **Mạng nơ-ron trong SSL cũng hoạt động y hệt:** 
  Khi không có chuyên gia ngồi dán nhãn từng file dữ liệu (không có nhãn "Đi bộ", "Chạy bộ"), ta bắt mạng nơ-ron phải giải một câu đố tương tự:
  1. Lấy một cửa sổ tín hiệu cảm biến nguyên bản $\mathbf{X} \in \mathbb{R}^{C \times T}$.
  2. Cố tình "bóp méo" tín hiệu $\mathbf{X}$ theo hai cách độc lập để tạo ra hai góc nhìn khác nhau:
     - **Góc nhìn Yếu ($\mathbf{X}_{weak}$):** Rung lắc nhẹ (Jitter), co giãn biên độ một chút (Scale). Tín hiệu hầu như vẫn giữ nguyên dáng dấp ban đầu.
     - **Góc nhìn Mạnh ($\mathbf{X}_{strong}$):** Cắt xáo trộn vị trí các đoạn (Permutation), kéo giãn và uốn cong dòng thời gian lúc nhanh lúc chậm (Time-Warp). Tín hiệu bị biến dạng cấu trúc mạnh mẽ.
  3. **Nhiệm vụ tương phản (Contrastive Task):** Ép mô hình học sâu nhìn cả $\mathbf{X}_{weak}$ và $\mathbf{X}_{strong}$ rồi nhận định: *"Dù hai tín hiệu này bị bóp méo khác nhau, nhưng chúng cùng sinh ra từ một hành vi ban đầu $\mathbf{X}$!"*
  4. Lặp lại quá trình này hàng triệu lần trên dữ liệu không gán nhãn, mô hình sẽ tự động bỏ qua các nhiễu rung lắc cơ học và nắm bắt được bản chất ngữ nghĩa của từng hoạt động.

---

### 1.2. Cơ Chế Hai Góc Nhìn (Two-View Strategy) Trong Kiến Trúc TS-TCC
Kiến trúc **TS-TCC (Time-Series Representation Learning via Temporal and Contextual Contrasting)** không dùng chung một kiểu biến đổi mà phân tầng thành **Weak View** và **Strong View**:

```text
                     ┌────────────────────────────────────────┐
                     │   Tín hiệu gốc X (Shape: 6 x 128)      │
                     └───────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ Weak Augmentation ]                     [ Strong Augmentation ]
      Scaling nhẹ + Jittering nhẹ                Permutation + Time-Warping
                    │                                         │
                    ▼                                         ▼
         Góc nhìn Yếu (X_weak)                     Góc nhìn Mạnh (X_strong)
                    │                                         │
                    ▼                                         ▼
         [ Temporal Contrasting ]                  [ Contextual Contrasting ]
   (Dùng quá khứ X_weak dự đoán              (Kéo biểu diễn X_weak & X_strong
      tương lai của X_strong)                    lại gần nhau qua InfoNCE)

````

1. **Temporal Contrasting (Đối chiếu thời gian):** Mô hình dùng ngữ cảnh quá khứ của $\mathbf{X}\_{weak}$ để dự đoán trạng thái tương lai của $\mathbf{X}\_{strong}$. Việc này ép mô hình phải học mối tương quan thời gian bền vững, không thể "học vẹt" vì bước sóng tương lai ở view mạnh đã bị bóp méo trục thời gian.



2. **Contextual Contrasting (Đối chiếu ngữ cảnh):** Sử dụng hàm mất mát NT-Xent (InfoNCE) để tối đa hóa độ tương đồng giữa vector biểu diễn của $\mathbf{X}\_{weak}$ và $\mathbf{X}\_{strong}$ cùng gốc (Positive Pair), đồng thời đẩy xa vector của các mẫu khác trong cùng Batch (Negative Pairs).




### 1.3. Cơ Sở Toán Học & Ý Nghĩa Vật Lý Của Từng Thuật Toán Biến Đổi

#### 1. Jittering (Thêm Nhiễu Trắng Gaussian)

- **Bản chất toán học:** Thêm một biến ngẫu nhiên tuân theo phân phối chuẩn độc lập vào từng điểm thời gian:




  $$\mathbf{x}'(t) = \mathbf{x}(t) + \epsilon(t), \quad \epsilon(t) \sim \mathcal{N}(0, \sigma\_{jitter}^2)$$
- **Ý nghĩa vật lý cảm biến MEMS:** Mô phỏng sự dao động nhiệt của vi mạch điện tử và rung chấn cơ học ngẫu nhiên khi điện thoại cọ xát với túi quần.




#### 2. Scaling (Co Giãn Biên Độ Ngẫu Nhiên)

- **Bản chất toán học:** Nhân toàn bộ tín hiệu của từng kênh cảm biến với một hệ số tỉ lệ ngẫu nhiên:




  $$\mathbf{x}'\_c(t) = \alpha\_c \cdot \mathbf{x}\_c(t), \quad \alpha\_c \sim \mathcal{N}(1.0, \sigma\_{scale}^2)$$
- **Ý nghĩa vật lý cảm biến MEMS:** Mô phỏng sự khác biệt về thể trọng, sức mạnh cơ bắp và lực giẫm chân giữa các đối tượng khác nhau (người nặng cân hơn tạo ra gia tốc tiếp đất lớn hơn).




#### 3. Permutation (Xáo Trộn Thứ Tự Đoạn Thời Gian)

- **Bản chất toán học:** Chia chiều dài $T$ thành $K$ đoạn con bằng nhau:




  $$\mathbf{S} = [\mathbf{s}\_1, \mathbf{s}\_2, \dots, \mathbf{s}\_K], \quad \text{với } \text{len}(\mathbf{s}\_k) = \lfloor T/K \rfloor$$

  Xáo trộn thứ tự các đoạn theo một hoán vị ngẫu nhiên $\pi$:




  $$\mathbf{x}' = \text{Concat}(\mathbf{s}\_{\pi(1)}, \mathbf{s}\_{\pi(2)}, \dots, \mathbf{s}\_{\pi(K)})$$
- **Ý nghĩa học máy:** Phá vỡ trật tự chuỗi thời gian dài hạn nhưng giữ nguyên hình dạng sóng cục bộ bên trong mỗi đoạn con, buộc mạng nơ-ron phải nhận diện hành vi dựa trên đặc trưng cục bộ (Local Patterns).




#### 4. Time-Warping (Uốn Cong Trục Thời Gian Phi Tuyến Bằng Cubic Spline)

- **Bản chất toán học:**



  1. Đặt $M$ điểm mốc thời gian dọc theo cửa sổ: $t\_0, t\_1, \dots, t\_{M+1}$.



  2. Tạo độ lệch ngẫu nhiên tại các mốc: $\tau\_m = t\_m \cdot \eta\_m$, với $\eta\_m \sim \mathcal{N}(1.0, \sigma\_{warp}^2)$.



  3. Khóa an toàn 2 biên: $\tau\_0 = 0$ và $\tau\_{M+1} = T-1$.



  4. Nội suy đường cong thời gian liên tục $\phi(t)$ bằng **Cubic Spline**, sau đó lấy mẫu lại tín hiệu: $\mathbf{x}'(t) = \mathbf{x}(\phi(t))$.



- **Ý nghĩa vật lý cảm biến MEMS:** Mô phỏng sự thay đổi nhịp điệu vận động bất đối xứng (lúc bước nhanh vội vã, lúc bước chậm lại).




### 1.4. Vai Trò Của Lớp Bọc Dữ Liệu (`SSLTimeSeriesDataset`)

- **Chuyển đổi chiều Tensor tự động:** Đảm bảo ma trận dữ liệu luôn có dạng $(N, C, T)$ tương thích hoàn toàn với các tầng tích chập `torch.nn.Conv1d`.



- **Cơ chế On-The-Fly Generation:** Cặp $(\mathbf{X}\_{weak}, \mathbf{X}\_{strong})$ được tính toán tức thời trên CPU/RAM mỗi khi DataLoader gọi `__getitem__`. Nhờ đó, qua mỗi Epoch, mạng nơ-ron luôn nhìn thấy các biến thể mới mà không tốn dung lượng bộ nhớ lưu trữ sẵn ra ổ cứng.




## 2. NGUỒN TRÍCH DẪN THAM KHẢO HỌC THUẬT (ACADEMIC REFERENCES)

1. **Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021).** *Time-series representation learning via temporal and contextual contrasting.* In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), pp. 2352-2359. [Link: https\://doi.org/10.24963/ijcai.2021/324]



2. **Iwana, B. K., & Uchida, S. (2021).** *An empirical survey of data augmentation for time series classification with neural networks.* PLOS ONE, 16(7), e0254841. [Link: https\://doi.org/10.1371/journal.pone.0254841]



3. **Um, T. T., Pfister, F. M., Pichler, D., Endo, S., Lang, M., Hirche, S., ... & Kulić, D. (2017).** *Data augmentation of wearable sensor data for Parkinson's disease monitoring using convolutional neural networks.* In Proceedings of the 19th ACM ICMI, pp. 216-220.



4. **Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).** *A simple framework for contrastive learning of visual representations.* In ICML 2020. (Nguyên lý Two-View Contrastive Loss).




## 3. MÃ NGUỒN ĐẦY ĐỦ KÈM CHÚ THÍCH TỪNG DÒNG (FULL ANNOTATED CODE)

Python

```
"""
===============================================================================
MODULE: utils/augmentations.py & datasets/ssl_dataset.py
MỤC ĐÍCH:
    Module cung cấp các thuật toán biến đổi tín hiệu chuỗi thời gian (Time-Series
    Data Augmentation) và bọc dữ liệu (Dataset Wrapper) phục vụ cho quá trình
    huấn luyện Tự giám sát (Self-Supervised Learning - SSL) theo chuẩn TS-TCC.
===============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline


# =============================================================================
# 1. CÁC HÀM BIẾN ĐỔI TÍN HIỆU ĐƠN LẺ (SIGNAL TRANSFORMATIONS)
# =============================================================================

def jitter(x, sigma=0.05):
    """
    Thêm nhiễu trắng Gaussian vào tín hiệu chuỗi thời gian.
    Mô phỏng nhiễu cảm biến nhiệt và rung chấn cơ học ngẫu nhiên.

    Args:
        x (np.ndarray): Tín hiệu đầu vào, shape (C, T) với C là số kênh, T là độ dài.
        sigma (float): Độ lệch chuẩn của phân phối Gauss.

    Returns:
        np.ndarray: Tín hiệu sau khi cộng nhiễu, cùng shape (C, T).
    """
    # Sinh ma trận nhiễu N(0, sigma^2) có cùng kích thước với tín hiệu x
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    # Cộng trực tiếp vào tín hiệu gốc
    return x + noise


def scaling(x, sigma=0.1):
    """
    Nhân toàn bộ kênh tín hiệu với hệ số co giãn biên độ ngẫu nhiên.
    Mô phỏng sự khác biệt về lực tác động, thể trọng giữa các cá nhân.

    Args:
        x (np.ndarray): Tín hiệu đầu vào, shape (C, T).
        sigma (float): Độ biến thiên của hệ số co giãn quanh giá trị 1.0.

    Returns:
        np.ndarray: Tín hiệu sau khi co giãn biên độ, shape (C, T).
    """
    # Sinh hệ số tỉ lệ ngẫu nhiên cho từng kênh riêng biệt: shape (C, 1)
    # Tự động broadcast phép nhân dọc theo toàn bộ trục thời gian T
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], 1))
    return x * factor


def time_warp(x, sigma=0.2, num_knots=4):
    """
    Biến đổi tốc độ cục bộ của tín hiệu bằng Cubic Spline (Uốn cong trục thời gian).
    Mô phỏng nhịp điệu bước chân thay đổi bất đối xứng (nhanh/chậm bất thường).

    Args:
        x (np.ndarray): Tín hiệu đầu vào, shape (C, T).
        sigma (float): Mức độ co giãn ngẫu nhiên của các điểm mốc.
        num_knots (int): Số điểm mốc (knots) trung gian để nội suy spline.

    Returns:
        np.ndarray: Tín hiệu sau khi uốn cong trục thời gian, shape (C, T).
    """
    C, T = x.shape
    time_orig = np.arange(T)  # Trục thời gian gốc: [0, 1, 2, ..., T-1]

    # Thiết lập các điểm mốc thời gian cách đều nhau dọc theo cửa sổ
    knot_positions = np.linspace(0, T - 1, num=num_knots + 2)

    # Sinh độ lệch ngẫu nhiên tại các mốc
    random_shifts = np.random.normal(loc=1.0, scale=sigma, size=(num_knots + 2,))
    warped_knot_positions = knot_positions * random_shifts

    # Khóa an toàn 2 biên: Ép điểm đầu luôn là 0 và điểm cuối luôn là T-1
    warped_knot_positions = (warped_knot_positions - warped_knot_positions[0]) / (
        warped_knot_positions[-1] - warped_knot_positions[0]
    ) * (T - 1)

    # Xây dựng đường cong nội suy mượt Cubic Spline
    spline = CubicSpline(knot_positions, warped_knot_positions)
    warped_time = spline(time_orig)
    warped_time = np.clip(warped_time, 0, T - 1)  # Giới hạn giá trị trong khoảng hợp lệ

    # Lấy mẫu lại (Resampling) từng kênh tín hiệu trên trục thời gian mới
    x_warped = np.zeros_like(x)
    for c in range(C):
        x_warped[c] = np.interp(time_orig, warped_time, x[c])

    return x_warped


def permutation(x, max_segments=4):
    """
    Chia tín hiệu thành N đoạn con bằng nhau và xáo trộn ngẫu nhiên thứ tự.
    Thử thách khả năng nhận diện mẫu đặc trưng cục bộ (Local Motifs) của mô hình.

    Args:
        x (np.ndarray): Tín hiệu đầu vào, shape (C, T).
        max_segments (int): Số đoạn con cần chia.

    Returns:
        np.ndarray: Tín hiệu sau khi xáo trộn đoạn, shape (C, T).
    """
    C, T = x.shape
    seg_len = T // max_segments
    segments = []

    # Cắt tín hiệu thành max_segments đoạn con theo trục thời gian
    for i in range(max_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < max_segments - 1 else T
        segments.append(x[:, start:end])

    # Xáo trộn thứ tự các đoạn ngẫu nhiên tại chỗ
    np.random.shuffle(segments)
    
    # Ghép nối lại các đoạn thành mảng hoàn chỉnh theo trục thời gian (axis=1)
    return np.concatenate(segments, axis=1)


# =============================================================================
# 2. BỘ ĐIỀU PHỐI WEAK & STRONG AUGMENTATION
# =============================================================================

class TS_TCC_Augmentation:
    """
    Bộ tạo 2 views (Weak view và Strong view) chuẩn hóa theo kiến trúc TS-TCC.
    """
    def __init__(self, jitter_sigma=0.05, scale_sigma=0.1, warp_sigma=0.2, n_perm=4):
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.warp_sigma = warp_sigma
        self.n_perm = n_perm

    def weak_transform(self, x):
        """View yếu: Giữ lại hình dạng cơ bản thông qua Scaling + Jittering nhẹ."""
        x_aug = scaling(x, sigma=self.scale_sigma)
        x_aug = jitter(x_aug, sigma=self.jitter_sigma)
        return x_aug

    def strong_transform(self, x):
        """View mạnh: Biến dạng cấu trúc mạnh qua Permutation + Time-Warping."""
        x_aug = permutation(x, max_segments=self.n_perm)
        x_aug = time_warp(x_aug, sigma=self.warp_sigma)
        return x_aug

    def __call__(self, x):
        """
        Nhận vào mảng numpy (C, T) và trả về 2 views dạng PyTorch Tensor (float32).
        """
        x_w = self.weak_transform(x.copy())
        x_s = self.strong_transform(x.copy())

        return (
            torch.tensor(x_w, dtype=torch.float32),
            torch.tensor(x_s, dtype=torch.float32)
        )


# =============================================================================
# 3. PYTORCH DATASET DÀNH CHO SSL PRE-TRAINING (KHÔNG CẦN NHÃN)
# =============================================================================

class SSLTimeSeriesDataset(Dataset):
    """
    Dataset bọc dữ liệu chuỗi thời gian phục vụ Self-Supervised Pre-training.
    Mỗi bước lặp trả về bộ đôi (x_weak, x_strong) phục vụ tính Contrastive Loss.
    """
    def __init__(self, data, augmentor=None):
        """
        Args:
            data (np.ndarray): Mảng dữ liệu shape (N, C, T) hoặc (N, T, C).
            augmentor (callable, optional): Đối tượng tạo biến đổi Weak/Strong.
        """
        # Tự động phát hiện và chuẩn hóa chiều không gian về chuẩn Conv1D: (N, C, T)
        if data.shape[1] > data.shape[2]:  # Nếu dữ liệu đang ở dạng (N, T, C)
            data = np.transpose(data, (0, 2, 1))

        self.data = data
        self.augmentor = augmentor if augmentor is not None else TS_TCC_Augmentation()

    def __len__(self):
        """Trả về tổng số lượng mẫu cửa sổ trong tập huấn luyện SSL."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Truy xuất mẫu tại vị trí idx và sinh trực tiếp cặp (x_weak, x_strong).
        """
        x = self.data[idx]
        x_weak, x_strong = self.augmentor(x)
        return x_weak, x_strong

```

## 4. ĐỐI CHIẾU VỚI CODE GỐC TỪ BÀI BÁO TS-TCC (GITHUB REFERENCE)

Dưới đây là đoạn mã nguồn tham khảo từ kho lưu trữ GitHub chính thức của bài báo TS-TCC:

Python

```
# Code tham khảo từ bài báo gốc TS-TCC (Eldele et al., IJCAI 2021)
# [https://github.com/emadeldele/TS-TCC](https://github.com/emadeldele/TS-TCC)

def DataTransform(sample, config):
    # Góc nhìn yếu: Chỉ áp dụng Scaling
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # Góc nhìn mạnh: Kết hợp Permutation rồi cộng thêm Jittering
    strong_aug = jitter(
        permutation(sample, max_segments=config.augmentation.max_seg), 
        config.augmentation.jitter_ratio
    )
    return weak_aug, strong_aug

```

### Điểm Cải Tiến Trong Dự Án Này

1. **Bổ sung Time-Warping bằng Cubic Spline:** Thay vì chỉ dùng Jittering trên View mạnh như bản gốc của tác giả, việc tích hợp thêm **Time-Warping** giúp mô hình bất biến tốt hơn đối với sự thay đổi vận tốc di chuyển ngoài đời thực.



2. **Khóa biên an toàn $[0, T-1]$:** Thuật toán `time_warp` có thêm cơ chế chuẩn hóa 2 đầu mốc, ngăn chặn triệt để lỗi ngoại suy (`out-of-bounds index`) khi xử lý các chuỗi thời gian ngắn.



3. **Đóng gói PyTorch Dataset hướng đối tượng (****`SSLTimeSeriesDataset`****):** Tách bạch rõ ràng giữa thuật toán biến đổi tín hiệu thuần túy và luồng nạp dữ liệu Mini-Batch, giúp mã nguồn dễ bảo trì và mở rộng sang các bộ dữ liệu khác (`HHAR`, `UCI-HAR`).
