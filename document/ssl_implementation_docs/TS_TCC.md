# Tài liệu Kỹ thuật: TS-TCC (Temporal and Contextual Contrasting)

> **Mục đích tài liệu:** Mô tả chính xác implementation của phương pháp TS-TCC trong repository CrossDomain-HAR-SSL dựa trên phân tích mã nguồn thực tế.  
> **Phiên bản phân tích:** Dựa trên snapshot code tại thời điểm viết tài liệu.  
> **Tài liệu này dành cho:** AI review code — mọi thông tin đều dẫn chiếu trực tiếp đến file và function tương ứng.

---

## Phần 1. Tổng quan

### 1.1. Mục tiêu trong dự án

TS-TCC (Temporal and Contextual Contrasting) là một trong ba phương pháp SSL được triển khai trong dự án. Mục tiêu của nó là học biểu diễn đặc trưng từ chuỗi thời gian cảm biến quán tính (6 kênh, 128 timestep) mà **không cần nhãn hoạt động**.

Trong dự án này, TS-TCC được sử dụng để pre-train encoder trên source domain (MotionSense, UCI-HAR, HHAR), sau đó encoder pretrained được chuyển sang target domain để đánh giá qua hai protocol:
- **Linear Probing (LP):** Đóng băng encoder, chỉ huấn luyện classifier head trên dữ liệu đích có nhãn.
- **Full Fine-tuning (FT):** Mở khóa toàn bộ mạng, fine-tune encoder + head trên dữ liệu đích.

### 1.2. Lý thuyết gốc

**Lý thuyết (Eldele et al., IJCAI 2021 - emadeldeen24/TS-TCC):**
- Tạo hai view augmented từ cùng một chuỗi thời gian: **weak view** và **strong view**.
- **Temporal Contrasting (TC):** Dùng một Transformer để trích xuất context vector từ quá khứ, sau đó dự đoán đặc trưng tương lai của view còn lại (InfoNCE loss dạng CPC).
- **Contextual Contrasting (CC):** Kéo context vector của hai view lại gần nhau trong không gian embedding (NT-Xent loss).
- Loss tổng: $\mathcal{L} = \lambda_1 \mathcal{L}_{TC} + \lambda_2 \mathcal{L}_{CC}$.

### 1.3. Vai trò trong pipeline tổng thể

**Xác minh từ code:**
```text
SSL Pretraining (source domain)         → checkpoints/ssl_pretrain/contrastive/<source>/<backbone>/
        |
        ↓  encoder.state_dict() được lưu
K-shot / Fraction Cross-Domain          → run_few_shot.py / run_cross_domain.py
        |
        ↓  load encoder, freeze hoặc finetune
Linear Probing / Full Fine-tuning       → engines/transfer/finetune_trainer.py
        |
        ↓  
Đánh giá trên target test set           → outputs_evaluation/cross_k_shot/contrastive/
```

### 1.4. Các thành phần code chính

| Thành phần | File | Mô tả |
|---|---|---|
| Augmentation | `models/ssl/contrastive/augmentations.py` | Tạo weak/strong view |
| TSTCCEncoder | `models/encoders/tstcc_encoder.py` | Backbone 1D-CNN (3 block) |
| TC module | `models/ssl/contrastive/tc.py` | Temporal Contrasting, InfoNCE |
| Seq_Transformer | `models/ssl/contrastive/attention.py` | Transformer trích xuất c_t |
| TSTCCModel | `models/ssl/contrastive/ts_tcc_model.py` | Wrapper gộp encoder + TC + CC |
| NTXentLoss | `losses/nt_xent.py` | Contextual Contrasting loss |
| ContrastiveDatasetWrapper | `datasets/contrastive_dataset.py` | Dataset sinh (x_weak, x_strong) |
| Contrastive Trainer | `../../engines/pretrain_ssl/tstcc_trainer.py` | Training loop SSL |
| Entry point | `../../scripts/02_pretrain_ssl/run_pretrain_tstcc.py` | CLI script khởi chạy |

---

## Phần 2. Cấu trúc mã nguồn liên quan

### 2.1. Entry point và cấu hình

**File:** `../../scripts/02_pretrain_ssl/run_pretrain_tstcc.py`
- **Vai trò:** Script CLI chính để khởi chạy pretraining TS-TCC.
- **Nhận argument:** `--backbone` (default: `vit_1d`), `--epochs` (default: 60), `--batch_size` (default: 64), `--datasets` (default: `["uci_har", "motionsense"]`).
- **Gọi:** `train_contrastive_encoder()` trong `../../engines/pretrain_ssl/tstcc_trainer.py`.

**Cấu hình dataset được đọc từ:**
- `config/motionsense_config.py` → `MotionSenseConfig.DATA_ALL_PATH`
- `config/uci_har_config.py` → `UCIHARConfig.DATA_ALL_PATH`
- `config/hhar_config.py` → `HHARConfig`

Đường dẫn checkpoint đầu ra: `checkpoints/ssl_pretrain/contrastive/<dataset_name>/<backbone_type>/`

### 2.2. Dataset và DataLoader

**File:** `datasets/contrastive_dataset.py`
- **Class:** `ContrastiveDatasetWrapper`
- **Input:** Đường dẫn đến file `.pt` hoặc `torch.Tensor`.
- **Output:** Cặp `(x_weak, x_strong)` — mỗi phần tử có shape `(6, 128)`.
- **Không sử dụng nhãn** — SSL là unsupervised.
- Nạp từ key `"samples"` trong dict `.pt`.

**Được gọi từ:** `tstcc_trainer.py` dòng 57: `dataset = ContrastiveDatasetWrapper(data_path)`

### 2.3. Augmentation module

**File:** `models/ssl/contrastive/augmentations.py`
- **Class:** `TS_TCC_Augmentation`
- **Gọi từ:** `ContrastiveDatasetWrapper.__init__()` (mặc định nếu không truyền `aug_pipeline`)

### 2.4. Encoder backbone

**File:** `models/encoders/tstcc_encoder.py`
- **Class:** `TSTCCEncoder`
- **Được khởi tạo từ:** `build_encoder(backbone_type=backbone_type)` trong `models/encoders/builder.py`
- Khi `backbone_type="tstcc"`, `build_encoder` trả về `TSTCCEncoder`.
- Khi dùng `backbone_type="cnn_transformer"` hoặc `"standard"`, sẽ dùng encoder khác (xem Phần 6).

### 2.5. Core SSL model

**File:** `models/ssl/contrastive/ts_tcc_model.py`
- **Class:** `TSTCCModel`
- Wrapper gộp: `TSTCCEncoder` + `TC` + `NTXentLoss`.
- **Forward:** nhận `(x_weak, x_strong)` → trả về `(loss_total, loss_tc, loss_cc)`.

### 2.6. Temporal Contrasting module

**File:** `models/ssl/contrastive/tc.py`
- **Class:** `TC`
- Sử dụng `Seq_Transformer` từ `models/ssl/contrastive/attention.py`.

### 2.7. Attention / Transformer

**File:** `models/ssl/contrastive/attention.py`
- **Class:** `Seq_Transformer` — Transformer có CLS token để tạo context vector.
- **Classes hỗ trợ:** `Transformer`, `Attention`, `FeedForward`, `PreNorm`, `Residual`.
- Dùng thư viện `einops`.

### 2.8. Loss functions

**File:** `losses/nt_xent.py`
- **Class:** `NTXentLoss` — NT-Xent/InfoNCE cho Contextual Contrasting.

### 2.9. Training engine

**File:** `../../engines/pretrain_ssl/tstcc_trainer.py`
- **Function:** `train_contrastive_encoder()`
- Vòng lặp huấn luyện hoàn chỉnh + lưu checkpoint + ghi CSV log.

### 2.10. Downstream (dùng chung)

**File:** `engines/transfer/finetune_trainer.py` — `train_and_eval_finetune()` (dùng chung cho cả 3 phương pháp)  
**File:** `models/har_classifier.py` — `HARClassifier` (encoder + classifier head)  
**File:** `models/heads/classifier.py` — `ClassifierHead`

---

## Phần 3. Dữ liệu đầu vào và cấu trúc dữ liệu

### 3.1. Datasets được sử dụng

| Tên trong code | Đường dẫn file (local) | Đường dẫn file (Kaggle) |
|---|---|---|
| `uci_har` | `data/processed/uci_har/dataset_all.pt` | `/kaggle/input/.../uci_har/dataset_all.pt` |
| `motionsense` | `data/processed/motionsense/dataset_all.pt` | `/kaggle/input/.../motionsense/dataset_all.pt` |
| `hhar_phone` | `data/processed/hhar/phone/dataset_all.pt` | — |
| `hhar_watch` | `data/processed/hhar/watch/dataset_all.pt` | — |

**Dùng `dataset_all.pt`** (toàn bộ dữ liệu không chia train/val/test) cho SSL pretraining vì SSL không cần nhãn và cần nhiều dữ liệu nhất có thể.

### 3.2. Định dạng file `.pt`

Mỗi file `.pt` là dict với các key sau (xác minh từ `datasets/base_dataset.py` và `datasets/contrastive_dataset.py`):

```python
{
    "samples": torch.Tensor,   # shape (N, 6, 128) — float32
    "labels":  torch.Tensor,   # shape (N,) hoặc (N, 1) — long
    "subjects": torch.Tensor,  # shape (N,) — subject ID
}
```

> **Lưu ý:** `ContrastiveDatasetWrapper` chỉ đọc key `"samples"`, bỏ qua `"labels"` (SSL không cần nhãn).

### 3.3. Cấu trúc một mẫu

```text
x: (6, 128)
    Chiều 0 (6 kênh):
        [0] total_acc_x  hoặc userAcceleration.x
        [1] total_acc_y  hoặc userAcceleration.y
        [2] total_acc_z  hoặc userAcceleration.z
        [3] body_gyro_x  hoặc rotationRate.x
        [4] body_gyro_y  hoặc rotationRate.y
        [5] body_gyro_z  hoặc rotationRate.z
    Chiều 1 (128 timestep): 2.56 giây @ 50Hz, stride 64 (overlap 50%)
```

### 3.4. Phân chia dữ liệu (cho downstream)

**MotionSense (xác minh từ `config/motionsense_config.py`):**
- Train: subject 1–14
- Val: subject 15–18
- Test: subject 19–24

**UCI-HAR (xác minh từ `config/uci_har_config.py`):**
- Val subjects: [27, 28, 29, 30] (tách từ train gốc)
- Test: test gốc của UCI-HAR

### 3.5. Kích thước tensor qua pipeline

```text
File .pt → samples: (N, 6, 128)
                        ↓  ContrastiveDatasetWrapper.__getitem__
x_raw: (6, 128)  →  x_weak: (6, 128)  +  x_strong: (6, 128)
                        ↓  DataLoader collate
Batch: x_weak (B, 6, 128)  +  x_strong (B, 6, 128)
                        ↓  TSTCCEncoder.forward
z_weak: (B, 128, 18)  +  z_strong: (B, 128, 18)
                        ↓  TC.forward
c_t: (B, 64)           loss_tc: scalar
                        ↓  NTXentLoss
                       loss_cc: scalar
```

---

## Phần 4. Tiền xử lý và tạo cửa sổ dữ liệu

> Tiền xử lý và tạo sliding window được thực hiện trong các script `scripts/00_preprocess/` và kết quả lưu vào `data/processed/`. File tiền xử lý không thuộc scope TS-TCC nhưng output của nó là input cho TS-TCC.

### 4.1. Luồng tiền xử lý (tóm tắt)

```text
Dữ liệu thô (CSV/txt)
    → Đọc, chọn 6 cột cảm biến
    → Chuẩn hóa (chi tiết chưa xác minh từ preprocess scripts)
    → Sliding window: size=128, stride=64
    → Gán subject ID
    → Lưu thành train.pt / val.pt / test.pt / dataset_all.pt
```

**Lưu ý:** Chi tiết chuẩn hóa trong `scripts/00_preprocess/` chưa được đọc trong phân tích này. Cần kiểm tra thêm.

### 4.2. Tiền xử lý trong Dataset

`ContrastiveDatasetWrapper` chỉ ép kiểu sang float32, không có bước chuẩn hóa bổ sung:

```python
# datasets/contrastive_dataset.py, dòng 82-84
self.samples = self.samples.float()
```

**Không có:** Normalization trong-lúc-chạy, mean subtraction hay standardization trong `ContrastiveDatasetWrapper`. Nếu dữ liệu cần normalize thêm, điều này cần được thực hiện trong preprocessing script.

---

## Phần 5. Data Augmentation

### 5.1. Overview

**File:** `models/ssl/contrastive/augmentations.py`  
**Class:** `TS_TCC_Augmentation`

Augmentation được áp dụng **on-the-fly** trong `__getitem__` của dataset, mỗi lần lấy mẫu sinh ra cặp view mới ngẫu nhiên.

### 5.2. Weak Augmentation

```python
def weak_transform(self, x: np.ndarray) -> np.ndarray:
    return scaling(x, sigma=self.jitter_scale_ratio)  # sigma=1.1 mặc định
```

**Phép biến đổi `scaling`:**
- **Cơ chế:** Nhân tín hiệu với hệ số lấy từ phân phối Gaussian $\mathcal{N}(2.0, 1.1)$.
- **Lấy ngẫu nhiên theo batch/time:** `factor = np.random.normal(loc=2.0, scale=sigma, size=(B, T))` — factor shape `(B, T)`, sau đó expand thành `(B, 1, T)` để broadcast qua các kênh.
- **Ảnh hưởng:** Biến đổi biên độ nhưng giữ nguyên cấu trúc thời gian.
- **Tham số:** `jitter_scale_ratio=1.1` (mặc định).

> **Quan sát cần chú ý:** `loc=2.0` nghĩa là biên độ trung bình bị nhân đôi. Đây là thiết kế theo bài báo gốc TS-TCC, không phải lỗi. Tuy nhiên, nếu dữ liệu đã được normalize về range [-1, 1] thì việc nhân 2 sẽ đưa tín hiệu ra ngoài range đó.

### 5.3. Strong Augmentation

```python
def strong_transform(self, x: np.ndarray) -> np.ndarray:
    x_perm = permutation(x, max_segments=self.max_seg)  # max_seg=5
    x_strong = jitter(x_perm, sigma=self.jitter_ratio)  # sigma=0.8
    return x_strong
```

**Bước 1 — `permutation`:**
- **Cơ chế:** Cắt chuỗi thời gian thành `num_segs ∈ [1, max_segments]` đoạn ngẫu nhiên, sau đó xáo trộn thứ tự các đoạn.
- `seg_mode="random"`: Chọn điểm chia ngẫu nhiên.
- Áp dụng đồng bộ cho **tất cả các kênh** (dùng cùng bộ chỉ số `warp`).
- **Ảnh hưởng:** Phá vỡ cấu trúc thời gian nhưng giữ các pattern cục bộ.

**Bước 2 — `jitter`:**
- **Cơ chế:** Thêm nhiễu Gaussian $\mathcal{N}(0, 0.8)$ vào tất cả các điểm.
- **Tham số:** `jitter_ratio=0.8` (mặc định).

> **Quan sát:** Với `sigma=0.8`, nhiễu Gaussian có độ lệch chuẩn khá lớn. Kết hợp với việc dữ liệu có thể đã normalized về range nhỏ, nhiễu này có thể che lấp nhiều thông tin. Cần kiểm tra scale của dữ liệu thực tế sau preprocessing.

### 5.4. Xử lý chiều tensor trong augmentation

`TS_TCC_Augmentation.__call__()` có logic phát hiện và đảo chiều nếu nhận được tensor dạng `(T, C)` thay vì `(C, T)`:

```python
# augmentations.py, dòng 111-112
if x_numpy.ndim == 2 and x_numpy.shape[0] == 128 and x_numpy.shape[1] == 6:
    x_numpy = x_numpy.T
```

Điều này xử lý trường hợp data có layout `(128, 6)` thay vì `(6, 128)`.

### 5.5. Tổng kết augmentation pipeline

```text
x_raw: (6, 128) [numpy]
    |
    +─── weak_transform  → scaling(sigma=1.1)     → x_weak: (6, 128) [numpy] → [torch.Tensor float32]
    |
    +─── strong_transform → permutation(max_seg=5) → jitter(sigma=0.8) → x_strong: (6, 128) [numpy] → [torch.Tensor float32]
```

---

## Phần 6. Kiến trúc mô hình

### 6.1. Encoder backbone — TSTCCEncoder

**File:** `models/encoders/tstcc_encoder.py`  
**Class:** `TSTCCEncoder`  
**Nguồn gốc:** Reproduction 100% của `base_Model` trong repo gốc `emadeldeen24/TS-TCC`.

```text
Input: (B, 6, 128)
    ↓ Block 1: Conv1d(6→32, k=8, p=4) → BN → ReLU → MaxPool1d(k=2,s=2,p=1) → Dropout(0.35)
    → (B, 32, 65)
    ↓ Block 2: Conv1d(32→64, k=8, p=4) → BN → ReLU → MaxPool1d(k=2,s=2,p=1)
    → (B, 64, 34)
    ↓ Block 3: Conv1d(64→128, k=8, p=4) → BN → ReLU → MaxPool1d(k=2,s=2,p=1)
    → (B, 128, 18)
Output: (B, 128, 18)
```

**Tính toán kích thước (xác minh từ docstring trong file):**
- Block 1: `L_out = floor((128 + 2*4 - 8)/1) + 1 = 129`, sau MaxPool(k=2,s=2,p=1): `floor((129+2-2)/2)+1 = 65`
- Block 2: `floor((65+8-8)/1)+1 = 66`, sau MaxPool: `floor((66+2-2)/2)+1 = 34`
- Block 3: `floor((34+8-8)/1)+1 = 35`, sau MaxPool: `floor((35+2-2)/2)+1 = 18`

> **Lưu ý thiết kế:** Encoder **không có Global Average Pooling** ở cuối — output là `(B, 128, 18)`, không phải `(B, 128)`. Điều này quan trọng cho TC module (cần sequence dimension).

> **Các backbone khác:** Nếu dùng `backbone_type="standard"` (StandardSensorEncoder1D), output là `(B, 128, 8)`. Nếu dùng `cnn_transformer`, output là `(B, 128, 32)`. Tuy nhiên `TSTCCModel` mặc định dùng `TSTCCEncoder` (output 18 timestep). Việc kết hợp backbone khác với `TC` module cần kiểm tra khả năng tương thích.

### 6.2. Temporal Contrasting module — TC

**File:** `models/ssl/contrastive/tc.py`  
**Class:** `TC`

```text
Input: z_aug1: (B, 128, 18), z_aug2: (B, 128, 18), t_samples: int

Bước 1: Transpose → (B, 18, 128) [= (batch, seq_len, channels)]
Bước 2: Lấy target samples từ z_aug2: encode_samples[k] = z_aug2[:, t_samples+k+1, :] → (K, B, 128)
Bước 3: forward_seq = z_aug1[:, :t_samples+1, :] → (B, t_samples+1, 128)
Bước 4: c_t = seq_transformer(forward_seq) → (B, hidden_dim=64)
Bước 5: pred[k] = Wk[k](c_t) → (K, B, 128)
Bước 6: InfoNCE = Σ_k diag(LogSoftmax(encode_samples[k] @ pred[k].T)) / (-B * K)
Output: nce_loss, projection_head(c_t), t_samples
```

**Projection head trong TC:**
```python
self.projection_head = nn.Sequential(
    nn.Linear(hidden_dim=64, feature_dim//2=64),  # 64 → 64
    nn.BatchNorm1d(64),
    nn.ReLU(inplace=True),
    nn.Linear(64, feature_dim//4=32),            # 64 → 32
)
```

Output của projection_head: `(B, 32)` — đây là `context_w` và `context_s` dùng cho NT-Xent.

### 6.3. Seq_Transformer

**File:** `models/ssl/contrastive/attention.py`  
**Class:** `Seq_Transformer`

```text
Input: forward_seq: (B, t+1, 128)
    ↓ patch_to_embedding: Linear(128, hidden_dim=64) → (B, t+1, 64)
    ↓ Prepend c_token: (B, t+2, 64)  [c_token là learnable parameter (1, 1, 64)]
    ↓ Transformer (depth=4, heads=4, mlp_dim=64) → (B, t+2, 64)
    ↓ Lấy vị trí 0 (CLS token): x[:, 0] → (B, 64)
Output: c_t: (B, 64)
```

### 6.4. NTXentLoss — Contextual Contrasting

**File:** `losses/nt_xent.py`  
**Class:** `NTXentLoss(temperature=0.2)`

```text
Input: h_weak: (B, 32), h_strong: (B, 32)
    ↓ L2 normalize mỗi vector
    ↓ Concat: representations = [h_weak; h_strong] → (2B, 32)
    ↓ Similarity matrix = representations @ representations.T / temperature → (2B, 2B)
    ↓ Mask self-similarity (diagonal)
    ↓ Positive pairs: index B..2B-1 là positive của 0..B-1 và ngược lại
    ↓ CrossEntropyLoss
Output: loss_cc: scalar
```

### 6.5. TSTCCModel — Wrapper tổng hợp

**File:** `models/ssl/contrastive/ts_tcc_model.py`  
**Class:** `TSTCCModel`

```text
Input: x_weak: (B, 6, 128), x_strong: (B, 6, 128)

1. Encoder: x_weak → z_weak (B, 128, 18), x_strong → z_strong (B, 128, 18)
2. Chọn t_samples ngẫu nhiên ∈ [0, 18 - timesteps - 1] = [0, 14]
3. TC chiều 1 (Weak→Strong): TC(z_weak, z_strong, t_samples) → loss_tc1, context_w, _
4. TC chiều 2 (Strong→Weak): TC(z_strong, z_weak, t_samples) → loss_tc2, context_s, _
5. loss_tc = loss_tc1 + loss_tc2
6. NT-Xent(context_w, context_s) → loss_cc
7. loss_total = 1.0 * loss_tc + 0.7 * loss_cc
```

**Sơ đồ luồng:**

```
x_weak (B,6,128)  ──→  TSTCCEncoder  ──→  z_weak (B,128,18)  ──┐
                                                                  ├──→ TC (Weak→Strong) ──→ loss_tc1, context_w
x_strong (B,6,128) ──→  TSTCCEncoder  ──→  z_strong (B,128,18) ─┘
                                                                  ├──→ TC (Strong→Weak) ──→ loss_tc2, context_s
                                                              ────┘
                                                context_w, context_s
                                                      ↓
                                                  NTXentLoss  ──→  loss_cc
                                                      ↓
                                    loss_total = 1.0*loss_tc + 0.7*loss_cc
```

---

## Phần 7. Quy trình Self-Supervised Pretraining

### 7.1. Khởi tạo (từ `tstcc_trainer.py`)

```python
# Encoder backbone
encoder_backbone = build_encoder(backbone_type=backbone_type, in_channels=6)

# SSL Model
ssl_model = TSTCCModel(
    encoder=encoder_backbone,
    in_channels=6,
    feature_dim=128,
    timesteps=3,      # K bước tương lai dự đoán
    lambda1=1.0,      # Trọng số L_TC
    lambda2=0.7,      # Trọng số L_CC
    temperature=0.2   # Nhiệt độ NT-Xent
)
```

### 7.2. Optimizer

```python
optimizer = AdamW(
    ssl_model.parameters(),  # TẤT CẢ tham số: encoder + TC + NTXent
    lr=3e-4,                 # Default learning rate
    weight_decay=3e-4
)
```

> **Lưu ý:** Chỉ có **một optimizer** cho toàn bộ mô hình. Không có học phân lớp (layerwise LR) trong pretraining.

### 7.3. Một training iteration (xác minh từ `tstcc_trainer.py`, dòng 126–140)

```python
for x_w, x_s in loader:      # x_w, x_s: (B, 6, 128)
    x_w = x_w.to(device)
    x_s = x_s.to(device)

    optimizer.zero_grad()
    loss, loss_tc, loss_cc = ssl_model(x_w, x_s)   # Forward pass
    loss.backward()                                  # Backward pass

    gnorm = nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=2.0)  # Gradient clipping
    optimizer.step()
```

**Gradient clipping:** max_norm=2.0, áp dụng cho toàn bộ ssl_model.parameters().

### 7.4. Điều kiện lưu checkpoint

```python
# contrastive_trainer.py, dòng 155-157
if avg_loss < best_loss:
    best_loss = avg_loss
    torch.save(ssl_model.encoder.state_dict(), checkpoint_path)
```

**Chỉ lưu `encoder.state_dict()`**, không lưu TC module hay NTXentLoss.

> **Quan sát quan trọng:** Checkpoint được lưu dựa trên `avg_loss` của **training set**, không có validation. Đây là khác biệt với CrossHAR trainer (có validation).

### 7.5. Loss functions chi tiết

#### 7.5.1. InfoNCE Loss (Temporal Contrasting)

**Công thức (từ `tc.py`, dòng 111–119):**

$$\mathcal{L}_{TC} = -\frac{1}{B \cdot K} \sum_{k=1}^{K} \sum_{i=1}^{B} \log \frac{\exp(\text{enc}_k^{(i)} \cdot \text{pred}_k^{(i)})}{\sum_{j=1}^{B} \exp(\text{enc}_k^{(i)} \cdot \text{pred}_k^{(j)})}$$

Trong đó:
- $K=3$: số bước tương lai (timesteps)
- $B$: batch size
- $\text{enc}_k^{(i)} = z_{\text{aug2}}[i, t\_samples+k, :]$: đặc trưng thực tế tại bước $t+k$ của view 2
- $\text{pred}_k^{(i)} = W_k(c_t^{(i)})$: dự đoán của mẫu $i$ tại bước $k$
- Ma trận similarity: `torch.mm(encode_samples[i], pred[i].T)` — shape `(B, B)`
- Loss được cộng dồn qua K bước và chia cho `B * K`

**Implementation (từ tc.py, dòng 111–119):**
```python
nce = 0.0
for i in range(self.timestep):
    total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
    nce += torch.sum(torch.diag(self.lsoftmax(total)))
nce /= -1.0 * batch_size * self.timestep
```

#### 7.5.2. NT-Xent Loss (Contextual Contrasting)

**Công thức:**

$$\mathcal{L}_{CC} = -\frac{1}{2B} \sum_{i=1}^{2B} \log \frac{\exp(\text{sim}(h_i, h_{i+})/\tau)}{\sum_{j \neq i} \exp(\text{sim}(h_i, h_j)/\tau)}$$

Trong đó $\tau = 0.2$ (temperature), $h_{i+}$ là positive pair (view còn lại của cùng mẫu).

**Implementation (từ nt_xent.py):**
- L2 normalize cả hai view
- Tạo similarity matrix (2B, 2B)
- Mask self-similarity
- CrossEntropyLoss với targets là positive pair indices

#### 7.5.3. Loss tổng

$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{TC} + \lambda_2 \mathcal{L}_{CC} = 1.0 \cdot \mathcal{L}_{TC} + 0.7 \cdot \mathcal{L}_{CC}$$

Xác minh từ `ts_tcc_model.py`, dòng 147: `loss_total = self.lambda1 * loss_tc + self.lambda2 * loss_cc`

Và `loss_tc = loss_tc1 + loss_tc2` (hai chiều Weak→Strong và Strong→Weak).

### 7.6. Outputs sau pretraining

**Checkpoint:**
```
checkpoints/ssl_pretrain/contrastive/<dataset_name>/<backbone_type>/
    ├── tstcc_<backbone>_encoder_pretrained_<dataset>.pt  ← encoder.state_dict()
    ├── loss_history.csv                                  ← epoch, loss_total, loss_tc, loss_cc, grad_norm
    ├── loss_curve.png                                    ← biểu đồ hội tụ
    ├── config.json                                       ← cấu hình huấn luyện
    ├── complexity.json                                   ← FLOPs, params
    └── pretrain_summary.json
```

---

## Phần 8. Chi tiết riêng của TS-TCC

### 8.1. Bidirectional Temporal Contrasting

Một đặc điểm quan trọng trong implementation là TC được chạy **hai chiều** với cùng `t_samples`:

```python
# ts_tcc_model.py, dòng 132–137
loss_tc1, context_w, _ = self.tc(z_weak, z_strong, t_samples=t_samples)  # Weak predict Strong
loss_tc2, context_s, _ = self.tc(z_strong, z_weak, t_samples=t_samples)  # Strong predict Weak
loss_tc = loss_tc1 + loss_tc2
```

Trong bài báo gốc TS-TCC, TC cũng bidirectional (xem docstring `ts_tcc_model.py` dòng 12-14). Implementation này nhất quán với bài báo.

### 8.2. t_samples được chia sẻ giữa hai chiều

`t_samples` là một scalar int được lấy ngẫu nhiên một lần và dùng chung cho cả hai chiều TC:

```python
# ts_tcc_model.py, dòng 124–126
seq_len = z_weak.shape[2]  # = 18 với TSTCCEncoder
t_samples = torch.randint(0, seq_len - self.timesteps, size=(1,)).item()
# t_samples ∈ [0, 14] vì timesteps=3 và seq_len=18
```

### 8.3. Context vector cho NT-Xent

`context_w` và `context_s` được lấy từ kết quả của TC:
- `context_w = TC(z_weak, z_strong).projection_head(c_t)` — shape `(B, 32)`
- `context_s = TC(z_strong, z_weak).projection_head(c_t)` — shape `(B, 32)`

Đây là projection của CLS token từ Seq_Transformer qua một MLP nhỏ (64→64→32).

### 8.4. Lưu ý về backbone compatibility

**Vấn đề tiềm ẩn (Cần kiểm chứng thêm):**

`TSTCCModel` nhận `encoder` từ bên ngoài. Nếu encoder là `StandardSensorEncoder1D`, output sẽ là `(B, 128, 8)` thay vì `(B, 128, 18)`. TC module sẽ hoạt động với `seq_len=8`, và `t_samples ∈ [0, 8-3-1] = [0, 4]`. Điều này về nguyên tắc vẫn chạy được nhưng có ít timestep hơn để học temporal relations.

**Đã xác minh:** Khi `backbone_type="tstcc"`, `build_encoder` trả về `TSTCCEncoder` (output `(B,128,18)`) — phù hợp nhất cho TC module theo thiết kế gốc.

---

## Phần 9. Downstream Task: Linear Probing và Fine-tuning

### 9.1. Downstream model

**File:** `models/har_classifier.py`  
**Class:** `HARClassifier`

```text
HARClassifier:
    encoder: encoder backbone (được load từ checkpoint)
    classifier: ClassifierHead(feature_dim=128, num_classes=5, dropout=0.2)
```

**ClassifierHead (`models/heads/classifier.py`):**
```python
x = dropout(x)
if x.dim() == 3:  # Nếu encoder trả về (B, D, L), lấy mean theo L
    x = torch.mean(x, dim=-1)
logits = fc(x)  # Linear(128, num_classes)
```

> **Quan sát:** `ClassifierHead` tự động xử lý cả input 2D `(B, D)` và 3D `(B, D, L)`. Với `TSTCCEncoder`, output encoder là `(B, 128, 18)` → mean pooling → `(B, 128)` → `fc` → `(B, 5)`.

### 9.2. Linear Probing

**File:** `engines/transfer/finetune_trainer.py`  
**Function:** `train_and_eval_finetune(freeze_backbone=True)`

```python
# Đóng băng encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Chỉ optimize classifier head
optimizer = Adam(model.classifier.parameters(), lr=head_lr, weight_decay=weight_decay)
```

- `head_lr = 1e-3` (mặc định)
- Scheduler: `ReduceLROnPlateau(mode="max", factor=0.5, patience=5)` theo val Macro F1
- Criterion: `nn.CrossEntropyLoss()`

### 9.3. Full Fine-tuning

```python
# Mở khóa encoder
for param in model.encoder.parameters():
    param.requires_grad = True

# Layer-wise learning rate
optimizer = Adam([
    {"params": model.encoder.parameters(), "lr": backbone_lr},  # backbone_lr=1e-4
    {"params": model.classifier.parameters(), "lr": head_lr},   # head_lr=1e-3
])
```

### 9.4. Tải checkpoint

```python
# finetune_trainer.py, dòng 101–113
checkpoint = torch.load(encoder_ckpt_path, map_location=target_device, weights_only=True)

# Hỗ trợ nhiều định dạng:
if isinstance(checkpoint, dict):
    if "encoder_state_dict" in checkpoint: encoder_dict = checkpoint["encoder_state_dict"]
    elif "encoder" in checkpoint: encoder_dict = checkpoint["encoder"]
    else: encoder_dict = checkpoint
else:
    encoder_dict = checkpoint

model.encoder.load_state_dict(encoder_dict, strict=True)
```

### 9.5. Best model selection

```python
# Lưu best model dựa trên val Macro F1
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Sau training, load lại best state trước khi evaluate trên test set.

### 9.6. Few-shot K-shot sampling

**File:** `scripts/03_cross_domain/run_few_shot.py`

```python
def sample_k_shot(X, y, k, seed):
    rng = np.random.default_rng(seed)
    for c in range(NUM_COMMON_CLASSES):  # 5 classes
        c_idx = torch.where(y == c)[0].numpy()
        rng.shuffle(c_idx)
        idx_list.extend(c_idx[:k])
    return X[idx_list], y[idx_list]
```

- Lấy đúng **k mẫu mỗi lớp** → K*5 mẫu train tổng cộng.
- Cố định bởi `seed`.
- Val và Test sử dụng **100% dữ liệu** (không K-shot).

---

## Phần 10. Cross-Domain Training và Evaluation

### 10.1. Kịch bản

Source domain: pretraining trên UCI-HAR hoặc MotionSense  
Target domain: đánh giá trên UCI-HAR, MotionSense, HHAR (phone/watch)

Cụ thể: Encoder pretrained trên source, được load vào `HARClassifier`, fine-tune/LP trên K mẫu của target train, đánh giá trên **toàn bộ** target test set.

### 10.2. Checkpoint convention

```
checkpoints/ssl_pretrain/contrastive/<src_dataset>/<backbone>/tstcc_<backbone>_encoder_pretrained_<src_dataset>.pt
```

Ví dụ: `checkpoints/ssl_pretrain/contrastive/motionsense/standard/tstcc_standard_encoder_pretrained_motionsense.pt`

### 10.3. Xử lý label space

Cả source và target đều sử dụng **5 lớp chung** (label 0–4: Walking, Upstairs, Downstairs, Sitting, Standing). Nhãn `Laying` (UCI-HAR) và `Jogging` (MotionSense) bị lọc bỏ:

```python
# run_few_shot.py, dòng 58
mask = (d["labels"].squeeze() >= 0) & (d["labels"].squeeze() < 5)
```

### 10.4. Phân loại thực nghiệm

Đây là **Transfer Learning**, không phải Domain Adaptation:
- Không có mechanism align feature distribution giữa source và target.
- Không có batch normalization với domain-specific statistics.
- Không có adversarial training hay gradient reversal.
- Chỉ pretrain trên source, fine-tune trên target.

---

## Phần 11. Evaluation Metrics và Kết quả đầu ra

### 11.1. Metrics được tính (xác minh từ code)

| Metric | Function | Thư viện |
|---|---|---|
| Accuracy | `accuracy_score(y_true, y_pred)` | sklearn |
| Macro F1 | `f1_score(..., average="macro")` | sklearn |
| Per-class F1 | `classification_report(..., output_dict=True)` | sklearn |
| Cohen's Kappa | `cohen_kappa_score(y_true, y_pred)` | sklearn |
| Confusion Matrix | `confusion_matrix(y_true, y_pred)` | sklearn |

**Metric chính:** Macro F1 (%).

### 11.2. Đầu ra

```text
outputs_evaluation/cross_k_shot/contrastive/<backbone>/<src>_to_<tgt>/
    ├── experiment_metadata.json
    └── <k>_shot/
        ├── linear_probing/
        │   ├── summary_linear_probing_<k>_shot.json   ← mean ± std qua seeds
        │   ├── detailed_all_seeds.json                 ← chi tiết từng seed
        │   └── aggregated_confusion_matrix.png         ← CM trung bình qua seeds
        └── full_finetuning/
            ├── summary_full_finetuning_<k>_shot.json
            ├── detailed_all_seeds.json
            └── aggregated_confusion_matrix.png
```

---

## Phần 12. Checkpoint và Khả năng Tái sử dụng

### 12.1. Cấu trúc checkpoint pretrain

```python
# Chỉ lưu encoder, không lưu TC hay NTXentLoss
torch.save(ssl_model.encoder.state_dict(), checkpoint_path)
```

Key trong state_dict phụ thuộc vào loại encoder:
- `TSTCCEncoder`: `conv_block1.0.weight`, `conv_block1.1.weight`, ... `conv_block3.4.weight`
- `StandardSensorEncoder1D`: `block1.0.weight`, ... `block4.3.weight`

### 12.2. Tải checkpoint (downstream)

```python
model.encoder.load_state_dict(encoder_dict, strict=True)
```

Dùng `strict=True` — nếu key không khớp (ví dụ checkpoint `tstcc` encoder nhưng model dùng `standard` encoder), sẽ raise exception.

### 12.3. Vấn đề tương thích checkpoint

> **Cần kiểm chứng:** Script `run_pretrain_tstcc.py` tên checkpoint là `tstcc_<backbone>_encoder_pretrained_<dataset>.pt`. Backbone có thể là `vit_1d`, `standard`, hay `cnn_transformer`. Script `run_few_shot.py` load từ `checkpoints/ssl_pretrain/contrastive/<src>/<backbone>/`. Cần đảm bảo `--backbone` arg khi pretrain và khi run_few_shot phải khớp nhau.

---

## Phần 13. Cài đặt và Chạy Code

### 13.1. Yêu cầu môi trường

```
Python >= 3.11
torch >= 2.0.0
torchvision
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scipy >= 1.7.0
tqdm
einops  # Cần cho attention.py (TS-TCC) — KHÔNG có trong requirements.txt!
```

> **Vấn đề tiềm ẩn:** `einops` được dùng trong `models/ssl/contrastive/attention.py` nhưng **không xuất hiện trong `requirements.txt`**. Cần cài thêm: `pip install einops`.

### 13.2. Cài đặt

```bash
git clone https://github.com/Dungx1107/CrossDomain-HAR-SSL.git
cd CrossDomain-HAR-SSL
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install einops  # Bổ sung thiếu
```

### 13.3. Chuẩn bị dữ liệu

```bash
# Đặt dữ liệu thô vào data/raw/ theo cấu trúc config
python scripts/00_preprocess/preprocess_uci_har.py
python scripts/00_preprocess/preprocess_motionsense.py
```

### 13.4. Chạy SSL Pretraining

```bash
# Sử dụng TS-TCC backbone
python scripts/02_pretrain_ssl/run_pretrain_contrastive.py \
  --backbone tstcc \
  --datasets uci_har motionsense \
  --epochs 60 \
  --batch_size 64
```

> **Lệnh này suy ra từ code và README.** Default backbone trong script là `vit_1d`. README ví dụ dùng `cnn_transformer`. Backbone `tstcc` là lựa chọn phù hợp nhất cho `TSTCCModel`.

### 13.5. Chạy K-shot cross-domain

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method contrastive \
  --backbone tstcc \
  --k_shots 10 20 50 100 \
  --seeds 42 100 2024 7 99 \
  --epochs 40 \
  --batch_size 16 \
  --pairs motionsense:uci_har uci_har:motionsense
```

---

## Phần 14. Hyperparameters và Cấu hình Thực nghiệm

| Hyperparameter | Giá trị mặc định | Vị trí khai báo | Ý nghĩa |
|---|---|---|---|
| Backbone | `vit_1d` (script) / `tstcc` (model default) | `run_pretrain_tstcc.py` | Loại encoder |
| Batch size | 64 | `run_pretrain_tstcc.py` | Số mẫu/batch |
| Learning rate | 3e-4 | `tstcc_trainer.py` | Không có arg override |
| Weight decay | 3e-4 | `tstcc_trainer.py` | Không có arg override |
| Epochs | 60 | `run_pretrain_tstcc.py` | Số epoch pretrain |
| Optimizer | AdamW | `tstcc_trainer.py` | — |
| Grad clip | 2.0 | `tstcc_trainer.py` | max_norm |
| Timesteps (TC) | 3 | `TSTCCModel.__init__` | K bước dự đoán |
| Lambda1 (L_TC) | 1.0 | `TSTCCModel.__init__` | Trọng số TC |
| Lambda2 (L_CC) | 0.7 | `TSTCCModel.__init__` | Trọng số CC |
| Temperature (NT-Xent) | 0.2 | `TSTCCModel.__init__` | Nhiệt độ contrastive |
| TC hidden_dim | 64 | `TC.__init__` | Chiều Transformer |
| TC depth | 4 | `TC.__init__` | Số layer Transformer |
| TC heads | 4 | `TC.__init__` | Số attention heads |
| feature_dim | 128 | `TSTCCEncoder` | Chiều encoder output |
| Window size | 128 | Config files | Timestep/window |
| Stride | 64 | Config files | Overlap 50% |
| K-shot (downstream) | [10, 20, 50, 100] | `run_few_shot.py` | Số mẫu/lớp |
| Seeds (downstream) | [42, 100, 2024, 7, 99] | `run_few_shot.py` | 5 seeds |
| Backbone LR (FT) | 1e-4 | `finetune_trainer.py` | LR encoder khi FT |
| Head LR | 1e-3 | `finetune_trainer.py` | LR classifier head |
| FT Epochs | 40 | `run_few_shot.py` | — |

---

## Phần 15. Checklist Kiểm tra Tính Đúng đắn

### 15.1. Data Leakage

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| SSL pretraining có dùng nhãn không | ✅ Đã xác minh | `ContrastiveDatasetWrapper` chỉ đọc `samples`, bỏ qua `labels` |
| Chia dữ liệu theo subject | ✅ Đã xác minh | MotionSense: train/val/test theo subject trong config |
| Val/Test không K-shot | ✅ Đã xác minh | `run_few_shot.py` dùng 100% val và test |
| Source/target separation | ✅ Đã xác minh | Pretrain trên source `dataset_all.pt`, fine-tune trên target train.pt |

### 15.2. Loss Function

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| InfoNCE bidirectional | ✅ Đã xác minh | Chạy TC hai chiều (Weak→Strong và Strong→Weak) |
| Positive pair trong NT-Xent | ✅ Đã xác minh | View từ cùng sample là positive |
| Detach gradient ở target | ❓ Cần kiểm chứng | `encode_samples` trong TC không có `.detach()` — target được dùng cho loss mà không detach. Đây là InfoNCE tiêu chuẩn (không như BYOL/MoCo) nên không cần detach, nhưng cần xác nhận. |
| Temperature scaling | ✅ Đã xác minh | `temperature=0.2` cho NT-Xent; InfoNCE không có temperature scaling riêng |

### 15.3. Kiến trúc

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| seq_len sau encoder | ✅ Đã xác minh | TSTCCEncoder: (B,6,128) → (B,128,18). seq_len=18 |
| t_samples range | ✅ Đã xác minh | t_samples ∈ [0, 14] với seq_len=18, timesteps=3 |
| projection_head output dim | ✅ Đã xác minh | TC.projection_head → (B, 32); NT-Xent nhận (B, 32) |
| Encoder output dim | ✅ Đã xác minh | feature_dim=128, T=18 |

### 15.4. Downstream

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| Encoder frozen trong LP | ✅ Đã xác minh | `param.requires_grad = False` cho encoder |
| Strict loading checkpoint | ✅ Đã xác minh | `strict=True` trong `load_state_dict` |
| Best model selection | ✅ Đã xác minh | Dựa trên val Macro F1 |
| Metric đúng | ✅ Đã xác minh | `f1_score(average="macro")` |

### 15.5. Vấn đề tiềm ẩn cần kiểm tra

**1. `einops` thiếu trong requirements.txt**
- File: `models/ssl/contrastive/attention.py`, dòng 14: `from einops import rearrange, repeat`
- Hậu quả: `ImportError` khi chạy TS-TCC nếu chưa cài `einops`
- Cách kiểm tra: `pip show einops`

**2. `scaling` augmentation với `loc=2.0`**
- File: `augmentations.py`, dòng 32: `factor = np.random.normal(loc=2.0, scale=sigma, size=(B, T))`
- Vấn đề: Nếu dữ liệu đã normalized về mean ≈ 0, việc nhân với factor mean=2.0 sẽ biến đổi mạnh biên độ. Cần kiểm tra scale dữ liệu sau preprocessing.
- Trạng thái: **Cần kiểm chứng thêm**

**3. Default backbone `vit_1d` trong run_pretrain_contrastive.py**
- File: `../../scripts/02_pretrain_ssl/run_pretrain_tstcc.py`, dòng 33: `default="vit_1d"`
- Nhưng `TSTCCModel` được thiết kế với `TSTCCEncoder` (output 18 timestep). Nếu dùng `vit_1d`, seq_len sẽ khác.
- Trạng thái: **Cần kiểm chứng** — xem `vit_1d` encoder output shape để tính t_samples range.

**4. Checkpoint selection không có validation**
- File: `tstcc_trainer.py`, dòng 155
- Checkpoint được lưu theo training loss, không có validation set
- Khác biệt với CrossHAR (có val loss)
- Trạng thái: **Phát hiện cần chú ý** — không nhất thiết là lỗi nhưng có thể dẫn đến overfitting checkpoint

---

## Phần 16. Tài liệu Tham khảo

**Bài báo gốc:**
- Eldele, E., et al. "Time-Series Representation Learning via Temporal and Contextual Contrasting." IJCAI 2021.
- Repository gốc: `emadeldeen24/TS-TCC`

**Tài liệu từ repository (đã xác minh):**
- Docstring `models/ssl/contrastive/ts_tcc_model.py` — mô tả rõ nguồn gốc từ repo gốc
- Docstring `models/encoders/tstcc_encoder.py` — "EXACT 100% REPRODUCTION" của `base_Model`
- Docstring `models/ssl/contrastive/tc.py` — "Mã nguồn gốc từ file `models/TC.py` của repo `emadeldeen24/TS-TCC`"
- Docstring `models/ssl/contrastive/attention.py` — "Mã nguồn gốc từ file `models/attention.py` của repo `emadeldeen24/TS-TCC`"
