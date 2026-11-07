# Tài liệu Kỹ thuật: TS-TCC (Temporal and Contextual Contrasting)

> **Mục đích tài liệu:** Mô tả chính xác implementation của phương pháp TS-TCC trong repository CrossDomain-HAR-SSL dựa trên phân tích mã nguồn thực tế.
> **Phiên bản phân tích:** Cập nhật theo snapshot code hiện tại (November 2026).
> **Tài liệu này dành cho:** AI review code — mọi thông tin đều dẫn chiếu trực tiếp đến file và function tương ứng.

---

## Phần 1. Tổng quan

### 1.1. Mục tiêu trong dự án

TS-TCC (Temporal and Contextual Contrasting) là một trong các phương pháp SSL được triển khai trong dự án. Mục tiêu là học biểu diễn đặc trưng từ chuỗi thời gian cảm biến quán tính (6 kênh, 128 timestep) **không cần nhãn hoạt động**.

Encoder được pretrain trên source domain (MotionSense, UCI-HAR, HHAR), sau đó chuyển sang target domain qua hai protocol:
- **Linear Probing (LP):** Đóng băng encoder, chỉ huấn luyện classifier head.
- **Full Fine-tuning (FT):** Mở khóa toàn bộ mạng, fine-tune encoder + head.

### 1.2. Lý thuyết gốc

**Lý thuyết (Eldele et al., IJCAI 2021 — emadeldeen24/TS-TCC):**
- Tạo hai view augmented từ cùng một chuỗi thời gian: **weak view** và **strong view**.
- **Temporal Contrasting (TC):** Dùng Seq_Transformer trích xuất context vector từ quá khứ, dự đoán đặc trưng tương lai của view kia (InfoNCE loss dạng CPC).
- **Contextual Contrasting (CC):** Kéo context vector của hai view gần nhau (NT-Xent loss).
- Loss tổng: $\mathcal{L} = \lambda_1 \mathcal{L}_{TC} + \lambda_2 \mathcal{L}_{CC}$, với $\lambda_1=1.0, \lambda_2=0.7$.

### 1.3. Vai trò trong pipeline tổng thể

```text
SSL Pretraining (source domain)         → checkpoints/ssl_pretrain/tstcc/<source>/<backbone>/
        |
        ↓  encoder.state_dict() được lưu
K-shot / Fraction Cross-Domain          → scripts/03_cross_domain/run_few_shot.py
        |
        ↓  load encoder, freeze hoặc finetune
Linear Probing / Full Fine-tuning       → engines/transfer/finetune_trainer.py
        |
        ↓
Đánh giá trên target test set           → outputs_evaluation/cross_k_shot/tstcc/
```

> **Lưu ý thay đổi quan trọng:** Checkpoint lưu tại `tstcc/` (không phải `contrastive/` như phiên bản cũ). Quy ước này thống nhất với `utils/checkpoint_paths.py`.

### 1.4. Các thành phần code chính

| Thành phần | File | Mô tả |
|---|---|---|
| Augmentation | `../../models/ssl/tstcc/augmentations.py` | Tạo weak/strong view |
| TSTCCEncoder | `models/encoders/tstcc_encoder.py` | Backbone 1D-CNN (3 block) |
| TC module | `../../models/ssl/tstcc/tc.py` | Temporal Contrasting, InfoNCE |
| Seq_Transformer | `../../models/ssl/tstcc/attention.py` | Transformer trích xuất c_t |
| TSTCCModel | `../../models/ssl/tstcc/ts_tcc_model.py` | Wrapper gộp encoder + TC + CC |
| NTXentLoss | `losses/nt_xent.py` | Contextual Contrasting loss |
| ContrastiveDatasetWrapper | `datasets/contrastive_dataset.py` | Dataset sinh (x_weak, x_strong) |
| TSTCCConfig | `config/methods/tstcc_config.py` | Tập trung hyperparameters |
| Pretrain Engine | `engines/pretrain_ssl/tstcc/tstcc_trainer.py` | Hàm `train_tstcc_encoder()` |
| Reporting | `engines/pretrain_ssl/tstcc/tstcc_reporting.py` | Log CSV, biểu đồ, complexity |
| Entry point | `scripts/02_pretrain_ssl/run_pretrain_tstcc.py` | CLI script khởi chạy |
| Checkpoint paths | `utils/checkpoint_paths.py` | Helper đường dẫn checkpoint |

---

## Phần 2. Cấu trúc mã nguồn liên quan

### 2.1. Cấu trúc thư mục (các file liên quan TS-TCC)

```
CrossDomain-HAR-SSL/
├── config/
│   └── methods/
│       └── tstcc_config.py              ← Tất cả hyperparameters TS-TCC
├── datasets/
│   └── contrastive_dataset.py           ← ContrastiveDatasetWrapper (dùng chung với SwAV)
├── engines/
│   └── pretrain_ssl/
│       └── tstcc/
│           ├── tstcc_trainer.py         ← train_tstcc_encoder(), _build_loader(), _train_one_epoch()
│           └── tstcc_reporting.py       ← Logging, plotting, complexity
├── losses/
│   └── nt_xent.py                       ← NTXentLoss
├── models/
│   ├── encoders/
│   │   ├── builder.py                   ← build_encoder() factory
│   │   ├── tstcc_encoder.py             ← TSTCCEncoder backbone
│   │   ├── cnn1d.py                     ← StandardSensorEncoder1D
│   │   ├── cnn_transformer.py           ← CNNTransformerEncoder
│   │   └── vit_1d.py                    ← ViT1DEncoder
│   ├── heads/
│   │   └── classifier.py               ← ClassifierHead (downstream)
│   └── ssl/
│       └── contrastive/
│           ├── attention.py             ← Seq_Transformer, Transformer, Attention, FeedForward
│           ├── augmentations.py         ← TS_TCC_Augmentation (scaling, permutation, jitter)
│           ├── tc.py                    ← TC class (InfoNCE + projection head)
│           └── ts_tcc_model.py          ← TSTCCModel wrapper
├── scripts/
│   └── 02_pretrain_ssl/
│       └── run_pretrain_tstcc.py        ← Entry point CLI
└── utils/
    └── checkpoint_paths.py              ← ssl_checkpoint_dir(), ssl_checkpoint_name(), get_output_root()
```

### 2.2. Entry point và cấu hình

**File:** `scripts/02_pretrain_ssl/run_pretrain_tstcc.py`
- **Vai trò:** Script CLI chính.
- **Argument:** `--backbone` (default: `standard`), `--epochs` (default: `60`), `--batch_size` (default: `64`), `--datasets` (default: `["uci_har", "motion_sense"]`).
- **Gọi:** `train_tstcc_encoder()` trong `engines/pretrain_ssl/tstcc/tstcc_trainer.py`.
- Tự phát hiện Kaggle qua biến môi trường `KAGGLE_KERNEL_RUN_TYPE`.

**Checkpoint output pattern:**
```
checkpoints/ssl_pretrain/tstcc/<dataset_name>/<backbone_type>/
    tstcc_<backbone>_encoder_pretrained_<dataset>.pt
```

**Dataset registry** được đọc từ `config/datasets/registry.py` → `DATASETS[name]["all"]`.

### 2.3. Config tập trung — TSTCCConfig

**File:** `config/methods/tstcc_config.py`
**Class:** `TSTCCConfig`

```python
class TSTCCConfig:
    # Temporal Contrasting
    TIMESTEPS = 6               # K bước tương lai (mặc định 6, bài báo gốc 3)
    TC_HIDDEN_DIM = 64          # Chiều ẩn Transformer trong TC
    TC_DEPTH = 4                # Số tầng Transformer
    TC_HEADS = 4                # Số attention heads
    TC_MLP_DIM = 64             # Chiều FFN trong Transformer

    # Contrastive Learning
    LAMBDA1 = 1.0               # Trọng số L_TC
    LAMBDA2 = 0.7               # Trọng số L_CC
    TEMPERATURE = 0.2           # NT-Xent temperature

    # Encoder Architecture
    KERNEL_SIZE = 8             # Conv1d kernel size
    DROPOUT = 0.35              # Dropout rate

    # Training Configuration
    EPOCHS = 60
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 3e-4
    GRAD_CLIP_MAX_NORM = 2.0    # Gradient clipping (thêm mới so với tài liệu cũ)
```

> **Thay đổi quan trọng so với tài liệu cũ:**
> - `TIMESTEPS = 6` (tài liệu cũ ghi 3). Đây là giá trị hiện tại trong config.
> - `GRAD_CLIP_MAX_NORM = 2.0` được đưa vào config (trước đây hard-coded trong trainer).

### 2.4. Dataset và DataLoader

**File:** `datasets/contrastive_dataset.py`
**Class:** `ContrastiveDatasetWrapper`
- **Input:** Đường dẫn đến file `.pt`.
- **Output:** Cặp `(x_weak, x_strong)` — mỗi phần tử shape `(6, 128)`.
- Nạp từ key `"samples"` trong dict `.pt`.
- Không dùng nhãn (SSL là unsupervised).

**Được tạo trong:** `tstcc_trainer.py` → `_build_loader()`.

**DataLoader settings trong `_build_loader()`:**
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=len(dataset) >= batch_size,
    num_workers=use_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(use_workers > 0),
)
```

### 2.5. Training engine — `train_tstcc_encoder()`

**File:** `engines/pretrain_ssl/tstcc/tstcc_trainer.py`

Hàm này nhận toàn bộ tham số và điều phối:
1. `_build_loader()` — tạo DataLoader
2. `_build_ssl_model()` — tạo TSTCCModel (hoặc dùng model đã truyền vào)
3. Lưu `config.json` qua `save_json(build_config(...))`
4. Tạo `AdamW` optimizer
5. Vòng lặp epoch gọi `_train_one_epoch()`
6. Checkpoint selection: lưu khi `loss_total < best_loss`
7. `save_training_outputs()` — CSV, PNG, complexity.json, pretrain_summary.json

---

## Phần 3. Dữ liệu đầu vào và cấu trúc dữ liệu

### 3.1. Datasets được sử dụng

| Tên trong code | Key trong registry | File data |
|---|---|---|
| `uci_har` | `DATASETS["uci_har"]["all"]` | `data/processed/uci_har/dataset_all.pt` |
| `motion_sense` | `DATASETS["motion_sense"]["all"]` | `data/processed/motionsense/dataset_all.pt` |
| `hhar_phone` | `DATASETS["hhar_phone"]["all"]` | `data/processed/hhar/phone/dataset_all.pt` |
| `hhar_watch` | `DATASETS["hhar_watch"]["all"]` | `data/processed/hhar/watch/dataset_all.pt` |

> **Lưu ý:** Key trong registry là `"motion_sense"` (có underscore), không phải `"motionsense"`.

Dùng `dataset_all.pt` (toàn bộ dữ liệu không tách train/val/test) cho SSL pretraining.

### 3.2. Định dạng file `.pt`

```python
{
    "samples":  torch.Tensor,   # shape (N, 6, 128) — float32
    "labels":   torch.Tensor,   # shape (N,) — long
    "subjects": torch.Tensor,   # shape (N,) — subject ID
}
```

> `ContrastiveDatasetWrapper` chỉ đọc `"samples"`, bỏ qua `"labels"`.

### 3.3. Cấu trúc một mẫu

```text
x: (6, 128)
    Chiều 0 (6 kênh):
        [0] acc_x  [1] acc_y  [2] acc_z
        [3] gyro_x [4] gyro_y [5] gyro_z
    Chiều 1 (128 timestep): 2.56 giây @ 50Hz, stride 64 (overlap 50%)
```

### 3.4. Kích thước tensor qua pipeline

```text
File .pt → samples: (N, 6, 128)
                        ↓  ContrastiveDatasetWrapper.__getitem__
x_raw: (6, 128)  →  x_weak: (6, 128)  +  x_strong: (6, 128)
                        ↓  DataLoader collate
Batch: x_weak (B, 6, 128)  +  x_strong (B, 6, 128)
                        ↓  TSTCCEncoder.forward
z_weak: (B, 128, 18)  +  z_strong: (B, 128, 18)    [với tstcc backbone]
                        ↓  TC.forward (Bidirectional)
c_t: (B, 64)  → projection_head(c_t): (B, 32)      loss_tc: scalar
                        ↓  NTXentLoss
                       loss_cc: scalar
                        ↓
              loss_total = 1.0*loss_tc + 0.7*loss_cc
```

---

## Phần 4. Tiền xử lý và chuẩn hóa

> Tiền xử lý và tạo sliding window được thực hiện trong `scripts/00_preprocess/`, kết quả lưu vào `data/processed/`.

### 4.1. Luồng tiền xử lý

```text
Dữ liệu thô (CSV/txt)
    → Đọc, chọn 6 cột cảm biến
    → Chuẩn hóa (tùy dataset)
    → Sliding window: size=128, stride=64
    → Gán subject ID
    → Lưu thành train.pt / val.pt / test.pt / dataset_all.pt
```

### 4.2. Tiền xử lý trong Dataset

`ContrastiveDatasetWrapper` chỉ ép kiểu sang float32:
```python
self.samples = self.samples.float()
```
**Không có** normalization bổ sung trong runtime.

---

## Phần 5. Data Augmentation

### 5.1. Overview

**File:** `../../models/ssl/tstcc/augmentations.py`
**Class:** `TS_TCC_Augmentation`

Augmentation áp dụng **on-the-fly** trong `__getitem__`, mỗi lần lấy mẫu sinh view ngẫu nhiên.

> **Dùng chung với SwAV:** `ContrastiveDatasetWrapper` và `TS_TCC_Augmentation` được dùng cho cả TS-TCC và SwAV.

### 5.2. Weak Augmentation

```python
def weak_transform(self, x: np.ndarray) -> np.ndarray:
    return scaling(x, sigma=self.jitter_scale_ratio)  # sigma=1.1 mặc định
```

**`scaling`:** Nhân tín hiệu với hệ số ~ N(2.0, 1.1). Factor shape `(B, 1, T)` broadcast qua kênh.

> **Lưu ý:** `loc=2.0` nghĩa là biên độ trung bình bị nhân đôi — thiết kế theo bài báo gốc.

### 5.3. Strong Augmentation

```python
def strong_transform(self, x: np.ndarray) -> np.ndarray:
    x_perm = permutation(x, max_segments=self.max_seg)   # max_seg=5
    x_strong = jitter(x_perm, sigma=self.jitter_ratio)   # sigma=0.8
    return x_strong
```

**`permutation`:** Cắt chuỗi thành `num_segs ∈ [1, 5]` đoạn ngẫu nhiên, xáo trộn, áp dụng đồng bộ tất cả kênh.
**`jitter`:** Thêm nhiễu N(0, 0.8).

### 5.4. Xử lý chiều tensor

```python
# augmentations.py — detect và transpose nếu nhận (T, C) thay vì (C, T)
if x_numpy.ndim == 2 and x_numpy.shape[0] == 128 and x_numpy.shape[1] == 6:
    x_numpy = x_numpy.T
```

### 5.5. Pipeline augmentation tóm tắt

```text
x_raw: (6, 128) [numpy]
    +─── weak_transform  → scaling(σ=1.1)                         → x_weak: (6,128) → Tensor float32
    +─── strong_transform → permutation(max_seg=5) → jitter(σ=0.8) → x_strong: (6,128) → Tensor float32
```

---

## Phần 6. Kiến trúc mô hình

### 6.1. Encoder backbone — TSTCCEncoder

**File:** `models/encoders/tstcc_encoder.py`
**Class:** `TSTCCEncoder`
**Nguồn gốc:** Reproduction của `base_Model` trong repo gốc `emadeldeen24/TS-TCC`.

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

> **Thiết kế:** Encoder **không có Global Average Pooling** ở cuối — TC module cần sequence dimension.

### 6.2. Các backbone khác

| backbone_type | Class | Output Shape | Ghi chú |
|---|---|---|---|
| `tstcc` | `TSTCCEncoder` | `(B, 128, 18)` | Phù hợp nhất cho TC |
| `standard` | `StandardSensorEncoder1D` | `(B, 128, T')` | Default trong CLI |
| `cnn_transformer` | `CNNTransformerEncoder` | `(B, 128, T')` | — |
| `vit_1d` | `ViT1DEncoder` | `(B, 128, T')` | — |

Khi dùng backbone khác `tstcc`, TC module vẫn hoạt động với `seq_len` tương ứng.

### 6.3. TC module — Temporal Contrasting

**File:** `../../models/ssl/tstcc/tc.py`
**Class:** `TC`

```python
class TC(nn.Module):
    def __init__(self, final_out_channels=128, timesteps=3, hidden_dim=64,
                 depth=4, heads=4, mlp_dim=64):
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, final_out_channels)
                                  for _ in range(timesteps)])  # K projection layers
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, final_out_channels // 2),  # 64 → 64
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),  # 64 → 32
        )
        self.seq_transformer = Seq_Transformer(
            patch_size=final_out_channels, dim=hidden_dim,
            depth=depth, heads=heads, mlp_dim=mlp_dim,
        )
```

**Khởi tạo trong `TSTCCModel`:** `timesteps=TSTCCConfig.TIMESTEPS=6`.

**Forward flow:**
```text
Input: z_aug1 (B, 128, 18), z_aug2 (B, 128, 18), t_samples: int

Bước 1: Transpose → z_aug1: (B, 18, 128), z_aug2: (B, 18, 128)
Bước 2: encode_samples[k] = z_aug2[:, t_samples+k, :]  → (K, B, 128)   k=1..K=6
Bước 3: forward_seq = z_aug1[:, :t_samples+1, :]        → (B, t_samples+1, 128)
Bước 4: c_t = seq_transformer(forward_seq)              → (B, 64)
Bước 5: pred[k] = Wk[k](c_t)                           → (K, B, 128)
Bước 6: InfoNCE = -Σ_k diag(LogSoftmax(encode_samples[k] @ pred[k].T)) / (B * K)
Output: (nce_loss, projection_head(c_t), t_samples)
         scalar    (B, 32)              int
```

**Projection head output `(B, 32)`** — đây là `context_w` / `context_s` cho NT-Xent.

### 6.4. Seq_Transformer

**File:** `../../models/ssl/tstcc/attention.py`
**Class:** `Seq_Transformer`

```text
Input: forward_seq: (B, t+1, 128)
    ↓ patch_to_embedding: Linear(128, 64) → (B, t+1, 64)
    ↓ Prepend CLS token (learnable param (1, 1, 64)): (B, t+2, 64)
    ↓ Transformer (depth=4, heads=4, mlp_dim=64)
    ↓ Lấy CLS token: x[:, 0] → (B, 64)
Output: c_t (B, 64)
```

**Classes hỗ trợ:** `Transformer`, `Attention`, `FeedForward`, `PreNorm`, `Residual`. Dùng thư viện `einops`.

### 6.5. NTXentLoss — Contextual Contrasting

**File:** `losses/nt_xent.py`
**Class:** `NTXentLoss(temperature=0.2)`

```text
Input: h_weak (B, 32), h_strong (B, 32)
    ↓ L2 normalize
    ↓ Concat: representations (2B, 32)
    ↓ Similarity matrix (2B, 2B) / temperature
    ↓ Mask self-similarity (diagonal)
    ↓ CrossEntropyLoss (positive pair = view kia)
Output: loss_cc: scalar
```

### 6.6. TSTCCModel — Wrapper tổng hợp

**File:** `../../models/ssl/tstcc/ts_tcc_model.py`
**Class:** `TSTCCModel`

```python
class TSTCCModel(nn.Module):
    def __init__(self, encoder=None, in_channels=6, feature_dim=128,
                 timesteps=6, tc_hidden_dim=64, tc_depth=4, tc_heads=4,
                 tc_mlp_dim=64, lambda1=1.0, lambda2=0.7, temperature=0.2):
```

**Forward:**
```text
x_weak (B,6,128) + x_strong (B,6,128)
    ↓ Encoder (trọng số chung)
z_weak (B,128,18) + z_strong (B,128,18)
    ↓ Bốc t_samples ∈ [0, seq_len - timesteps - 1] = [0, 11]
TC(z_weak, z_strong, t_samples)   → loss_tc1, context_w (B,32), _
TC(z_strong, z_weak, t_samples)   → loss_tc2, context_s (B,32), _
loss_tc = loss_tc1 + loss_tc2
    ↓ NTXentLoss
NT-Xent(context_w, context_s)     → loss_cc
    ↓
loss_total = 1.0 * loss_tc + 0.7 * loss_cc
```

**Sơ đồ luồng:**

```
x_weak (B,6,128)   ──→  TSTCCEncoder  ──→  z_weak (B,128,18)  ──┐
                                                                   ├──→ TC(W→S) ──→ loss_tc1, context_w (B,32)
x_strong (B,6,128) ──→  TSTCCEncoder  ──→  z_strong (B,128,18) ─┤
                                                                   ├──→ TC(S→W) ──→ loss_tc2, context_s (B,32)
                                                                   │
                                              context_w, context_s ↓
                                                             NTXentLoss  ──→  loss_cc
                                                                   ↓
                                              loss_total = 1.0·loss_tc + 0.7·loss_cc
```

---

## Phần 7. Quy trình Self-Supervised Pretraining

### 7.1. Hàm chính — `train_tstcc_encoder()`

**File:** `engines/pretrain_ssl/tstcc/tstcc_trainer.py`

```python
def train_tstcc_encoder(
    data_path,            # Path đến dataset_all.pt
    save_dir,             # Thư mục cha chứa checkpoint
    checkpoint_name,      # Tên file checkpoint
    backbone_type,        # "standard" | "tstcc" | "cnn_transformer" | "vit_1d"
    model=None,           # Truyền model sẵn hoặc tự build từ TSTCCConfig
    in_channels=6,
    feature_dim=128,
    batch_size=64,
    learning_rate=3e-4,
    weight_decay=3e-4,
    epochs=60,
    temperature=0.2,
    device="cuda",
    measure_complexity=True,
    seq_len=128,
    num_workers=2
):
```

**Checkpoint output directory:**
```python
target_dir = save_dir / backbone_type
# Ví dụ: checkpoints/ssl_pretrain/tstcc/uci_har/standard/
```

### 7.2. Khởi tạo model — `_build_ssl_model()`

```python
def _build_ssl_model(model, backbone_type, in_channels, feature_dim, temperature, device):
    if model is not None:
        return model.to(device)
    return TSTCCModel(
        encoder=build_encoder(backbone_type=backbone_type, in_channels=in_channels),
        in_channels=in_channels,
        feature_dim=feature_dim,
        timesteps=TSTCCConfig.TIMESTEPS,          # 6
        lambda1=TSTCCConfig.LAMBDA1,              # 1.0
        lambda2=TSTCCConfig.LAMBDA2,              # 0.7
        temperature=temperature,
        tc_hidden_dim=TSTCCConfig.TC_HIDDEN_DIM,  # 64
        tc_depth=TSTCCConfig.TC_DEPTH,            # 4
        tc_heads=TSTCCConfig.TC_HEADS,            # 4
        tc_mlp_dim=TSTCCConfig.TC_MLP_DIM,        # 64
    ).to(device)
```

### 7.3. Optimizer

```python
optimizer = AdamW(ssl_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# learning_rate=3e-4, weight_decay=3e-4
# Một optimizer duy nhất cho encoder + TC + NTXent
```

> **Khác SwAV:** SwAV dùng AdamW với 2 nhóm param (decay/no_decay). TS-TCC dùng AdamW đơn giản cho toàn bộ model.

### 7.4. Một training iteration — `_train_one_epoch()`

```python
def _train_one_epoch(ssl_model, loader, optimizer, device):
    ssl_model.train()
    totals = {"loss_total": 0.0, "loss_tc": 0.0, "loss_cc": 0.0, "grad_norm": 0.0}

    for x_w, x_s in loader:
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss, loss_tc, loss_cc = ssl_model(x_w, x_s)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(
            ssl_model.parameters(), max_norm=TSTCCConfig.GRAD_CLIP_MAX_NORM  # 2.0
        )
        optimizer.step()

        totals["loss_total"] += loss.item()
        totals["loss_tc"]    += loss_tc.item()
        totals["loss_cc"]    += loss_cc.item()
        totals["grad_norm"]  += gnorm.item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}
```

**Gradient clipping:** `max_norm=2.0` (từ `TSTCCConfig.GRAD_CLIP_MAX_NORM`).

### 7.5. Checkpoint selection

```python
if metrics["loss_total"] < best_loss:
    best_loss = metrics["loss_total"]
    torch.save(ssl_model.encoder.state_dict(), checkpoint_path)
```

**Chỉ lưu `encoder.state_dict()`** — TC module và NTXentLoss không được lưu.
Checkpoint chọn theo **train loss thấp nhất**, không có validation set.

> **Khác SwAV:** TS-TCC không có warmup period — lưu checkpoint ngay từ epoch 1 nếu loss tốt hơn best.

### 7.6. Outputs sau pretraining

```text
checkpoints/ssl_pretrain/tstcc/<dataset_name>/<backbone_type>/
    ├── tstcc_<backbone>_encoder_pretrained_<dataset>.pt  ← encoder.state_dict()
    ├── loss_history.csv        ← epoch, loss_total, loss_tc, loss_cc, grad_norm
    ├── loss_curve.png          ← biểu đồ hội tụ
    ├── config.json             ← cấu hình huấn luyện đầy đủ
    ├── complexity.json         ← FLOPs, params của backbone
    └── pretrain_summary.json   ← (tại thư mục cha tstcc/)
```

---

## Phần 8. Loss Functions chi tiết

### 8.1. InfoNCE Loss (Temporal Contrasting)

**Triển khai trong `tc.py`:**

```python
nce = 0.0
for i in range(self.timestep):  # timestep = 6
    # total: (B, B) — similarity matrix
    total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
    # Chỉ lấy giá trị trên đường chéo (cùng mẫu)
    nce += torch.sum(torch.diag(self.lsoftmax(total)))
nce /= -1.0 * batch_size * self.timestep
```

**Công thức:**

$$\mathcal{L}_{TC} = -\frac{1}{B \cdot K} \sum_{k=1}^{K} \sum_{i=1}^{B} \log \frac{\exp(\text{enc}_k^{(i)} \cdot \text{pred}_k^{(i)})}{\sum_{j=1}^{B} \exp(\text{enc}_k^{(i)} \cdot \text{pred}_k^{(j)})}$$

Trong đó $K = \text{TIMESTEPS} = 6$, $B$ là batch size.

### 8.2. NT-Xent Loss (Contextual Contrasting)

$$\mathcal{L}_{CC} = -\frac{1}{2B} \sum_{i=1}^{2B} \log \frac{\exp(\text{sim}(h_i, h_{i+})/\tau)}{\sum_{j \neq i} \exp(\text{sim}(h_i, h_j)/\tau)}$$

Với $\tau = 0.2$. L2-normalized trước khi tính cosine similarity.

### 8.3. Loss tổng

$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{TC} + 0.7 \cdot \mathcal{L}_{CC}$$

Và $\mathcal{L}_{TC} = \mathcal{L}_{TC}^{W \to S} + \mathcal{L}_{TC}^{S \to W}$ (bidirectional, dùng cùng `t_samples`).

---

## Phần 9. Chi tiết riêng của TS-TCC

### 9.1. Bidirectional Temporal Contrasting

TC chạy hai chiều với cùng `t_samples` (lấy ngẫu nhiên một lần trước):

```python
# ts_tcc_model.py
seq_len = z_weak.shape[2]  # = 18 với TSTCCEncoder
t_samples = torch.randint(0, seq_len - self.timesteps, size=(1,)).item()
# t_samples ∈ [0, 18 - 6 - 1] = [0, 11] với TIMESTEPS=6

loss_tc1, context_w, _ = self.tc(z_weak, z_strong, t_samples=t_samples)   # Weak → Strong
loss_tc2, context_s, _ = self.tc(z_strong, z_weak, t_samples=t_samples)   # Strong → Weak
loss_tc = loss_tc1 + loss_tc2
```

### 9.2. Context vector cho NT-Xent

- `context_w = TC.projection_head(c_t)` khi TC nhận `(z_weak → z_strong)` — shape `(B, 32)`
- `context_s = TC.projection_head(c_t)` khi TC nhận `(z_strong → z_weak)` — shape `(B, 32)`

Projection head: `Linear(64→64) → BN → ReLU → Linear(64→32)`.

### 9.3. Gradient clipping — khác với SwAV

```python
gnorm = nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=2.0)
```

SwAV không có gradient clipping. TS-TCC có clipping max_norm=2.0 để ổn định training.

### 9.4. Không có warmup / scheduler — khác với SwAV

TS-TCC không dùng LR scheduler trong pretraining. Learning rate cố định 3e-4 trong suốt quá trình.
SwAV có linear warmup + cosine annealing.

---

## Phần 10. Downstream Task: Linear Probing và Fine-tuning

### 10.1. Downstream model

**File:** `models/har_classifier.py` — `HARClassifier`

```text
HARClassifier:
    encoder: encoder backbone (load từ checkpoint SSL pretrain)
    classifier: ClassifierHead(feature_dim=128, num_classes=5, dropout=0.2)
```

**ClassifierHead (`models/heads/classifier.py`):**
```python
if x.dim() == 3:   # Encoder trả (B, D, L) — tức là TSTCCEncoder
    x = torch.mean(x, dim=-1)  # → (B, D) mean pooling
logits = fc(x)     # Linear(128, num_classes=5)
```

> `ClassifierHead` tự động xử lý cả input 2D `(B, D)` và 3D `(B, D, L)`.

### 10.2. Load checkpoint

```python
# finetune_trainer.py
checkpoint = torch.load(encoder_ckpt_path, map_location=target_device, weights_only=True)
# Hỗ trợ nhiều format:
if isinstance(checkpoint, dict):
    if "encoder_state_dict" in checkpoint: encoder_dict = checkpoint["encoder_state_dict"]
    elif "encoder" in checkpoint: encoder_dict = checkpoint["encoder"]
    else: encoder_dict = checkpoint
else:
    encoder_dict = checkpoint
model.encoder.load_state_dict(encoder_dict, strict=True)
```

### 10.3. Linear Probing

```python
for param in model.encoder.parameters():
    param.requires_grad = False
optimizer = Adam(model.classifier.parameters(), lr=1e-3, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(mode="max", factor=0.5, patience=5)  # theo val Macro F1
criterion = nn.CrossEntropyLoss()
```

### 10.4. Full Fine-tuning

```python
for param in model.encoder.parameters():
    param.requires_grad = True
optimizer = Adam([
    {"params": model.encoder.parameters(),    "lr": 1e-4},   # backbone_lr (nhỏ hơn)
    {"params": model.classifier.parameters(), "lr": 1e-3},   # head_lr
])
```

### 10.5. K-shot sampling

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

- Lấy đúng **k mẫu mỗi lớp** → K×5 mẫu train tổng.
- Val và Test dùng **100% dữ liệu** (không K-shot).
- Cố định bởi `seed`.

---

## Phần 11. Cross-Domain Training và Evaluation

### 11.1. Kịch bản

Source domain: pretrain trên UCI-HAR hoặc MotionSense (hoặc HHAR).
Target domain: đánh giá trên các dataset còn lại.

Encoder pretrained trên source → load vào `HARClassifier` → fine-tune/LP trên K mẫu của target train → đánh giá trên toàn bộ target test set.

### 11.2. Checkpoint convention (cập nhật)

```
checkpoints/ssl_pretrain/tstcc/<src_dataset>/<backbone>/
    tstcc_<backbone>_encoder_pretrained_<src_dataset>.pt
```

Đường dẫn tạo bởi `utils/checkpoint_paths.py`:
```python
ssl_checkpoint_dir(output_root, method="tstcc", dataset=dataset_name, backbone=backbone_type)
# → <output_root>/checkpoints/ssl_pretrain/tstcc/<dataset>/<backbone>/

ssl_checkpoint_name(method="tstcc", backbone=backbone, dataset=dataset_name)
# → "tstcc_<backbone>_encoder_pretrained_<dataset>.pt"
```

Tag phụ (khi có `--tag`): `backbone_<tag>/` thay vì `backbone/`.

### 11.3. Label space

5 lớp chung (label 0–4: Walking, Upstairs, Downstairs, Sitting, Standing):
```python
# run_few_shot.py
mask = (d["labels"].squeeze() >= 0) & (d["labels"].squeeze() < 5)
```

---

## Phần 12. Evaluation Metrics và Kết quả đầu ra

### 12.1. Metrics được tính

| Metric | Function | Thư viện |
|---|---|---|
| Accuracy | `accuracy_score(y_true, y_pred)` | sklearn |
| Macro F1 | `f1_score(..., average="macro")` | sklearn |
| Per-class F1 | `classification_report(..., output_dict=True)` | sklearn |
| Cohen's Kappa | `cohen_kappa_score(y_true, y_pred)` | sklearn |
| Confusion Matrix | `confusion_matrix(y_true, y_pred)` | sklearn |

**Metric chính:** Macro F1 (%).

### 12.2. Đầu ra evaluation

```text
outputs_evaluation/cross_k_shot/tstcc/<backbone>/<src>_to_<tgt>/
    ├── experiment_metadata.json
    └── <k>_shot/
        ├── linear_probing/
        │   ├── summary_linear_probing_<k>_shot.json   ← mean ± std qua seeds
        │   ├── detailed_all_seeds.json
        │   └── aggregated_confusion_matrix.png
        └── full_finetuning/
            ├── summary_full_finetuning_<k>_shot.json
            ├── detailed_all_seeds.json
            └── aggregated_confusion_matrix.png
```

---

## Phần 13. Hyperparameters Đầy đủ

| Hyperparameter | Giá trị | Nguồn |
|---|---|---|
| `backbone` | `standard` (default) | CLI `--backbone` |
| `epochs` | 60 | `TSTCCConfig.EPOCHS` |
| `batch_size` | 64 | `TSTCCConfig.BATCH_SIZE` |
| `learning_rate` | 3e-4 | `TSTCCConfig.LEARNING_RATE` |
| `weight_decay` | 3e-4 | `TSTCCConfig.WEIGHT_DECAY` |
| `grad_clip_max_norm` | 2.0 | `TSTCCConfig.GRAD_CLIP_MAX_NORM` |
| `timesteps` (K) | **6** | `TSTCCConfig.TIMESTEPS` |
| `lambda1` | 1.0 | `TSTCCConfig.LAMBDA1` |
| `lambda2` | 0.7 | `TSTCCConfig.LAMBDA2` |
| `temperature` (NT-Xent) | 0.2 | `TSTCCConfig.TEMPERATURE` |
| `tc_hidden_dim` | 64 | `TSTCCConfig.TC_HIDDEN_DIM` |
| `tc_depth` | 4 | `TSTCCConfig.TC_DEPTH` |
| `tc_heads` | 4 | `TSTCCConfig.TC_HEADS` |
| `tc_mlp_dim` | 64 | `TSTCCConfig.TC_MLP_DIM` |
| `feature_dim` | 128 | Hard-coded |
| `in_channels` | 6 | `BaseDatasetConfig.IN_CHANNELS` |
| `window_size` | 128 | Config dataset |
| `stride` | 64 | Config dataset (overlap 50%) |
| Augmentation (weak) | scaling σ=1.1, loc=2.0 | `TS_TCC_Augmentation` |
| Augmentation (strong) | permutation(max_seg=5) + jitter(σ=0.8) | `TS_TCC_Augmentation` |
| Optimizer | AdamW (1 group) | Hard-coded |
| LR Scheduler | Không có | — |
| Checkpoint criterion | Min train loss (không có warmup) | tstcc_trainer.py |

---

## Phần 14. Cài đặt và Chạy Code

### 14.1. Yêu cầu môi trường

```
Python >= 3.11
torch >= 2.0.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
einops           # cần thiết cho attention.py (Seq_Transformer)
thop             # optional, cho complexity measurement (FLOPs)
```

### 14.2. Chạy SSL Pretraining

```bash
# Test nhanh 1 epoch
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc.py \
  --backbone standard --epochs 1 --batch_size 64 --datasets uci_har

# Chạy thật đủ 4 dataset, 60 epoch
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc.py \
  --backbone standard --epochs 60 --batch_size 64 \
  --datasets uci_har motion_sense hhar_phone hhar_watch

# Dùng backbone nặng hơn, giảm batch nếu OOM
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc.py \
  --backbone vit_1d --epochs 60 --batch_size 32 --datasets uci_har motion_sense
```

### 14.3. Chạy K-shot cross-domain evaluation

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method tstcc \
  --backbone standard \
  --k_shots 10 20 50 100 \
  --seeds 42 100 2024 7 99 \
  --epochs 40 \
  --pairs motion_sense:uci_har uci_har:motion_sense
```

### 14.4. Checklist trước khi chạy thật

```bash
# 1. Data tồn tại
ls -la data/processed/uci_har/dataset_all.pt

# 2. thop đã cài
uv pip list | grep -i thop

# 3. Test 1 epoch không lỗi
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc.py --epochs 1 --datasets uci_har

# 4. Xem kết quả
cat checkpoints/ssl_pretrain/tstcc/uci_har/standard/loss_history.csv
cat checkpoints/ssl_pretrain/tstcc/uci_har/standard/complexity.json
```

---

## Phần 15. Tài liệu Tham khảo

**Bài báo gốc TS-TCC:**
- Eldele, E., et al. "Time-Series Representation Learning via Temporal and Contextual Contrasting." IJCAI 2021.
- Repository gốc: `emadeldeen24/TS-TCC`.

**Implementation trong repository:**
- `../../models/ssl/tstcc/ts_tcc_model.py` — `TSTCCModel` wrapper
- `../../models/ssl/tstcc/tc.py` — `TC` class (Temporal Contrasting)
- `../../models/ssl/tstcc/attention.py` — `Seq_Transformer`
- `engines/pretrain_ssl/tstcc/tstcc_trainer.py` — `train_tstcc_encoder()`
- `config/methods/tstcc_config.py` — `TSTCCConfig` (tất cả hyperparameters)
- `utils/checkpoint_paths.py` — `ssl_checkpoint_dir()`, `ssl_checkpoint_name()`
