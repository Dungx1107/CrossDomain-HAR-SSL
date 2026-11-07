# Tài liệu Kỹ thuật: SwAV — Prototype-based SSL cho HAR

> **Mục đích tài liệu:** Mô tả chính xác implementation của phương pháp SwAV trong repository CrossDomain-HAR-SSL dựa trên phân tích mã nguồn thực tế.
> **Phiên bản phân tích:** Cập nhật theo snapshot code hiện tại (November 2026).
> **Tên file cũ:** `PROTOTYPE_SSL.md` → đổi thành `SwAV_SSL.md` để khớp với tên method thực tế trong code (`swav`).
> **Tài liệu này dành cho:** AI review code — mọi thông tin đều dẫn chiếu trực tiếp đến file và function tương ứng.

---

## Phần 1. Tổng quan

### 1.1. Mục tiêu trong dự án

SwAV (Swapping Assignments between Views) là phương pháp SSL prototype-based được triển khai trong dự án, nằm trong nhóm phương pháp phân cụm (clustering-based SSL). Mục tiêu là học biểu diễn đặc trưng bằng cách gán mỗi sample vào một trong $K$ **prototype vectors** (cluster centers) và huấn luyện model dự đoán soft assignment của view này từ view kia thông qua thuật toán **Sinkhorn-Knopp optimal transport**.

Phương pháp được thiết kế để:
1. Học biểu diễn có cấu trúc phân cụm ngữ nghĩa (prototype là "trung tâm đặc trưng" của các pattern hoạt động).
2. Tránh mode collapse thông qua Sinkhorn-Knopp equipartition (mỗi prototype phải được gán đều).

Sau pretraining, **chỉ encoder backbone** được giữ lại (prototype và projection head bị loại bỏ) để dùng trong downstream fine-tuning/linear probing.

### 1.2. Lý thuyết gốc

**SwAV (Caron et al., NeurIPS 2020 — facebookresearch/swav):**
- Không so sánh trực tiếp embedding của hai view (như NT-Xent/TS-TCC).
- Thay vào đó, gán mỗi view vào các prototype (soft assignment `q` qua Sinkhorn-Knopp) và dạy view này tái tạo assignment của view kia (`p` qua softmax).
- Sinkhorn-Knopp chuẩn hóa soft assignments thành phân phối đều để tránh collapse.

**Điểm khác biệt với SwAV gốc trong repository này (có chủ ý):**
- Dùng cho dữ liệu cảm biến 1D thay vì ảnh 2D.
- AdamW thay vì SGD + LARC; LR theo epoch thay vì iteration.
- 2 view (weak/strong) thay multi-crop. Không có queue.
- Chuẩn hóa prototype trong `forward` (gradient-based, "cách B") thay vì in-place trước forward ("cách A").
- Prototype đóng băng trong `FREEZE_PROTOTYPES_EPOCHS` đầu, không phải `freeze_prototypes_niters` iterations.
- Checkpoint chọn sau warmup theo train loss.

### 1.3. Vai trò trong pipeline tổng thể

```text
SSL Pretraining (source domain)         → checkpoints/ssl_pretrain/swav/<source>/<backbone>/
        |
        ↓  encoder.state_dict() được lưu (prototype & projection head bị loại bỏ)
K-shot / Fraction Cross-Domain          → scripts/03_cross_domain/run_few_shot.py
        |
        ↓  load encoder, freeze hoặc finetune
Linear Probing / Full Fine-tuning       → engines/transfer/finetune_trainer.py
        |
        ↓
Đánh giá trên target test set           → outputs_evaluation/cross_k_shot/swav/
```

### 1.4. Các thành phần code chính

| Thành phần | File | Mô tả |
|---|---|---|
| ContrastiveDatasetWrapper | `datasets/contrastive_dataset.py` | Dataset sinh (x_weak, x_strong) — dùng chung với TS-TCC |
| Encoder backbone | `models/encoders/builder.py` | Factory — bất kỳ backbone nào |
| ProjectionHead | `models/heads/projection.py` | MLP 2 lớp + L2 norm |
| SwAVModel | `models/ssl/swav/swav_model.py` | Wrapper: encoder + projection + prototypes |
| SwAVLoss | `losses/swav_loss.py` | Loss SwAV + Sinkhorn-Knopp + entropy monitor |
| SwAVConfig | `config/methods/swav_config.py` | Tập trung hyperparameters |
| Pretrain Engine | `engines/pretrain_ssl/swav/swav_trainer.py` | `train_swav_encoder()` |
| Reporting | `engines/pretrain_ssl/swav/swav_reporting.py` | Log CSV, biểu đồ, complexity |
| Entry point | `scripts/02_pretrain_ssl/run_pretrain_swav.py` | CLI script khởi chạy |
| Checkpoint paths | `utils/checkpoint_paths.py` | Helper đường dẫn checkpoint |
| Downstream | `engines/transfer/finetune_trainer.py` | Dùng chung với TS-TCC |

---

## Phần 2. Cấu trúc mã nguồn liên quan

### 2.1. Cấu trúc thư mục (các file liên quan SwAV)

```
CrossDomain-HAR-SSL/
├── config/
│   └── methods/
│       └── swav_config.py               ← SwAVConfig: tất cả hyperparameters
├── datasets/
│   └── contrastive_dataset.py           ← ContrastiveDatasetWrapper (dùng chung với TS-TCC)
├── engines/
│   └── pretrain_ssl/
│       └── swav/
│           ├── swav_trainer.py          ← train_swav_encoder(), các helper
│           └── swav_reporting.py        ← Logging, plotting, complexity
├── losses/
│   └── swav_loss.py                     ← SwAVLoss, sinkhorn(), cluster_usage_entropy()
├── models/
│   ├── encoders/
│   │   ├── builder.py                   ← build_encoder() factory
│   │   └── ...                          ← cnn1d.py, tstcc_encoder.py, vit_1d.py, cnn_transformer.py
│   ├── heads/
│   │   └── projection.py               ← ProjectionHead (feature_dim→projection_dim, L2 norm)
│   └── ssl/
│       └── swav/
│           └── swav_model.py            ← SwAVModel (encoder + projection + prototypes)
├── scripts/
│   └── 02_pretrain_ssl/
│       └── run_pretrain_swav.py         ← Entry point CLI (thay thế run_pretrain_prototype.py cũ)
└── utils/
    └── checkpoint_paths.py              ← ssl_checkpoint_dir(), ssl_checkpoint_name()
```

### 2.2. Entry point và cấu hình

**File:** `scripts/02_pretrain_ssl/run_pretrain_swav.py`
- **Vai trò:** Script CLI chính (thay thế `run_pretrain_prototype.py` cũ).
- **Argument:**
  - `--datasets` (required): danh sách dataset key từ registry.
  - `--backbone` (default: `standard`): `tstcc | standard | cnn_transformer | vit_1d`.
  - `--epochs` (default: `SwAVConfig.EPOCHS = 40`).
  - `--batch_size` (default: `SwAVConfig.BATCH_SIZE = 64`).
  - `--lr` (default: `SwAVConfig.LEARNING_RATE = 3e-4`).
  - `--num_workers` (default: 4).
  - `--device` (default: `cuda`).
  - `--tag` (default: `""`): tag phụ cho thư mục checkpoint (vd `"bs128"`).
- **Gọi:** `train_swav_encoder()` trong `engines/pretrain_ssl/swav/swav_trainer.py`.

**Checkpoint output:**
```
checkpoints/ssl_pretrain/swav/<dataset_name>/<backbone_type>/
    swav_<backbone>_encoder_pretrained_<dataset>.pt

# Với tag:
checkpoints/ssl_pretrain/swav/<dataset_name>/<backbone_type>_<tag>/
    swav_<backbone>_encoder_pretrained_<dataset>.pt
```

**Tên file được tạo bởi:** `utils/checkpoint_paths.py`
```python
ssl_checkpoint_name(method="swav", backbone=backbone_type, dataset=dataset_name)
# → "swav_<backbone>_encoder_pretrained_<dataset>.pt"

ssl_checkpoint_dir(output_root, method="swav", dataset=dataset_name, backbone=backbone_type, tag=tag)
# → "<output_root>/checkpoints/ssl_pretrain/swav/<dataset>/<backbone[_tag]>/"
```

### 2.3. Config tập trung — SwAVConfig

**File:** `config/methods/swav_config.py`
**Class:** `SwAVConfig`

```python
class SwAVConfig:
    # SwAV Hyperparameters
    FEATURE_DIM = 128           # Chiều encoder output (phải khớp backbone output)
    PROJECTION_DIM = 64         # Chiều projection head output (và prototype)
    NUM_PROTOTYPES = 45         # K — số cluster centers (45 = 5 lớp × 9 sub-patterns)
    TAU_S = 0.1                 # τ — softmax temperature cho prediction p
    EPSILON = 0.05              # ε — Sinkhorn temperature cho target assignment q

    # Training Configuration
    EPOCHS = 40
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5           # LR warmup; cũng là mốc bắt đầu checkpoint selection
    FREEZE_PROTOTYPES_EPOCHS = 1  # Số epoch đầu prototype bị đóng băng (grad=None)
    SINKHORN_ITERS = 3          # Số vòng lặp Sinkhorn (bài báo gốc dùng 3)

    # Giám sát sụp cụm
    MAX_ENTROPY_COLLAPSE_THRESHOLD = 0.3  # Cảnh báo khi entropy < 30% * ln(K)
```

> **Lưu ý FREEZE_PROTOTYPES_EPOCHS = 1** (bài báo gốc: ~313 iterations ≈ 1 epoch). Prototype đóng băng epoch 1, bắt đầu học từ epoch 2.

### 2.4. Dataset và DataLoader — dùng chung với TS-TCC

**File:** `datasets/contrastive_dataset.py`
**Class:** `ContrastiveDatasetWrapper`

SwAV dùng **cùng dataset wrapper** với TS-TCC — sinh cặp `(x_weak, x_strong)` dạng `(6, 128)` với `TS_TCC_Augmentation`.

**DataLoader settings trong SwAV trainer:**
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=len(dataset) >= batch_size,  # QUAN TRỌNG: Sinkhorn cần batch kích thước đủ lớn
    num_workers=use_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(use_workers > 0),
)
```

> **Lý do `drop_last`:** Sinkhorn ép mỗi prototype nhận ~B/K mẫu. Batch cuối nhỏ làm ràng buộc này mất ý nghĩa.

---

## Phần 3. Dữ liệu đầu vào và cấu trúc dữ liệu

### 3.1. Datasets — giống TS-TCC

| Key trong registry | File data |
|---|---|
| `uci_har` | `data/processed/uci_har/dataset_all.pt` |
| `motion_sense` | `data/processed/motionsense/dataset_all.pt` |
| `hhar_phone` | `data/processed/hhar/phone/dataset_all.pt` |
| `hhar_watch` | `data/processed/hhar/watch/dataset_all.pt` |

### 3.2. Pipeline tensor qua SwAV

```text
File .pt → samples: (N, 6, 128)
                        ↓  ContrastiveDatasetWrapper.__getitem__
x_raw: (6, 128)  →  x_weak: (6, 128)  +  x_strong: (6, 128)
                        ↓  DataLoader collate
Batch: x_weak (B, 6, 128)  +  x_strong (B, 6, 128)
                        ↓  SwAVModel.forward
p_w (B, D=64)  +  p_s (B, D=64)       [L2-normalized embeddings]
                        ↓  @ prototypes.T (K, D).T = (D, K)
cos_w (B, K=45)  +  cos_s (B, K=45)   [cosine similarity THÔ, chưa chia τ hay ε]
                        ↓  SwAVLoss.forward
q_w = Sinkhorn(cos_w, ε)  →  (B, K)   [target assignment, NO_GRAD]
q_s = Sinkhorn(cos_s, ε)  →  (B, K)   [target assignment, NO_GRAD]
p_w_pred = softmax(cos_w/τ)  →  (B, K) [prediction, HAS_GRAD]
p_s_pred = softmax(cos_s/τ)  →  (B, K) [prediction, HAS_GRAD]
loss = ½ [CE(q_s, p_w) + CE(q_w, p_s)]   scalar
entropy = cluster_usage_entropy(avg_prob) scalar [NO_GRAD, chỉ monitoring]
```

---

## Phần 4. Kiến trúc mô hình

### 4.1. SwAVModel

**File:** `models/ssl/swav/swav_model.py`
**Class:** `SwAVModel`

```python
class SwAVModel(nn.Module):
    def __init__(self, encoder, feature_dim, projection_dim=64, num_prototypes=45):
        self.encoder = encoder               # backbone (bất kỳ loại nào)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Gộp temporal nếu encoder trả (B, F, T')
        self.projection_head = ProjectionHead(
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=feature_dim,          # (B, 128) → (B, 128) → (B, 64)
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, projection_dim))
        nn.init.orthogonal_(self.prototypes)  # K=45 vectors trực giao nhau trong D=64 chiều
```

**Lưu ý `feature_dim`:** Phải khớp đầu ra thật của encoder. Với tất cả backbone trong dự án, encoder output 128 kênh (`feature_dim=128`).

**Method `forward_backbone(x)` — x (B,C,T) → (B, F):**
```python
def forward_backbone(self, x):
    feat = self.encoder(x)
    if feat.dim() == 2: return feat                      # encoder trả (B, F)
    if feat.dim() == 3: return self.pool(feat).flatten(1) # (B, F, T') → (B, F)
```

**Method `embed(x)` — x (B,C,T) → (B, D) L2-normalized:**
```python
def embed(self, x):
    z = self.forward_backbone(x)       # (B, F=128)
    p = self.projection_head(z)        # (B, D=64)
    return F.normalize(p, dim=-1, p=2) # (B, D), ||p||=1
```

**Method `forward(x_w, x_s)` — trả về `(cos_w, cos_s)` hai (B, K):**
```python
def forward(self, x_w, x_s):
    p_w = self.embed(x_w)                           # (B, D)
    p_s = self.embed(x_s)                           # (B, D)
    # Chuẩn hóa prototype TRONG forward (cách B — gradient-based)
    w = F.normalize(self.prototypes, dim=-1, p=2)   # (K, D)
    cos_w = p_w @ w.t()                             # (B, K), giá trị trong [-1, 1]
    cos_s = p_s @ w.t()                             # (B, K)
    return cos_w, cos_s
```

> **"Cách B" vs "Cách A":**
> - **Cách B (hiện tại):** `F.normalize(self.prototypes)` trong forward — gradient chạy qua bước normalization, tác động lên hướng prototype.
> - **Cách A (SwAV gốc):** in-place normalize prototype với `torch.no_grad()` trước mỗi forward iteration — prototype chỉ được cập nhật về hướng.
> - Cách B đơn giản hơn, ít lỗi hơn (cách A nếu quên normalize thì cos ra ngoài [-1,1] mà không báo lỗi).

### 4.2. ProjectionHead

**File:** `models/heads/projection.py`
**Class:** `ProjectionHead`

```python
class ProjectionHead(nn.Module):
    def __init__(self, feature_dim=128, projection_dim=64, hidden_dim=None):
        hidden_dim = hidden_dim if hidden_dim is not None else feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),    # 128 → 128
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)  # 128 → 64
        )

    def forward(self, x, normalize=True):
        z = self.net(x)
        if normalize:
            z = F.normalize(z, p=2, dim=1)         # L2-normalize
        return z
```

Output: `(B, 64)` — L2-normalized embedding.

> **Khởi tạo trong SwAVModel:** `ProjectionHead(feature_dim=128, projection_dim=64, hidden_dim=128)`.
> `ProjectionHead` được dùng chung giữa SwAV và các method khác có thể cần projection head.

### 4.3. Prototype Matrix

- **Shape:** `(K=45, D=64)` — `nn.Parameter`.
- **Khởi tạo:** `nn.init.orthogonal_()` — 45 vectors trực giao nhau trong không gian 64 chiều (K=45 < D=64, feasible).
- **Trong forward:** Normalize lại mỗi step (`F.normalize(self.prototypes, dim=-1)`).
- **Score computation:** `p @ w.T / τ` nhưng **SwAVModel chỉ trả cosine THÔ** (không chia τ). Việc chia τ được thực hiện bên trong `SwAVLoss`.

> **Lưu ý phân tách trách nhiệm:** `SwAVModel.forward()` trả cosine THÔ. `SwAVLoss.forward()` áp dụng τ cho softmax và ε cho Sinkhorn. **Không được trộn lẫn hai nhiệt độ này.**

### 4.4. So sánh kiến trúc: SwAV vs Prototype cũ

| Khía cạnh | Prototype cũ (`PrototypicalHARModel`) | SwAV hiện tại (`SwAVModel`) |
|---|---|---|
| Prototype normalization | `with torch.no_grad(): F.normalize(...)` trong forward — detached | `F.normalize(...)` trong forward — gradient chạy qua |
| Temperature trong forward | Chia τ trong forward (`scores = z @ proto.T / τ`) | Trả cosine THÔ; τ chỉ áp dụng trong loss |
| AdaptiveAvgPool1d | Manual `mean(dim=-1)` trong forward | `nn.AdaptiveAvgPool1d(1)` như attribute |
| Hai optimizer riêng | Có (opt_network + opt_proto) | Không — 1 optimizer với 2 nhóm param (decay/no_decay) |
| Prototype đóng băng | Không (update từ epoch 1) | Có (FREEZE_PROTOTYPES_EPOCHS = 1) |
| Gradient prototype | Có thể bị block do `no_grad` wrap trong forward | Luôn chạy đúng (gradient qua normalize trong forward) |

---

## Phần 5. Loss Function — SwAVLoss

**File:** `losses/swav_loss.py`
**Classes:** `SwAVLoss`, `sinkhorn()`, `cluster_usage_entropy()`

### 5.1. Hai nhiệt độ — KHÔNG được trộn lẫn

```text
ε (epsilon = 0.05) : CHỈ dùng trong Sinkhorn → exp(cos / ε)
τ (tau = 0.1)      : CHỈ dùng trong softmax  → softmax(cos / τ)
Cả hai đều nhận cosine THÔ từ SwAVModel.forward().
```

### 5.2. Sinkhorn-Knopp — sinh target assignment `q`

**Function:** `sinkhorn(cos, epsilon=0.05, n_iters=3)` — được đánh dấu `@torch.no_grad()`.

```python
@torch.no_grad()
def sinkhorn(cos: torch.Tensor, epsilon: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    # Bước 0: ổn định số học (trừ max toàn cục trước khi exp)
    cos = cos - cos.max()

    # Bước 1: ma trận "độ phù hợp" dương
    Q = torch.exp(cos / epsilon).t()   # (K, B); chuyển vị giống ký hiệu bài báo
    K, B = Q.shape

    # Bước 2: normalize tổng = 1
    Q /= Q.sum()

    # Bước 3: lặp normalize hàng/cột (3 lần)
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True)   # mỗi cụm tổng = 1
        Q /= K                             # → mỗi cụm tổng 1/K
        Q /= Q.sum(dim=0, keepdim=True)   # mỗi mẫu tổng = 1
        Q /= B                             # → mỗi mẫu tổng 1/B

    # Bước 4: đưa về phân phối theo mẫu
    Q *= B
    return Q.t()   # (B, K), mỗi hàng tổng = 1
```

**Mục đích Sinkhorn:**
- Ràng buộc "mỗi cụm nhận ~B/K mẫu" → chống mode collapse (không thể dồn tất cả vào 1 prototype).
- Output `q`: mỗi hàng là phân phối xác suất trên K prototype cho một mẫu.
- Với `n_iters=3`: xấp xỉ, tải mỗi cụm vẫn có thể lệch ~0.65–1.27 lần so với B/K.

### 5.3. Cluster usage entropy — giám sát sụp cụm

**Function:** `cluster_usage_entropy(prob)` — `@torch.no_grad()`.

```python
@torch.no_grad()
def cluster_usage_entropy(prob: torch.Tensor) -> torch.Tensor:
    avg = prob.mean(dim=0)                        # (K,) — tỉ lệ dùng mỗi cụm, TB trên batch
    return -(avg * torch.log(avg + 1e-8)).sum()   # scalar, trong [0, ln K]
```

Giá trị `ln(K) = ln(45) ≈ 3.81` là entropy tối đa (tất cả cụm dùng đều).

### 5.4. SwAVLoss.forward()

```python
class SwAVLoss(nn.Module):
    def __init__(self, tau=0.1, epsilon=0.05, sinkhorn_iters=3):
        self.tau = tau
        self.epsilon = epsilon
        self.sinkhorn_iters = sinkhorn_iters

    def forward(self, cos_w, cos_s):
        # (a) Target assignments: Sinkhorn, NO gradient
        q_w = sinkhorn(cos_w, self.epsilon, self.sinkhorn_iters)  # (B, K)
        q_s = sinkhorn(cos_s, self.epsilon, self.sinkhorn_iters)  # (B, K)

        # (b) Predictions: log_softmax(cos/τ), HAS gradient
        log_p_w = F.log_softmax(cos_w / self.tau, dim=1)   # (B, K)
        log_p_s = F.log_softmax(cos_s / self.tau, dim=1)   # (B, K)

        # (c) Swapped prediction loss [Eq. 1 bài báo]
        # View weak dự đoán mã của view strong, và ngược lại
        loss_w_predicts_s = -(q_s * log_p_w).sum(dim=1).mean()
        loss_s_predicts_w = -(q_w * log_p_s).sum(dim=1).mean()
        loss = 0.5 * (loss_w_predicts_s + loss_s_predicts_w)

        # (d) Monitor cluster entropy (NO gradient)
        entropy = cluster_usage_entropy(0.5 * (log_p_w.exp() + log_p_s.exp()))

        return loss, entropy
```

**Gradient flow:**
```
loss → log_p_w, log_p_s
     → cos_w, cos_s (thông qua log_softmax/τ)
     → p_w, p_s (L2-normalized embeddings)
     → projection_head → encoder
     → prototypes (thông qua cos = p @ w.T, w = normalize(prototypes))

q_w, q_s: KHÔNG nhận gradient (Sinkhorn chạy dưới @torch.no_grad)
```

### 5.5. Công thức toán học tóm tắt

$$\mathcal{L}_{\text{SwAV}} = -\frac{1}{2} \left[ \frac{1}{B}\sum_{i=1}^{B} \sum_{k=1}^{K} q_s^{(i,k)} \log p_w^{(i,k)} + \frac{1}{B}\sum_{i=1}^{B} \sum_{k=1}^{K} q_w^{(i,k)} \log p_s^{(i,k)} \right]$$

Trong đó:
- $q_w = \text{Sinkhorn}(\text{cos}_w / \varepsilon)$, $\varepsilon = 0.05$
- $p_w = \text{softmax}(\text{cos}_w / \tau)$, $\tau = 0.1$
- $K = 45$, $B$ là batch size

---

## Phần 6. Quy trình Self-Supervised Pretraining

### 6.1. Hàm chính — `train_swav_encoder()`

**File:** `engines/pretrain_ssl/swav/swav_trainer.py`

```python
def train_swav_encoder(
    data_path,          # Path đến dataset_all.pt
    dataset_name,       # Key dataset (vd "uci_har")
    backbone_type,      # "standard" | "tstcc" | "cnn_transformer" | "vit_1d"
    in_channels,        # 6
    seq_len,            # 128
    epochs,             # Từ CLI (default SwAVConfig.EPOCHS = 40)
    batch_size,         # Từ CLI (default SwAVConfig.BATCH_SIZE = 64)
    learning_rate,      # Từ CLI (default SwAVConfig.LEARNING_RATE = 3e-4)
    num_workers,
    device,
    measure_complexity=True,
    tag="",             # Tag phụ cho thư mục
):
```

**Checkpoint output directory** (sử dụng `utils/checkpoint_paths.py`):
```python
ckpt_dir = ssl_checkpoint_dir(get_output_root(), "swav", dataset_name, backbone_type, tag=tag)
# → <output_root>/checkpoints/ssl_pretrain/swav/<dataset>/<backbone[_tag]>/
checkpoint_name = ssl_checkpoint_name("swav", backbone_type, dataset_name)
# → "swav_<backbone>_encoder_pretrained_<dataset>.pt"
```

### 6.2. Khởi tạo model — `_build_model()`

```python
def _build_model(backbone_type, in_channels, device):
    return SwAVModel(
        encoder=build_encoder(backbone_type=backbone_type, in_channels=in_channels),
        feature_dim=SwAVConfig.FEATURE_DIM,          # 128
        projection_dim=SwAVConfig.PROJECTION_DIM,     # 64
        num_prototypes=SwAVConfig.NUM_PROTOTYPES,     # 45
    ).to(device)
```

### 6.3. Optimizer — `_build_optimizer()` với 2 nhóm param

```python
def _build_optimizer(model, learning_rate):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name == "prototypes":
            no_decay.append(param)   # bias, norm-layers, prototypes
        else:
            decay.append(param)       # conv/linear weights
    return AdamW(
        [{"params": decay, "weight_decay": SwAVConfig.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )
```

**Lý do không weight_decay cho prototype:** Prototype chỉ có ý nghĩa theo hướng (đã L2-normalize trong forward), co về 0 là vô nghĩa.

### 6.4. Learning Rate — `_lr_at_epoch()`

```python
def _lr_at_epoch(epoch, total_epochs, base_lr, warmup_epochs, min_lr=1e-6):
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)      # linear warmup
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
```

**Với SwAVConfig defaults (`base_lr=3e-4`, `warmup_epochs=5`, `total_epochs=40`, `min_lr=1e-6`):**
- Epoch 1–5: lr linear từ `6e-5` đến `3e-4`.
- Epoch 6–40: cosine decay từ `3e-4` về `1e-6`.

LR được cập nhật **mỗi epoch** (không phải mỗi iteration như SwAV gốc).

### 6.5. Một training iteration — `_train_one_epoch()`

```python
def _train_one_epoch(model, criterion, loader, optimizer, device, freeze_prototypes=False):
    model.train()
    total_loss = total_entropy = 0.0

    for x_w, x_s in loader:
        x_w = x_w.to(device, non_blocking=True)
        x_s = x_s.to(device, non_blocking=True)

        optimizer.zero_grad()

        # (3) FORWARD — trả cosine THÔ (chưa chia τ hay ε)
        cos_w, cos_s = model(x_w, x_s)   # mỗi cái (B, K)

        # (4) SINKHORN + LOSS
        loss, entropy = criterion(cos_w, cos_s)

        # (5) BACKWARD — gradient chạy về cả embedding và prototype
        loss.backward()

        # (6) ĐÓNG BĂNG PROTOTYPE nếu cần — ĐẶT GRADIENT = NONE SAU backward
        if freeze_prototypes:
            model.prototypes.grad = None   # Optimizer bỏ qua param không có grad

        # (7) CẬP NHẬT
        optimizer.step()

        total_loss    += loss.item()
        total_entropy += entropy.item()

    n = len(loader)
    return {"loss_total": total_loss / n, "entropy": total_entropy / n}
```

**Thứ tự quan trọng:** `loss.backward()` TRƯỚC `model.prototypes.grad = None` TRƯỚC `optimizer.step()`.

### 6.6. Checkpoint selection (sau warmup)

```python
# Chỉ xét SAU warmup (loss trong warmup chưa đại diện)
if epoch > SwAVConfig.WARMUP_EPOCHS and row["loss_total"] < best_loss:
    best_loss, saved = row["loss_total"], True
    torch.save(model.encoder.state_dict(), checkpoint_path)

# Fallback nếu epochs <= warmup (không có epoch nào được lưu)
if not saved:
    best_loss = history[-1]["loss_total"]
    torch.save(model.encoder.state_dict(), checkpoint_path)
```

**Chỉ lưu `encoder.state_dict()`** — prototype và projection head bị loại bỏ.

### 6.7. Giám sát sụp cụm

```python
max_entropy = math.log(SwAVConfig.NUM_PROTOTYPES)      # ln(45) ≈ 3.81
collapse_floor = SwAVConfig.MAX_ENTROPY_COLLAPSE_THRESHOLD * max_entropy  # 0.3 * 3.81 ≈ 1.14

if row["entropy"] < collapse_floor:
    print(f"⚠️  Cảnh báo sụp cụm: Entropy={row['entropy']:.2f} < {collapse_floor:.2f} tại epoch {epoch}")
```

**Không có early stopping** — chỉ in cảnh báo. Entropy được log vào CSV.

### 6.8. Outputs sau pretraining

```text
checkpoints/ssl_pretrain/swav/<dataset_name>/<backbone_type>/
    ├── swav_<backbone>_encoder_pretrained_<dataset>.pt  ← encoder.state_dict()
    ├── loss_history.csv        ← epoch, loss_total, entropy, lr
    ├── loss_curve.png          ← 2 subplots: SwAV Loss + Cluster Entropy
    ├── config.json             ← cấu hình huấn luyện đầy đủ
    └── complexity.json         ← FLOPs, params của backbone
```

---

## Phần 7. Chi tiết so sánh với SwAV gốc

### 7.1. Bảng so sánh

| Khía cạnh | SwAV gốc (facebookresearch) | Implementation trong repo |
|---|---|---|
| Input | Ảnh (C, H, W) | Sensor signal (6, 128) |
| Augmentation | Random crop, flip, color jitter | TS_TCC_Augmentation (scaling, permutation, jitter) |
| Multi-crop | Có (nhiều resolution) | Không — chỉ 2 views |
| Queue | Có | Không |
| Optimizer | SGD + LARC | AdamW (2 nhóm param) |
| LR schedule | Cosine theo iteration | Linear warmup + Cosine theo epoch |
| Prototype normalization | In-place, trước forward (cách A) | Trong forward, gradient-based (cách B) |
| Freeze prototypes | ~313 iterations | FREEZE_PROTOTYPES_EPOCHS = 1 epoch |
| Checkpoint selection | Không rõ | Min train loss SAU warmup |
| Cơ chế anti-collapse | Sinkhorn equipartition | Sinkhorn + entropy monitoring |

### 7.2. Global Average Pooling

Encoder output 3D `(B, F, T')` được GAP về `(B, F)` qua `nn.AdaptiveAvgPool1d(1)`:

```python
# SwAVModel.forward_backbone()
if feat.dim() == 3:
    return self.pool(feat).flatten(1)   # (B, F, T') → (B, F)
```

Khác với TS-TCC (cần sequence dimension cho TC module), SwAV yêu cầu vector embedding cố định chiều.

### 7.3. Orthogonal Initialization của Prototype

`nn.init.orthogonal_` với shape `(45, 64)`:
- Tạo ma trận $45 \times 64$ với các hàng trực giao nhau.
- $K=45 < D=64$ → có thể tạo 45 vector trực giao trong không gian 64 chiều.
- Nếu $K > D$: `orthogonal_` sẽ không tạo được K vector trực giao độc lập — cần kiểm tra khi thay đổi NUM_PROTOTYPES.

---

## Phần 8. Learning Rate Scheduling chi tiết

### 8.1. Linear Warmup + Cosine Annealing

```python
# _lr_at_epoch() trong swav_trainer.py
# 5 epochs warmup tuyến tính từ 0 đến base_lr
# Sau warmup: cosine decay từ base_lr về min_lr=1e-6
```

**Cụ thể với defaults (`base_lr=3e-4`, `warmup_epochs=5`, `total_epochs=40`):**
- Epoch 1: lr = 6e-5
- Epoch 2: lr = 1.2e-4
- Epoch 3: lr = 1.8e-4
- Epoch 4: lr = 2.4e-4
- Epoch 5: lr = 3e-4
- Epoch 6–40: cosine decay từ ~3e-4 về ~1e-6

### 8.2. LR áp dụng cho cả hai nhóm param

```python
# Vòng lặp epoch trong train_swav_encoder()
lr = _lr_at_epoch(epoch, epochs, learning_rate, SwAVConfig.WARMUP_EPOCHS)
for group in optimizer.param_groups:
    group["lr"] = lr   # Cả nhóm decay và no_decay đều dùng cùng lr
```

> **Khác với SwAV gốc:** SwAV gốc update LR theo iteration (batch), còn đây update theo epoch.

---

## Phần 9. Augmentation — dùng chung với TS-TCC

**File:** `../../models/ssl/tstcc/augmentations.py`
**Class:** `TS_TCC_Augmentation`

Cùng augmentation strategy với TS-TCC:
- **Weak:** `scaling(sigma=1.1, loc=2.0)`
- **Strong:** `permutation(max_seg=5)` → `jitter(sigma=0.8)`

---

## Phần 10. Downstream Task — giống TS-TCC

Downstream pipeline **giống hoàn toàn** với TS-TCC (xem TS_TCC.md Phần 10).

**Checkpoint pattern khi load:**
```python
# Checkpoint pattern
"swav_<backbone>_encoder_pretrained_<dataset>.pt"

# Ví dụ
"checkpoints/ssl_pretrain/swav/motion_sense/standard/swav_standard_encoder_pretrained_motion_sense.pt"
```

**Loading vào HARClassifier:**
```python
model.encoder.load_state_dict(encoder_dict, strict=True)
```

---

## Phần 11. Hyperparameters Đầy đủ

| Hyperparameter | Giá trị | Nguồn |
|---|---|---|
| `backbone` | `standard` (default) | CLI `--backbone` |
| `epochs` | 40 | `SwAVConfig.EPOCHS` |
| `batch_size` | 64 | `SwAVConfig.BATCH_SIZE` |
| `learning_rate` (base) | 3e-4 | `SwAVConfig.LEARNING_RATE` / CLI `--lr` |
| `weight_decay` | 1e-4 | `SwAVConfig.WEIGHT_DECAY` |
| `warmup_epochs` | 5 | `SwAVConfig.WARMUP_EPOCHS` |
| `freeze_prototypes_epochs` | 1 | `SwAVConfig.FREEZE_PROTOTYPES_EPOCHS` |
| `min_lr` | 1e-6 | Hard-coded trong `swav_trainer.py` |
| `num_prototypes` (K) | 45 | `SwAVConfig.NUM_PROTOTYPES` |
| `feature_dim` (F) | 128 | `SwAVConfig.FEATURE_DIM` |
| `projection_dim` (D) | 64 | `SwAVConfig.PROJECTION_DIM` |
| `tau` (τ) | 0.1 | `SwAVConfig.TAU_S` |
| `epsilon` (ε) | 0.05 | `SwAVConfig.EPSILON` |
| `sinkhorn_iters` | 3 | `SwAVConfig.SINKHORN_ITERS` |
| `max_entropy_collapse_threshold` | 0.3 | `SwAVConfig.MAX_ENTROPY_COLLAPSE_THRESHOLD` |
| `in_channels` | 6 | Dataset config |
| `window_size` | 128 | Dataset config |
| `stride` | 64 | Dataset config (overlap 50%) |
| Augmentation (weak) | scaling σ=1.1, loc=2.0 | `TS_TCC_Augmentation` |
| Augmentation (strong) | permutation(max_seg=5) + jitter(σ=0.8) | `TS_TCC_Augmentation` |
| Optimizer | AdamW (2 nhóm: decay/no_decay) | `_build_optimizer()` |
| LR Scheduler | Linear warmup + Cosine annealing | `_lr_at_epoch()` (cập nhật theo epoch) |
| Gradient clipping | Không có | Khác với TS-TCC |
| Checkpoint criterion | Min train loss SAU warmup | `swav_trainer.py` |
| Prototype init | Orthogonal | `nn.init.orthogonal_` |
| Prototype normalization | Trong forward, gradient-based | "Cách B" |

---

## Phần 12. Evaluation Metrics và Output

**Giống hoàn toàn với TS-TCC** (xem TS_TCC.md Phần 12).

**Output directory:**
```text
outputs_evaluation/cross_k_shot/swav/<backbone>/<src>_to_<tgt>/
    ├── experiment_metadata.json
    └── <k>_shot/
        ├── linear_probing/
        └── full_finetuning/
```

---

## Phần 13. Cài đặt và Chạy Code

### 13.1. Yêu cầu môi trường

```
Python >= 3.11
torch >= 2.0.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
thop             # optional, cho complexity measurement
# Không cần einops (SwAV không dùng attention.py)
```

### 13.2. Chạy SSL Pretraining

```bash
# Test nhanh 1 epoch
uv run python scripts/02_pretrain_ssl/run_pretrain_swav.py \
  --datasets uci_har --backbone standard --epochs 1 --device cpu

# Chạy thật đủ 4 dataset
uv run python scripts/02_pretrain_ssl/run_pretrain_swav.py \
  --datasets uci_har motion_sense hhar_phone hhar_watch \
  --backbone standard --epochs 40 --batch_size 64 --device cuda

# Với tag để phân biệt lần chạy
uv run python scripts/02_pretrain_ssl/run_pretrain_swav.py \
  --datasets uci_har --backbone standard --epochs 40 --tag "bs128" --batch_size 128
```

### 13.3. Chạy K-shot cross-domain evaluation

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method swav \
  --backbone standard \
  --k_shots 10 20 50 100 \
  --seeds 42 100 2024 7 99 \
  --epochs 40 \
  --pairs motion_sense:uci_har uci_har:motion_sense
```

### 13.4. Checklist trước khi chạy thật

```bash
# 1. Data tồn tại
ls -la data/processed/uci_har/dataset_all.pt

# 2. Test 1 epoch không lỗi
uv run python scripts/02_pretrain_ssl/run_pretrain_swav.py \
  --datasets uci_har --epochs 1 --device cpu

# 3. Xem kết quả
cat checkpoints/ssl_pretrain/swav/uci_har/standard/loss_history.csv
# Cột entropy: gần 3.81 → tốt; < 1.14 → cảnh báo collapse

# 4. Kiểm tra prototype gradient (debugging)
# Chạy 1 step, in model.prototypes.grad — nếu None sau backward thì có vấn đề
```

### 13.5. Self-test của SwAVLoss

```bash
uv run python losses/swav_loss.py
# Chạy 3 test cases và in kết quả
```

---

## Phần 14. Tóm tắt kiến trúc (Sơ đồ tổng quát)

```
[x_weak (B,6,128)] ──────────────────────────────────────┐
                                                          ├──→ SwAVModel
[x_strong (B,6,128)] ────────────────────────────────────┘
                                ↓
              [Encoder Backbone (any: standard/cnn_transformer/tstcc/vit_1d)]
                                ↓
                     [AdaptiveAvgPool1d(1)]  ← chỉ cần nếu encoder trả (B,F,T')
                      h_w (B,128)  +  h_s (B,128)
                                ↓
                    [ProjectionHead]
            Linear(128→128) → BN → ReLU → Linear(128→64) → L2-norm
                      p_w (B,64)  +  p_s (B,64)
                                ↓
              [Prototype Matrix: (K=45, D=64), L2-normalized trong forward]
              cos_w = p_w @ proto.T  → (B, 45)   [cosine THÔ, không chia τ]
              cos_s = p_s @ proto.T  → (B, 45)
                                ↓
                         [SwAVLoss.forward]
              q_w = Sinkhorn(cos_w, ε=0.05)   → (B, 45) [NO_GRAD]
              q_s = Sinkhorn(cos_s, ε=0.05)   → (B, 45) [NO_GRAD]
              log_p_w = log_softmax(cos_w/τ)  → (B, 45) [HAS_GRAD]
              log_p_s = log_softmax(cos_s/τ)  → (B, 45) [HAS_GRAD]
              L = ½ · [-mean(q_s·log_p_w) - mean(q_w·log_p_s)]
              entropy = H(avg_prob)                     [NO_GRAD, monitoring]
                                ↓
              [_build_optimizer: AdamW với 2 nhóm param]
              decay group  ← gradient(encoder weights, projection weights)
              no_decay group ← gradient(prototype, bias, norm-layers)
              [Nếu freeze_prototypes: model.prototypes.grad = None trước step()]
                                ↓
              [Checkpoint] encoder.state_dict() lưu khi loss < best, epoch > WARMUP_EPOCHS
```

---

## Phần 15. Tài liệu Tham khảo

**Bài báo gốc SwAV:**
- Caron, M., et al. "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments." NeurIPS 2020.
- Repository gốc: `facebookresearch/swav`.

**Implementation trong repository:**
- `models/ssl/swav/swav_model.py` — `SwAVModel` (backbone + projection + prototypes)
- `losses/swav_loss.py` — `SwAVLoss`, `sinkhorn()`, `cluster_usage_entropy()`
- `engines/pretrain_ssl/swav/swav_trainer.py` — `train_swav_encoder()`, `_build_optimizer()`, `_lr_at_epoch()`
- `config/methods/swav_config.py` — `SwAVConfig` (tất cả hyperparameters)
- `models/heads/projection.py` — `ProjectionHead` (dùng chung)
- `utils/checkpoint_paths.py` — `ssl_checkpoint_dir()`, `ssl_checkpoint_name()`
- `scripts/02_pretrain_ssl/run_pretrain_swav.py` — Entry point CLI
