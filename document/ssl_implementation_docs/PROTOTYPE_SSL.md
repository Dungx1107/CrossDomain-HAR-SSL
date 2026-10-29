# Tài liệu Kỹ thuật: Prototype-based SSL (SwAV-Inspired Prototypical HAR)

> **Mục đích tài liệu:** Mô tả chính xác implementation của phương pháp Prototype-based SSL trong repository CrossDomain-HAR-SSL dựa trên phân tích mã nguồn thực tế.  
> **Phiên bản phân tích:** Dựa trên snapshot code tại thời điểm viết tài liệu.  
> **Tài liệu này dành cho:** AI review code — mọi thông tin đều dẫn chiếu trực tiếp đến file và function tương ứng.

---

## Phần 1. Tổng quan

### 1.1. Mục tiêu trong dự án

Prototype-based SSL là phương pháp học tự giám sát thứ hai trong dự án, được lấy cảm hứng từ **SwAV (Swapping Assignments between Views)**. Mục tiêu là học biểu diễn đặc trưng bằng cách gán mỗi sample vào một trong $K$ **prototype vectors** (cluster centers) và huấn luyện mô hình dự đoán soft assignment của view này từ view kia (và ngược lại) thông qua thuật toán **Sinkhorn-Knopp optimal transport**.

Phương pháp này được thiết kế để:
1. Học biểu diễn có cấu trúc phân cụm ngữ nghĩa (prototype là "trung tâm đặc trưng" của các pattern hoạt động).
2. Tránh collapse thông qua Sinkhorn-Knopp constraints (mỗi prototype phải được gán đều, không phải tất cả dồn vào một prototype).

Trong dự án, sau khi pretrain, chỉ **encoder backbone** được giữ lại (prototype và projection head bị bỏ) để dùng trong downstream fine-tuning/linear probing.

### 1.2. Lý thuyết gốc

**SwAV (Caron et al., NeurIPS 2020):**
- Không so sánh trực tiếp các embedding của hai views (như NT-Xent).
- Thay vào đó, gán mỗi view vào các prototype (cluster assignment) và dạy view này tái tạo assignment của view kia.
- Sinkhorn-Knopp chuẩn hóa soft assignments thành phân phối đều để tránh mode collapse.

**Sự khác biệt với SwAV gốc trong repository này:**
- Dùng cho dữ liệu cảm biến quán tính 1D thay vì ảnh 2D.
- Prototype được khởi tạo bằng `nn.init.orthogonal_` thay vì random.
- Hai optimizer riêng biệt cho network và prototype.
- Có cơ chế monitor entropy để phát hiện cluster collapse.

### 1.3. Vai trò trong pipeline tổng thể

```text
SSL Pretraining (source domain)         → checkpoints/ssl_pretrain/prototype/<source>/<backbone>/
        |
        ↓  encoder.state_dict() được lưu (prototype & projection head bị loại bỏ)
K-shot / Fraction Cross-Domain          → run_few_shot.py / run_cross_domain.py
        |
        ↓  load encoder, freeze hoặc finetune
Linear Probing / Full Fine-tuning       → engines/transfer/finetune_trainer.py
        |
        ↓
Đánh giá trên target test set           → outputs_evaluation/cross_k_shot/prototype/
```

### 1.4. Các thành phần code chính

| Thành phần | File | Mô tả |
|---|---|---|
| ContrastiveDatasetWrapper | `datasets/contrastive_dataset.py` | Dataset sinh (x_weak, x_strong) |
| Encoder backbone | `models/encoders/builder.py` | Factory — bất kỳ backbone nào |
| ProjectionHead | `models/heads/projection.py` | MLP 2 lớp + L2 norm |
| PrototypicalHARModel | `models/ssl/prototype/cluster_model.py` | Wrapper: encoder + projection + prototypes |
| SwAVPrototypeLoss | `losses/swav_loss.py` | Loss SwAV + Sinkhorn-Knopp |
| Pretrain script | `scripts/02_pretrain_ssl/run_pretrain_prototype.py` | CLI và training loop |
| Downstream | `engines/transfer/finetune_trainer.py` | Dùng chung với TS-TCC và CrossHAR |

---

## Phần 2. Cấu trúc mã nguồn liên quan

### 2.1. Entry point — script pretrain

**File:** `scripts/02_pretrain_ssl/run_pretrain_prototype.py`
- **Chứa toàn bộ training loop** (không có engine module riêng như TS-TCC).
- **Nhận argument:** `--backbone`, `--epochs`, `--batch_size`, `--datasets`, `--lr`, `--num_prototypes`.
- **Default values:**
  - `backbone`: `standard`
  - `epochs`: 40
  - `batch_size`: 64
  - `datasets`: `["uci_har", "motionsense"]`
  - `lr`: 3e-4
  - `num_prototypes`: 45

> **Lưu ý thiết kế:** Training loop nằm trong `scripts/` thay vì `engines/` — đây là điểm khác biệt về cấu trúc so với TS-TCC. Prototype pretraining là self-contained trong `run_pretrain_prototype.py`.

### 2.2. Dataset và DataLoader

**File:** `datasets/contrastive_dataset.py`  
**Class:** `ContrastiveDatasetWrapper` — **dùng chung với TS-TCC**.

> **Điểm quan trọng:** Prototype SSL dùng chính xác cùng dataset wrapper với TS-TCC — sinh cặp `(x_weak, x_strong)` dạng `(6, 128)` với `TS_TCC_Augmentation` (jitter + scaling + permutation). Đây là cùng augmentation strategy.

### 2.3. Encoder backbone

**File:** `models/encoders/builder.py`  
**Function:** `build_encoder(backbone_type, in_channels)`

Prototype SSL có thể dùng bất kỳ backbone nào: `tstcc`, `standard`, `cnn_transformer`, `vit_1d`. Default trong script là `standard`.

**Đặc điểm quan trọng:** Sau encoder, cần có **Global Average Pooling** để ra vector `(B, D)`. Điều này được xử lý trong `PrototypicalHARModel.forward()`.

### 2.4. Projection Head

**File:** `models/heads/projection.py`  
**Class:** `ProjectionHead(input_dim=128, output_dim=64)`

```python
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),   # 128 → 128
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),  # 128 → 64
        )
    
    def forward(self, x):
        z = self.net(x)                          # (B, 64)
        z_norm = F.normalize(z, dim=-1)          # L2 normalize
        return z_norm
```

Output: `(B, 64)` — vector L2-normalized.

### 2.5. Core SSL model — PrototypicalHARModel

**File:** `models/ssl/prototype/cluster_model.py`  
**Class:** `PrototypicalHARModel`

### 2.6. Loss function — SwAVPrototypeLoss

**File:** `losses/swav_loss.py`  
**Class:** `SwAVPrototypeLoss`

### 2.7. Downstream (dùng chung)

Giống TS-TCC: `engines/transfer/finetune_trainer.py`, `models/har_classifier.py`, `models/heads/classifier.py`.

---

## Phần 3. Dữ liệu đầu vào và cấu trúc dữ liệu

### 3.1. Dataset giống TS-TCC

Cùng format và cùng set datasets: UCI-HAR, MotionSense, HHAR (phone/watch).  
Cùng dùng `dataset_all.pt` cho pretraining.  
Định dạng file `.pt`: `{"samples": (N, 6, 128), "labels": (N,), "subjects": (N,)}`.

### 3.2. Pipeline tensor

```text
File .pt → samples: (N, 6, 128)
                        ↓  ContrastiveDatasetWrapper.__getitem__
x_raw: (6, 128)  →  x_weak: (6, 128)  +  x_strong: (6, 128)
                        ↓  DataLoader collate
Batch: x_weak (B, 6, 128)  +  x_strong (B, 6, 128)
                        ↓  PrototypicalHARModel.forward
z_w (B, D, L)  [raw encoder output, L phụ thuộc backbone]
        ↓  GlobalAveragePooling [torch.mean(z_w, dim=-1)]
h_w (B, D)
        ↓  ProjectionHead + L2 Normalize
z_w_norm (B, 64)
        ↓  matmul với prototypes (K, 64)
scores_w (B, K)  ← similarity scores
                        ↓  SwAVPrototypeLoss.forward
                       loss: scalar
```

---

## Phần 4. Kiến trúc mô hình

### 4.1. PrototypicalHARModel

**File:** `models/ssl/prototype/cluster_model.py`  
**Class:** `PrototypicalHARModel`

```python
class PrototypicalHARModel(nn.Module):
    def __init__(self, encoder, feature_dim=128, projection_dim=64,
                 num_prototypes=45, temperature=0.1):
        super().__init__()
        self.encoder = encoder                               # Backbone (bất kỳ loại nào)
        self.proj_head = ProjectionHead(feature_dim, projection_dim)  # MLP + L2 norm
        
        # Prototype matrix: mỗi hàng là một prototype vector (đã L2-normalized)
        # Khởi tạo bằng orthogonal_ để các prototype ban đầu đa dạng nhau
        self.prototypes = nn.Parameter(
            torch.empty(num_prototypes, projection_dim)  # (K=45, 64)
        )
        nn.init.orthogonal_(self.prototypes)
        
        self.temperature = temperature  # τ_s = 0.1 cho softmax assignment
```

**`forward()` method:**

```python
def forward(self, x_w, x_s):
    # Bước 1: Encode cả hai view
    f_w = self.encoder(x_w)   # (B, D, L) — L phụ thuộc backbone
    f_s = self.encoder(x_s)   # (B, D, L)
    
    # Bước 2: Global Average Pooling (nếu tensor 3D)
    if f_w.dim() == 3:
        f_w = f_w.mean(dim=-1)   # (B, D)
    if f_s.dim() == 3:
        f_s = f_s.mean(dim=-1)   # (B, D)
    
    # Bước 3: Project và L2-normalize
    z_w = self.proj_head(f_w)   # (B, 64) — đã L2-normalized
    z_s = self.proj_head(f_s)   # (B, 64) — đã L2-normalized
    
    # Bước 4: Chuẩn hóa prototype vectors (detach, không flow gradient)
    with torch.no_grad():
        proto_norm = F.normalize(self.prototypes, dim=-1, p=2)  # (K=45, 64)
    
    # Bước 5: Tính cosine similarity scores
    scores_w = torch.mm(z_w, proto_norm.T) / self.temperature   # (B, K)
    scores_s = torch.mm(z_s, proto_norm.T) / self.temperature   # (B, K)
    
    return scores_w, scores_s
```

**Tensor shapes (với `standard` backbone, feature_dim=128):**

```text
x_w: (B, 6, 128)
    → encoder (StandardSensorEncoder1D) → (B, 128, 8)   [4 blocks, thay vì 18 của TSTCCEncoder]
    → mean(dim=-1) → (B, 128)
    → ProjectionHead → Linear(128→128) → BN → ReLU → Linear(128→64) → (B, 64)
    → F.normalize → (B, 64)  [L2 normalized]
    → mm with prototypes (45, 64).T → / temperature → (B, 45)
```

### 4.2. Prototype Matrix

- Shape: `(K=45, 64)` — `nn.Parameter`.
- **Khởi tạo:** `nn.init.orthogonal_(self.prototypes)`.
  - Phương thức `orthogonal_` tạo ma trận sao cho các hàng/cột trực giao nhau. Với shape `(45, 64)`, 45 prototype vectors trong không gian 64 chiều được khởi tạo trực giao nhau → đảm bảo sự đa dạng ban đầu.
- **Trong forward:** Prototype được **L2-normalized lại** mỗi step (`F.normalize(self.prototypes, dim=-1)`) để chỉ lấy hướng, không phải magnitude. Điều này đảm bảo scores là cosine similarity.
- **Detach:** `with torch.no_grad()` khi normalize prototype trong forward — prototype gradient chỉ từ optimizer riêng (`opt_proto`).

> **Thiết kế quan trọng:** Có **hai optimizer riêng biệt**:
> - `opt_network` = AdamW cho toàn bộ network (encoder + projection head) ngoại trừ prototypes.
> - `opt_proto` = AdamW cho `model.prototypes` (prototype matrix).
> Điều này được thực hiện trong `run_pretrain_prototype.py`:
> ```python
> network_params = [p for n, p in model.named_parameters() if n != "prototypes"]
> opt_network = torch.optim.AdamW(network_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
> opt_proto = torch.optim.AdamW([model.prototypes], lr=args.lr, weight_decay=WEIGHT_DECAY)
> ```

### 4.3. Score computation

$$\text{scores}_{w}[i, k] = \frac{\mathbf{z}_w^{(i)} \cdot \hat{\mathbf{p}}_k}{\tau_s}$$

Trong đó:
- $\mathbf{z}_w^{(i)}$: projection của sample $i$ từ weak view — đã L2-normalized, shape `(64,)`
- $\hat{\mathbf{p}}_k = \text{normalize}(\mathbf{p}_k)$: prototype thứ $k$ — đã L2-normalized, shape `(64,)`
- $\tau_s = 0.1$: softmax temperature
- Kết quả là scaled cosine similarity, không có bias

Với L2 normalization cả hai phía: $\text{scores}_{w}[i,k] = \frac{\cos(\mathbf{z}_w^{(i)}, \mathbf{p}_k)}{\tau_s}$

---

## Phần 5. Loss Function — SwAV Prototype Loss

### 5.1. Cơ chế SwAV

**File:** `losses/swav_loss.py`  
**Class:** `SwAVPrototypeLoss`

SwAV loss có cơ chế "swap" — dạy mỗi view dự đoán soft prototype assignment của view kia:

$$\mathcal{L}_{\text{SwAV}} = -\frac{1}{2} \left( \mathbf{q}_w \cdot \log \mathbf{p}_s + \mathbf{q}_s \cdot \log \mathbf{p}_w \right)$$

Trong đó:
- $\mathbf{p}_w = \text{softmax}(\text{scores}_w / \tau_s)$: soft assignment của weak view
- $\mathbf{p}_s = \text{softmax}(\text{scores}_s / \tau_s)$: soft assignment của strong view
- $\mathbf{q}_w, \mathbf{q}_s$: **target assignments** — được tính bởi Sinkhorn-Knopp

### 5.2. Sinkhorn-Knopp Optimal Transport

```python
# losses/swav_loss.py
def sinkhorn_knopp(self, scores: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """
    Input: scores (B, K) — raw logits
    Output: Q (B, K) — soft assignment, hàng-normalized, cột-normalized
    """
    Q = torch.exp(scores / self.epsilon).T   # (K, B)
    Q = Q / Q.sum()                          # normalize tổng = 1
    
    B = scores.shape[0]
    K = scores.shape[1]
    
    for _ in range(n_iters):
        # Normalize theo hàng (mỗi prototype phải được gán đều)
        Q = Q / Q.sum(dim=1, keepdim=True)   # (K, B)
        Q = Q / K
        
        # Normalize theo cột (mỗi sample phải thuộc về đúng một prototype)
        Q = Q / Q.sum(dim=0, keepdim=True)   # (K, B)
        Q = Q / B
    
    return (Q * B).T   # (B, K) — scale lại để có giá trị có ý nghĩa
```

**Mục đích của Sinkhorn-Knopp:**
- Mỗi prototype được gán trung bình $B/K$ samples (đều nhau) → tránh mode collapse (tất cả samples gán vào một prototype).
- Mỗi sample tổng soft assignment = 1 → phân phối xác suất hợp lệ.
- `n_iters=3`: 3 vòng lặp normalize xen kẽ (hàng rồi cột).
- `epsilon=0.05`: Sinkhorn temperature — nhỏ hơn → target assignments sắc nét (sharp) hơn.

**Tham số `epsilon`:**
- `epsilon=0.05` trong script (dòng 107: `EPSILON = 0.05`)
- Đây là tham số riêng của Sinkhorn, khác với `temperature=0.1` (τ_s) dùng cho softmax score computation trong PrototypicalHARModel.

### 5.3. Forward pass của SwAVPrototypeLoss

```python
def forward(self, scores_w, scores_s, return_q=False):
    """
    scores_w: (B, K) — similarity scores của weak view
    scores_s: (B, K) — similarity scores của strong view
    """
    # Tính target soft assignments qua Sinkhorn-Knopp
    with torch.no_grad():
        q_w = self.sinkhorn_knopp(scores_w)   # (B, K) — target của strong
        q_s = self.sinkhorn_knopp(scores_s)   # (B, K) — target của weak
    
    # Tính soft assignment predictions
    p_w = F.softmax(scores_w, dim=-1)   # (B, K)
    p_s = F.softmax(scores_s, dim=-1)   # (B, K)
    
    # Cross-entropy loss (swap): dự đoán assignment của view này từ view kia
    loss_w = -torch.mean(torch.sum(q_s * torch.log(p_w + 1e-8), dim=-1))
    loss_s = -torch.mean(torch.sum(q_w * torch.log(p_s + 1e-8), dim=-1))
    
    loss = (loss_w + loss_s) / 2.0
    
    if return_q:
        return loss, q_w   # q_w được return để monitor entropy
    return loss
```

> **Lưu ý implementation:** `q_w` và `q_s` được tính trong `torch.no_grad()` — Sinkhorn-Knopp không flow gradient, chỉ là target assignment. Gradient chỉ chạy qua `p_w` và `p_s`.

### 5.4. Tổng kết toán học

$$\mathcal{L} = -\frac{1}{2B} \sum_{i=1}^{B} \left( \sum_{k=1}^{K} q_s^{(i,k)} \log p_w^{(i,k)} + \sum_{k=1}^{K} q_w^{(i,k)} \log p_s^{(i,k)} \right)$$

Trong đó:
- $q_w = \text{SinkhornKnopp}(\text{scores}_w / \varepsilon)$ — target assignment từ weak view
- $q_s = \text{SinkhornKnopp}(\text{scores}_s / \varepsilon)$ — target assignment từ strong view  
- $p_w = \text{softmax}(\text{scores}_w)$ — predicted assignment cho weak view
- $p_s = \text{softmax}(\text{scores}_s)$ — predicted assignment cho strong view
- $\varepsilon = 0.05$, $\tau_s = 0.1$, $K = 45$

---

## Phần 6. Quy trình Self-Supervised Pretraining

### 6.1. Khởi tạo (từ `run_pretrain_prototype.py`, hàm `train_prototype_single_domain`)

```python
# Constants
FEATURE_DIM = 128       # Feature dim từ encoder
PROJECTION_DIM = 64     # Output dim của ProjectionHead
TAU_S = 0.1             # Softmax temperature cho score computation
EPSILON = 0.05          # Sinkhorn temperature
WARMUP_EPOCHS = 5       # Epochs warmup trước khi lưu checkpoint

# Build model
backbone = build_encoder(backbone_type=backbone_type, in_channels=in_channels)
model = PrototypicalHARModel(
    encoder=backbone,
    feature_dim=FEATURE_DIM,        # 128
    projection_dim=PROJECTION_DIM,  # 64
    num_prototypes=args.num_prototypes,  # 45 mặc định
    temperature=TAU_S               # 0.1
)

criterion = SwAVPrototypeLoss(epsilon=EPSILON)  # epsilon=0.05
```

### 6.2. Hai Optimizer riêng biệt

```python
# Tách network params và prototype params
network_params = [p for n, p in model.named_parameters() if n != "prototypes"]

opt_network = torch.optim.AdamW(network_params, lr=args.lr, weight_decay=WEIGHT_DECAY)
# WEIGHT_DECAY = 1e-4, args.lr = 3e-4

opt_proto = torch.optim.AdamW([model.prototypes], lr=args.lr, weight_decay=WEIGHT_DECAY)
```

**Lý do hai optimizer:** Trong SwAV gốc, prototype update có thể cần learning rate scheduling riêng, hoặc gradient clipping riêng. Việc tách cũng giúp có thể freeze prototype trong warmup nếu cần.

### 6.3. Learning Rate Scheduler

```python
def adjust_lr(optimizer, epoch, total_epochs, base_lr, warmup_epochs):
    """Linear warmup + Cosine annealing"""
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)  # Linear warmup
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))  # Cosine decay
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr
```

Scheduler này được áp dụng cho **cả hai optimizer** (`opt_network` và `opt_proto`).  
Không dùng `torch.optim.lr_scheduler`, mà implement thủ công.

### 6.4. Một training iteration

```python
for x_w, x_s in loader:
    x_w, x_s = x_w.to(DEVICE), x_s.to(DEVICE)
    
    opt_network.zero_grad()
    opt_proto.zero_grad()
    
    scores_w, scores_s = model(x_w, x_s)          # Forward → (B, K) mỗi view
    loss, q_w = criterion(scores_w, scores_s, return_q=True)  # SwAV loss
    
    loss.backward()     # Gradient chạy qua p_w, p_s → scores → z → encoder
    
    opt_network.step()  # Update encoder + projection head
    opt_proto.step()    # Update prototype matrix
    
    # Monitor cluster entropy để phát hiện collapse
    with torch.no_grad():
        avg_q = q_w.mean(dim=0)   # (K,) — trung bình soft assignment qua batch
        batch_entropy = -(avg_q * torch.log(avg_q + 1e-8)).sum().item()
```

**Gradient flow:**
```
x_w → encoder → GAP → projection_head → z_w → mm(proto_norm.T) → scores_w → softmax → p_w
                                                                                          ↓
                                             q_s (detached, Sinkhorn) → cross_entropy(q_s, log_p_w) → loss
```

Gradient chạy qua: `p_w` → `scores_w` → `z_w` → `projection_head` → encoder layers.  
Gradient **không chạy** qua: `q_s` (detached), prototype normalization (detached with `torch.no_grad()`).

> **Lưu ý:** Prototype gradient vẫn được nhận từ `scores_w = mm(z_w, proto_norm.T) / temp` — nhưng prototype normalization trong `proto_norm = F.normalize(self.prototypes, dim=-1)` bên trong `torch.no_grad()` block. Điều này có nghĩa là gradient **không chạy qua bước normalization** nhưng vẫn chạy đến `self.prototypes` trực tiếp qua chain rule ngoài `no_grad` block nếu implementation thực tế khác. **Cần kiểm chứng thêm** bằng cách xem chi tiết implementation `cluster_model.py`.

### 6.5. Cluster Collapse Detection

```python
# run_pretrain_prototype.py, dòng 216–219
max_entropy = math.log(args.num_prototypes)  # log(45) ≈ 3.807 — entropy tối đa (đều)

# Mỗi batch
avg_q = q_w.mean(dim=0)   # (K,) — average soft assignment
batch_entropy = -(avg_q * torch.log(avg_q + 1e-8)).sum().item()
```

**Điều kiện cảnh báo:**
```python
if avg_entropy < (0.3 * max_entropy):
    print(f"⚠️ Cảnh báo sụp cụm: Entropy={avg_entropy:.2f} < {0.3 * max_entropy:.2f}")
```

- `max_entropy = log(K) ≈ log(45) ≈ 3.81` — entropy khi tất cả prototypes được gán đều nhau.
- Nếu `avg_entropy < 0.3 * max_entropy ≈ 1.14`, nghĩa là phần lớn samples dồn vào ít prototype → collapse warning.
- **Không có early stopping** — chỉ in cảnh báo và tiếp tục.

### 6.6. Điều kiện lưu checkpoint

```python
# Chỉ lưu sau warmup (epoch > WARMUP_EPOCHS = 5)
if epoch > WARMUP_EPOCHS and avg_loss < best_loss:
    best_loss = avg_loss
    torch.save(model.encoder.state_dict(), ckpt_path)

# Fallback nếu không có epoch nào tốt hơn
if not ckpt_path.exists():
    torch.save(model.encoder.state_dict(), ckpt_path)
```

**Khác biệt với TS-TCC:** Có warmup period (5 epochs đầu) không lưu checkpoint. Lý do: trong các epoch đầu, model chưa ổn định, loss có thể thấp nhưng representation chất lượng kém.

### 6.7. Outputs sau pretraining

```
checkpoints/ssl_pretrain/prototype/<dataset_name>/<backbone_type>/
    ├── prototype_<backbone>_encoder_pretrained_<dataset>.pt  ← encoder.state_dict()
    ├── loss_history.csv                                       ← epoch, loss_total, entropy, lr
    ├── loss_curve.png                                         ← 2 subplots: loss + entropy
    ├── complexity.json                                        ← FLOPs, params của backbone
    └── metadata.json                                          ← toàn bộ hyperparams
```

> **Lưu ý tên file:** Checkpoint được đặt tên `prototype_<backbone>_encoder_pretrained_<dataset>.pt`. Script `run_few_shot.py` tìm checkpoint theo pattern `prototype_<backbone>_encoder_pretrained_<src>.pt`.

---

## Phần 7. Chi tiết riêng của Prototype SSL

### 7.1. So sánh với SwAV gốc

| Khía cạnh | SwAV gốc (ảnh) | Implementation trong repo |
|---|---|---|
| Input | Ảnh (C, H, W) | Sensor signal (6, 128) |
| Augmentation | Random crop, flip, color jitter | Jitter, scaling, permutation (TS_TCC_Augmentation) |
| Multi-crop | Có (nhiều resolution) | Không — chỉ 2 views |
| Prototype init | Random | Orthogonal |
| Prototype update | Gradient + momentum | 2 optimizer (AdamW network + AdamW proto) |
| Sinkhorn n_iters | 3 | 3 |
| Freeze prototypes trong warmup | Có (trong SwAV gốc) | **Không** (trong repo này) |
| Cơ chế anti-collapse | Sinkhorn equipartition | Sinkhorn + entropy monitoring |

### 7.2. Global Average Pooling

Trong `PrototypicalHARModel.forward()`, encoder output 3D `(B, D, L)` được GAP về `(B, D)`:

```python
if f_w.dim() == 3:
    f_w = f_w.mean(dim=-1)
```

Điều này quan trọng vì Prototype SSL yêu cầu vector embedding cố định chiều, không phải chuỗi. Khác với TS-TCC, nơi TC module cần sequence dimension.

### 7.3. Orthogonal Initialization của Prototype

`nn.init.orthogonal_` với shape `(45, 64)`:
- Tạo ma trận $45 \times 64$ với các hàng trực giao nhau.
- Điều này đảm bảo không prototype nào bị duplicate ban đầu.
- Đặc biệt hữu ích khi $K=45 < D=64$ — có thể tạo 45 vector trực giao trong không gian 64 chiều (hoàn toàn khả thi về mặt toán học).

> **Lý thuyết:** Nếu $K > D$, `orthogonal_` sẽ không tạo được $K$ vector trực giao độc lập trong không gian $D$-dimensional — sẽ báo lỗi hoặc cho kết quả không như mong đợi. Với $K=45, D=64$: $K < D$ nên OK.

### 7.4. Temperature Parameters

Có **hai temperature** khác nhau được dùng cho hai mục đích khác nhau:

| Parameter | Giá trị | Vị trí | Mục đích |
|---|---|---|---|
| `temperature = τ_s = 0.1` | 0.1 | `PrototypicalHARModel.__init__` | Scale scores trước softmax |
| `epsilon = 0.05` | 0.05 | `SwAVPrototypeLoss.__init__` | Sinkhorn temperature |

- **`τ_s`** nhỏ → softmax output sắc nét (peaky) → model tự tin hơn về assignment.
- **`epsilon`** nhỏ → target assignments `q` từ Sinkhorn sắc nét hơn → training signal mạnh hơn.

### 7.5. Augmentation được kế thừa từ TS-TCC

Prototype SSL dùng `ContrastiveDatasetWrapper` với `TS_TCC_Augmentation` — cùng augmentation pipeline. Điều này có thể ảnh hưởng đến chất lượng representation vì:
- `scaling` với `loc=2.0` có thể thay đổi biên độ mạnh.
- `permutation` phá vỡ temporal structure (có thể hữu ích cho HAR vì phá vỡ temporal ordering buộc model học pattern không phụ thuộc thứ tự).

---

## Phần 8. Learning Rate Scheduling

### 8.1. Linear Warmup + Cosine Annealing

```python
# run_pretrain_prototype.py, hàm adjust_lr
# 5 epochs warmup (tuyến tính từ 0 đến base_lr)
# Sau warmup: cosine decay từ base_lr về 0 (không có min_lr)
```

**Cụ thể với `base_lr=3e-4`, `warmup_epochs=5`, `total_epochs=40`:**
- Epoch 1-5: lr = 3e-4 * epoch/5 → [6e-5, 1.2e-4, 1.8e-4, 2.4e-4, 3e-4]
- Epoch 6-40: lr = 3e-4 * 0.5 * (1 + cos(π * (epoch-5)/35)) → giảm dần về ~0

### 8.2. Cùng LR cho network và prototype

Cả `opt_network` và `opt_proto` dùng cùng `adjust_lr()` với cùng `base_lr`:

```python
cur_lr = adjust_lr(opt_network, epoch, args.epochs, args.lr, WARMUP_EPOCHS)
adjust_lr(opt_proto, epoch, args.epochs, args.lr, WARMUP_EPOCHS)
```

> **Nhận xét:** Trong SwAV gốc, prototype thường có learning rate riêng (cao hơn hoặc thấp hơn). Việc dùng cùng LR cho cả hai là đơn giản hóa so với SwAV gốc.

---

## Phần 9. Downstream Task: Giống TS-TCC

Downstream pipeline **giống hoàn toàn** với TS-TCC. Chỉ khác checkpoint name và path:

**Checkpoint pattern khi load:**
```python
# run_few_shot.py, dòng 93-94
prefix_map = {"contrastive": "tstcc", "prototype": "prototype", "crosshar": "crosshar"}
source_ckpt = (OUTPUT_ROOT / "checkpoints/ssl_pretrain" / args.method / src / args.backbone /
               f"{prefix}_{args.backbone}_encoder_pretrained_{src}.pt")
```

Ví dụ: `checkpoints/ssl_pretrain/prototype/motionsense/standard/prototype_standard_encoder_pretrained_motionsense.pt`

**Loading vào HARClassifier:**
```python
model.encoder.load_state_dict(encoder_dict, strict=True)
```

Khi load encoder Prototype SSL (e.g., `StandardSensorEncoder1D`) vào `HARClassifier`:
- `HARClassifier.encoder` = `StandardSensorEncoder1D`
- `HARClassifier.classifier` = `ClassifierHead` (randomly initialized)
- Chỉ encoder được load từ SSL checkpoint, classifier head được train từ đầu.

---

## Phần 10. Hyperparameters và Cấu hình

| Hyperparameter | Giá trị mặc định | Vị trí khai báo | Ý nghĩa |
|---|---|---|---|
| `backbone` | `standard` | `args` | Encoder backbone |
| `num_prototypes` (K) | 45 | `args` | Số cluster centers |
| `feature_dim` | 128 | `FEATURE_DIM` (constant) | Chiều encoder output |
| `projection_dim` | 64 | `PROJECTION_DIM` (constant) | Chiều projection head |
| `temperature` (τ_s) | 0.1 | `TAU_S` (constant) | Scale cho score computation |
| `epsilon` | 0.05 | `EPSILON` (constant) | Sinkhorn temperature |
| `warmup_epochs` | 5 | `WARMUP_EPOCHS` (constant) | Epochs không lưu checkpoint |
| `sinkhorn_iters` | 3 | `SwAVPrototypeLoss.sinkhorn_knopp` | Số lần lặp normalize |
| `batch_size` | 64 | `args` | Số samples/batch |
| `learning_rate` | 3e-4 | `args` | Base LR (cả network và proto) |
| `weight_decay` | 1e-4 | `WEIGHT_DECAY` (constant) | L2 regularization |
| `epochs` | 40 | `args` | Số epochs pretrain |
| `optimizer` | AdamW | Hard-coded | — |
| Gradient clip | Không có | — | Khác với TS-TCC (có clip 2.0) |
| Scheduler | Linear warmup + Cosine | `adjust_lr()` | Manual (không dùng `torch.optim.lr_scheduler`) |
| Window size | 128 | Config | Timestep/window |
| Stride | 64 | Config | Overlap 50% |

> **Lưu ý quan trọng:** Prototype SSL **không có gradient clipping**, khác với TS-TCC (clip norm=2.0) và CrossHAR (clip norm=2.0). Điều này có thể gây gradient explosion nếu learning rate quá cao.

---

## Phần 11. Evaluation Metrics và Output

**Giống hoàn toàn với TS-TCC** — xem Phần 11 trong tài liệu TS-TCC.

**Output directory:**
```text
outputs_evaluation/cross_k_shot/prototype/<backbone>/<src>_to_<tgt>/
    ├── experiment_metadata.json
    └── <k>_shot/
        ├── linear_probing/
        └── full_finetuning/
```

---

## Phần 12. Checklist Kiểm tra Tính Đúng đắn

### 12.1. Loss Function

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| SwAV swap mechanism | ✅ Đã xác minh | `q_s → cross_entropy → p_w` và `q_w → cross_entropy → p_s` |
| Sinkhorn target detached | ✅ Đã xác minh | `q_w, q_s` được tính trong `torch.no_grad()` |
| Equipartition constraint | ✅ Đã xác minh | Row-normalize + col-normalize trong Sinkhorn |
| Anti-collapse mechanism | ✅ Đã xác minh | Sinkhorn cộng entropy monitoring |

### 12.2. Prototype

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| Orthogonal init | ✅ Đã xác minh | `nn.init.orthogonal_(self.prototypes)` |
| Prototype normalized trong forward | ✅ Đã xác minh | `F.normalize(self.prototypes, dim=-1)` mỗi forward |
| Gradient đến prototype | ❓ Cần kiểm chứng | `with torch.no_grad()` bọc normalization nhưng gradient vẫn đến `self.prototypes` qua `opt_proto` — cần kiểm tra xem gradient flow có đúng không |
| Hai optimizer riêng | ✅ Đã xác minh | `opt_network` và `opt_proto` |

### 12.3. Kiến trúc

| Vấn đề | Trạng thái | Chi tiết |
|---|---|---|
| GAP sau encoder | ✅ Đã xác minh | `f_w.mean(dim=-1)` trong forward |
| L2 normalize trước prototype matmul | ✅ Đã xác minh | `ProjectionHead` có `F.normalize` |
| Score = cosine similarity / temperature | ✅ Đã xác minh | Cả z và proto đều L2-normalized, chia temperature |

### 12.4. Vấn đề tiềm ẩn cần kiểm tra

**1. Gradient flow qua prototype normalization**

```python
# Trong PrototypicalHARModel.forward():
with torch.no_grad():
    proto_norm = F.normalize(self.prototypes, dim=-1, p=2)  # detached

scores_w = torch.mm(z_w, proto_norm.T) / self.temperature  # proto_norm đã detach
```

Nếu `proto_norm` được tạo với `no_grad`, thì `scores_w` sẽ không có gradient qua `proto_norm` (và do đó không qua `self.prototypes`). Điều này có nghĩa là `opt_proto.step()` có thể không update gì cả (gradient = 0).

> **Cần kiểm chứng:** Đây là vấn đề tiềm ẩn nghiêm trọng. Nếu gradient không chạy đến `self.prototypes`, prototype sẽ không được cập nhật dù có `opt_proto.step()`. Kết quả: prototype giữ nguyên orthogonal init suốt quá trình training.

**Cách kiểm tra:**
```python
model.prototypes.grad  # Nếu None sau backward, có vấn đề
```

**2. Không có gradient clipping**
- Prototype SSL không có `clip_grad_norm_` trong training loop.
- Với learning rate 3e-4 và projection head MLP, gradient explosion ít khả năng nhưng nên theo dõi.

**3. `num_prototypes=45` với không có domain knowledge**
- 45 được chọn vì 5 lớp × 9 = 45 (có thể). Hoặc số này là hyperparameter cần tune.
- README/docstring không giải thích lý do chọn 45.

**4. Augmentation không matching với prototype clustering**
- `scaling(loc=2.0)` trong weak augmentation có thể tạo ra view rất khác biên độ với raw data.
- Với clustering, điều này có thể gây prototypes học biên độ thay vì pattern.

**5. Không freeze prototypes trong warmup**
- Trong SwAV gốc, prototypes thường không được update trong một số epochs đầu để encoder học được representation trước.
- Repo này update prototype từ epoch 1, có thể dẫn đến suboptimal prototypes trong giai đoạn đầu.

---

## Phần 13. Cài đặt và Chạy Code

### 13.1. Yêu cầu môi trường

Giống TS-TCC (xem `requirements.txt`). **Không cần `einops`** vì Prototype SSL không dùng `attention.py`.

```
torch >= 2.0.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
```

### 13.2. Chạy SSL Pretraining

```bash
python scripts/02_pretrain_ssl/run_pretrain_prototype.py \
  --backbone standard \
  --datasets uci_har motionsense \
  --epochs 40 \
  --batch_size 64 \
  --lr 3e-4 \
  --num_prototypes 45
```

### 13.3. Chạy K-shot cross-domain evaluation

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method prototype \
  --backbone standard \
  --k_shots 10 20 50 100 \
  --seeds 42 100 2024 7 99 \
  --epochs 40 \
  --pairs motionsense:uci_har uci_har:motionsense
```

---

## Phần 14. Tóm tắt kiến trúc (Sơ đồ tổng quát)

```
[x_weak (B,6,128)] ─────────────────────────────────────────────┐
                                                                  ├──→ PrototypicalHARModel
[x_strong (B,6,128)] ───────────────────────────────────────────┘
                               ↓
                [Encoder Backbone (any: standard/cnn_transformer/tstcc/vit)]
                               ↓
                     [Global Average Pooling]
                     h_w (B,128)  +  h_s (B,128)
                               ↓
                        [ProjectionHead]
                  Linear(128→128)→BN→ReLU→Linear(128→64)→L2-norm
                     z_w (B,64)  +  z_s (B,64)
                               ↓
                  [Prototype Matrix: (K=45, 64), L2-normalized]
            scores_w = z_w @ proto.T / τ_s    → (B, 45)
            scores_s = z_s @ proto.T / τ_s    → (B, 45)
                               ↓
                     [SwAVPrototypeLoss]
            q_w = Sinkhorn(scores_w, ε=0.05)  → (B, 45) [detached]
            q_s = Sinkhorn(scores_s, ε=0.05)  → (B, 45) [detached]
            p_w = softmax(scores_w)            → (B, 45)
            p_s = softmax(scores_s)            → (B, 45)
            
            L = -½ * mean(q_s * log(p_w) + q_w * log(p_s))
                               ↓
            [opt_network] ← gradient(encoder, projection_head)
            [opt_proto]   ← gradient(prototypes) [nếu gradient flow đúng]
                               ↓
            [Lưu] encoder.state_dict() nếu avg_loss < best_loss (sau warmup)
```

---

## Phần 15. Tài liệu Tham khảo

**Bài báo gốc SwAV:**
- Caron, M., et al. "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments." NeurIPS 2020.

**Implementation trong repository:**
- `models/ssl/prototype/cluster_model.py` — Docstring mô tả "Mô hình học tự giám sát dựa trên Prototype (SwAV-inspired)"
- `losses/swav_loss.py` — Docstring mô tả "Loss Function theo phong cách SwAV sử dụng Sinkhorn-Knopp"
- `scripts/02_pretrain_ssl/run_pretrain_prototype.py` — Docstring "PRETRAIN PROTOTYPE SSL ENCODER (SwAV-INSPIRED)"
