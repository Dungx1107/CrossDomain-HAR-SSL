# Tài liệu kỹ thuật: TS-TCC × SwAV (`tstcc_swav`)

> **Phạm vi:** Tài liệu này mô tả implementation hiện tại trong repository
> `CrossDomain-HAR-SSL`, bao gồm pretraining self-supervised và transfer
> cross-domain. Các đường dẫn và shape được đối chiếu trực tiếp với mã nguồn.

## 1. Tổng quan và động cơ nghiên cứu

### 1.1. Mục tiêu phương pháp

`tstcc_swav` kết hợp hai tín hiệu bổ sung cho chuỗi cảm biến quán tính:

- **TS-TCC Temporal Contrasting (TC):** học động học nhân quả bằng cách dùng
  ngữ cảnh quá khứ tại thời điểm `t` để dự đoán `K` bước tương lai
  `t+1, ..., t+K` của view còn lại. Tín hiệu này khuyến khích biểu diễn giữ
  thông tin về diễn tiến theo thời gian.
- **SwAV Prototype Loss:** ánh xạ context embedding vào một tập prototype
  dùng chung và trao đổi assignment giữa weak/strong view. Sinkhorn-Knopp
  áp đặt assignment gần equipartition, giúp biểu diễn học các cấu trúc ngữ
  nghĩa toàn cục và giảm nguy cơ class/mode collision.

Khác với TS-TCC gốc, bản hybrid không dùng NT-Xent contextual contrastive loss
trong cấu hình mặc định. Nhánh `L_CC` được thay bằng `L_SwAV`, nhận trực tiếp
context projection `(B, 32)` từ module TC để tính similarity với prototype.
Code vẫn giữ `--lambda_cc` cho ablation; chỉ khi đặt giá trị lớn hơn `0` thì
NT-Xent mới được cộng thêm.

### 1.2. Pipeline tổng thể

```text
Dataset không nhãn của source domain
        │
        ├─ weak/strong augmentation
        ▼
TS-TCC × SwAV pretraining
        │  encoder + TC + prototypes
        ├─ chọn best sau warm-up hoặc epoch cuối
        ▼
Lưu `encoder.state_dict()` duy nhất
        │  prototype, TC và projection head không được dùng downstream
        ▼
Target-domain K-shot sampling
        │
        ├─ Linear probing (đóng băng encoder)
        └─ Fine-tuning (mở khóa encoder theo chế độ)
        ▼
Đánh giá macro-F1/accuracy trên target test set
```

## 2. Bản đồ kiến trúc mã nguồn

| Thành phần | File thực tế | Vai trò |
|---|---|---|
| Model wrapper | `../../models/ssl/hybrid/tstcc_swav/tstcc_swav_model.py` | `TSTCCSwAVModel`: encoder, TC, projection context và prototype matrix |
| Encoder | `../../models/encoders/tstcc_encoder.py` | 1D-CNN TS-TCC, giữ chiều thời gian, xuất `(B, 128, 18)` |
| Encoder factory | `../../models/encoders/builder.py` | Tạo `standard`, `cnn_transformer`, `vit_1d` hoặc `tstcc` |
| TC module | `../../models/ssl/tstcc/tc.py` | `Seq_Transformer`, các head `W_k`, InfoNCE |
| Transformer attention | `../../models/ssl/tstcc/attention.py` | CLS token và self-attention để tạo context `c_t` |
| SwAV loss | `../../losses/swav_loss.py` | `sinkhorn_knopp()` và `SwAVPrototypeLoss` |
| Config hybrid | `../../config/methods/tstcc_swav_config.py` | `K`, nhiệt độ, số vòng Sinkhorn, trọng số loss, warm-up |
| Training engine | `../../engines/pretrain_ssl/tstcc_swav/tstcc_swav_trainer.py` | DataLoader, validation shape, optimizer, logging và checkpoint |
| CLI pretraining | `../../scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py` | Chọn dataset/backbone và khởi chạy engine |
| Dataset views | `../../datasets/contrastive_dataset.py`, `../../models/ssl/tstcc/augmentations.py` | Sinh `(x_weak, x_strong)` từ một sample |
| Downstream CLI | `../../scripts/03_cross_domain/run_few_shot.py` | Load encoder checkpoint, K-shot và LP/FT |
| Downstream trainer | `../../engines/transfer/finetune_trainer.py` | Huấn luyện classifier trên target domain |

> **Tên file cần lưu ý:** yêu cầu ban đầu có thể tham chiếu
> `models/ssl/contrastive/tstcc_swav_model.py`; implementation hiện tại nằm
> ở `../../models/ssl/hybrid/tstcc_swav/tstcc_swav_model.py`. CLI và trainer đang import đường
> dẫn `hybrid`, vì vậy không đổi tên nếu chưa cập nhật toàn bộ import.

## 3. Luồng dữ liệu và biến đổi tensor

Với batch kích thước `B`, input chuẩn là `(B, 6, 128)`:

1. **Hai view:** `ContrastiveDatasetWrapper` đọc `samples`, bỏ qua nhãn và
   áp dụng `TS_TCC_Augmentation`, tạo `x_w` và `x_s`, mỗi view `(B, 6, 128)`.
2. **Encoder:** `TSTCCEncoder` thực hiện ba block Conv1d/BN/ReLU/MaxPool:
   `(B, 6, 128) → (B, 32, 65) → (B, 64, 34) → (B, 128, 18)`.
   Encoder **không** global-average-pool; chiều thời gian được giữ cho TC.
3. **Cutoff:** chọn chung `t_samples` trong
   `[MIN_T, T_out - TIMESTEPS - 1]`. TC cắt quá khứ `0..t` và lấy
   `K=TIMESTEPS` target tương lai `t+1..t+K`.
4. **TC/Transformer:** feature map được transpose thành `(B, 18, 128)`.
   `Seq_Transformer` thêm CLS token, attention trên chuỗi quá khứ và xuất
   `c_t` có shape `(B, 64)`.
5. **Context projection:** `TC.projection_head` biến `(B, 64)` thành
   `(B, 32)` qua `Linear(64,64)`, BatchNorm, ReLU và `Linear(64,32)`.
   Hai output là `c_w` và `c_s`; sau đó L2-normalize thành `ec_w`, `ec_s`.
6. **Nhánh TC:** mỗi `W_k(c_t)` dự đoán một vector `(B, 128)`. Tích vô
   hướng với target tương lai tạo ma trận logits `(B, B)` cho từng `k`,
   sau đó tính InfoNCE và cộng hai chiều weak→strong, strong→weak.
7. **Nhánh SwAV:** prototype matrix `C` có shape `(K, 32)`, mặc định
   `K=45`. `ec @ C.T` tạo cosine scores `(B, K)`. Sinkhorn tạo target
   assignment `q` `(B, K)`; prediction là `softmax(scores / tau)`, rồi
   cross-entropy swapped giữa hai view.

Đối với backbone khác, trainer gọi `_check_encoder()` và yêu cầu output 3D
`(B, 128, T_out)` đủ dài cho `K` bước tương lai. Vì vậy không nên giả định
`T_out=18` nếu thay backbone; log `encoder_seq_len` trong `config.json` là
giá trị thực tế.

## 4. Toán học và hàm mất mát

### 4.1. Temporal Contrasting

Với mỗi bước tương lai $k$, đặt $y_k^i$ là target của sample $i$ và $\hat{y}_k^j = W_k(c_t^j)$ là vector dự đoán tương lai từ context embedding của sample $j$.

Logits trong batch được tính bằng tích vô hướng:

$$
s_{ij}^{(k)}
=
y_k^i \cdot \hat{y}_k^j
$$

Đường chéo là cặp positive pair và các phần tử còn lại là negative pairs.

Loss cho một chiều:

$$
\mathcal{L}_{TC}^{(1)}
=
-\frac{1}{B K_{\text{TC}}}
\sum_{k=1}^{K_{\text{TC}}}
\sum_{i=1}^{B}
\log
\frac{
\exp\left(s_{ii}^{(k)}\right)
}{
\sum_{j=1}^{B}
\exp\left(s_{ij}^{(k)}\right)
}
$$

Model chạy hai chiều weak-to-strong và strong-to-weak:

L_TC = L_TC(weak → strong) + L_TC(strong → weak)

### 4.2. SwAV Swapped Prediction

Sau L2 normalization, similarity scores giữa context embedding và prototype được tính:

$$
a_w = e_w C^\top
$$

$$
a_s = e_s C^\top
$$

Trong đó:

$$
a_w, a_s \in \mathbb{R}^{B \times K_{\text{proto}}}
$$

Sinkhorn-Knopp nhận cosine scores chia cho $\epsilon$ và trả về assignment:

$$
q_w, q_s \in \mathbb{R}^{B \times K_{\text{proto}}}
$$

Prediction sử dụng temperature $\tau$:

$$
p_w =
\operatorname{softmax}
\left(
\frac{a_w}{\tau}
\right)
$$

$$
p_s =
\operatorname{softmax}
\left(
\frac{a_s}{\tau}
\right)
$$

### 4.2. SwAV Swapped Prediction

Loss swapped prediction trong code được tính như sau:

<div align="center">

<strong>
L<sub>SwAV</sub> = −
<span style="display:inline-block; vertical-align:middle; text-align:center">
  <span style="display:block; border-bottom:1px solid; padding:0 4px">1</span>
  <span style="display:block; padding:0 4px">2B</span>
</span>
∑<sub>i=1</sub><sup>B</sup>
[
q<sub>s</sub><sup>i</sup> · log p<sub>w</sub><sup>i</sup>
+
q<sub>w</sub><sup>i</sup> · log p<sub>s</sub><sup>i</sup>
]
</strong>

</div>

Trong đó:

- B: batch size.
- q<sub>s</sub><sup>i</sup>: Sinkhorn assignment của strong view tại sample i.
- q<sub>w</sub><sup>i</sup>: Sinkhorn assignment của weak view tại sample i.
- p<sub>w</sub><sup>i</sup>: prediction probability của weak view.
- p<sub>s</sub><sup>i</sup>: prediction probability của strong view.

### 4.3. Tổng loss

```math
\mathcal{L}_{Total}
= \lambda_{TC}\mathcal{L}_{TC}
+ \lambda_{SwAV}\mathcal{L}_{SwAV}
+ \lambda_{CC}\mathcal{L}_{CC}.
```

Cấu hình mặc định là `lambda_tc=1.0`, `lambda_swav=1.0`,
`lambda_cc=0.0`, do đó thực tế là
`\mathcal{L}_{Total} = \mathcal{L}_{TC} + \mathcal{L}_{SwAV}`.

### 4.4. Sinkhorn-Knopp và prototype update

`losses/swav_loss.py::sinkhorn_knopp()`:

- ổn định số học bằng cách trừ max theo hàng trước khi `exp`;
- chuẩn hóa xen kẽ hàng/cột trong `n_iters=3`;
- giữ tổng mỗi hàng xấp xỉ `1` và tổng mỗi cột xấp xỉ `B/K`;
- dùng `epsilon=0.05`, nhỏ hơn sẽ làm target assignment sắc hơn và nhạy
  hơn với chênh lệch cosine.

Prototype được khởi tạo `orthogonal_` và được L2-normalize trong forward.
Trainer dùng **một `AdamW` với hai parameter groups**: network chịu
`weight_decay=3e-4`, còn `ssl_model.prototypes` có `weight_decay=0`.
Đây là tương đương về mục tiêu với việc tách optimizer network/prototype,
nhưng code hiện tại không tạo hai optimizer độc lập. Sau `optimizer.step()`,
không có lệnh gán lại `self.prototypes.data`; việc normalize được thực hiện
khi tính scores ở mỗi forward.

Trainer theo dõi `proto_entropy` và `proto_used` để phát hiện prototype
collapse. `proto_entropy` có trần lý thuyết `log(K)`.

## 5. Thiết lập và checkpoint

### 5.1. Tham số mặc định

| Tham số | Giá trị |
|---|---:|
| `backbone` | `standard` |
| `epochs` / `batch_size` | `60` / `64` |
| `lr` / `weight_decay` | `3e-4` / `3e-4` |
| `TIMESTEPS` | `6` |
| `TC hidden/depth/heads/mlp` | `64 / 4 / 4 / 64` |
| `num_prototypes (K)` | `45` |
| `tau` / `epsilon` / `sinkhorn_iters` | `0.1 / 0.05 / 3` |
| `lambda_tc` / `lambda_swav` / `lambda_cc` | `1.0 / 1.0 / 0.0` |
| `warmup_epochs` | `5` |
| `grad_clip_max_norm` | `2.0` |

`drop_last` được bật khi dataset đủ lớn hơn batch size để giữ batch đầy,
phù hợp với Sinkhorn và BatchNorm.

### 5.2. Cấu trúc output

CLI tạo:

```text
checkpoints/ssl_pretrain/tstcc_swav/<source>/<backbone>/
├── tstcc_swav_<backbone>_encoder_pretrained_<source>.pt  # best sau warm-up
├── last_tstcc_swav_<backbone>_encoder_pretrained_<source>.pt
├── config.json
├── training_log.csv
└── pretrain_summary.json
```

Chỉ `ssl_model.encoder.state_dict()` được lưu trong hai file `.pt`; prototype,
TC và projection head không được chuyển sang downstream. Nếu dùng `--complexity`,
thư mục còn có `complexity.json`.

## 6. Hướng dẫn vận hành

### 6.1. Kiểm tra nhanh một dataset

```bash
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
  --backbone standard --epochs 1 --datasets uci_har
```

Kiểm tra log phải có `T_out`, khoảng `t`, `K`, loss hữu hạn và checkpoint
`best`/`last`.

### 6.2. Pretrain đầy đủ

```bash
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
  --backbone standard --epochs 60 --batch_size 64 \
  --datasets uci_har motion_sense hhar_phone hhar_watch
```

Tên dataset phải tồn tại trong `../../config/datasets/registry.py`; nếu registry
đang dùng alias khác, dùng đúng key tại đó. Có thể chọn `cnn_transformer`,
`vit_1d` hoặc `tstcc` qua `--backbone`.

### 6.3. Ablation bật lại NT-Xent

Đây không phải cấu hình hybrid mặc định:

```bash
uv run python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
  --variant cc --lambda_cc 0.7 \
  --datasets uci_har motion_sense
```

Checkpoint được tách vào method `tstcc_swav_cc`, tránh ghi đè kết quả chuẩn.

### 6.4. Đánh giá downstream K-shot

Sau khi pretrain source checkpoint, chạy ví dụ:

```bash
uv run python scripts/03_cross_domain/run_few_shot.py \
  --pairs "uci_har:hhar_phone" "uci_har:hhar_watch" \
  --method tstcc_swav --backbone standard \
  --k_shots 10 30 50 100 150 200 --epochs 40 \
  --seeds 42 100 2024 7 99 --batch_size 16 \
  --freeze_modes full partial:2 none
```

`full` là linear probing, `partial:N` mở `N` block, và `none` là full
fine-tuning. Script đọc checkpoint theo đúng quy ước ở trên, lưu metadata,
`summary_*.json`, `detailed_all_seeds.json` và confusion matrix dưới:

```text
outputs_evaluation/cross_k_shot/tstcc_swav/<backbone>/<source>_to_<target>/
```

### 6.5. Chẩn đoán lỗi thường gặp

- **Thiếu checkpoint:** kiểm tra `<source>`, `<backbone>` và method variant;
  đường dẫn được dựng trong `run_few_shot.py`.
- **Shape không hợp lệ:** `_check_encoder()` yêu cầu output 3D có 128
  channels và còn đủ `TIMESTEPS` sau cutoff.
- **NaN/Inf:** trainer dừng ngay; kiểm tra learning rate, batch size, dữ liệu
  hữu hạn và `proto_entropy`/`proto_used` trong `training_log.csv`.
- **Prototype collapse:** xem entropy so với `log(45)` và số prototype được
  dùng; thử tăng batch size hoặc điều chỉnh `epsilon`, `tau` qua config/CLI
  ablation thay vì sửa trực tiếp checkpoint.

## 7. Kiểm định triển khai

Trước một run dài, nên xác nhận:

1. `python losses/swav_loss.py` chạy được các assertion về shape, row sum,
   positivity và backward.
2. Smoke pretrain một epoch tạo đủ `config.json`, CSV, summary và hai encoder
   checkpoints.
3. Smoke downstream với một cặp source→target và một seed xác nhận
   `run_few_shot.py` load được encoder-only `state_dict`.
4. Đối chiếu `encoder_seq_len`, `t_samples_range`, `loss_swav`,
   `proto_entropy` và `proto_used` trong log trước khi chạy toàn bộ benchmark.
