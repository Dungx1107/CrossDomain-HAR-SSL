# Tài liệu kỹ thuật: CrossHAR trong repository

> **Phạm vi.** Tài liệu này mô tả implementation hiện có trong repository, không
> khẳng định rằng mọi chi tiết tương ứng hoàn toàn với bài báo CrossHAR gốc.
> Dẫn chiếu dùng đường dẫn tương đối từ thư mục gốc; số dòng là số dòng tại thời
> điểm khảo sát và có thể thay đổi khi mã nguồn được chỉnh sửa.

## 1. Tổng quan

### 1.1. Điều đã xác minh từ code

Trong repository, tên `crosshar` chỉ một pipeline **hierarchical self-supervised
pretraining** gồm hai mục tiêu:

1. **Masked Sensor Modeling (MSM):** che các đoạn thời gian trên tín hiệu đã
   chuẩn hóa theo kênh, sau đó tái tạo tín hiệu bị che bằng decoder.
2. **Contrastive Regularization (CR):** tạo hai view của cùng một mẫu, đưa qua
   encoder, `TransformerContextModule` và projection head, rồi tính
   InfoNCE-like cross-entropy.

Entry point là
`scripts/02_pretrain_ssl/run_crosshar_pretrain.py::main()`. Trainer thực tế là
`engines/pretrain_ssl/crosshar_sequential_trainer.py::train_crosshar_sequential`.
Mô hình pretraining là
`models/ssl/masked/crosshar_model.py::CrossHARPretrainModel`.

CrossHAR trong code **không phải một domain-adversarial hoặc explicit
domain-alignment trainer**. Cross-domain được thực hiện ở giai đoạn sau:
checkpoint encoder của source được nạp vào mô hình phân loại và fine-tune/linear
probe bằng dữ liệu có nhãn của target trong
`scripts/03_cross_domain/run_few_shot.py` hoặc
`scripts/03_cross_domain/run_cross_domain.py`. Vì vậy cần phân biệt:

| Khái niệm | Trạng thái trong implementation |
|---|---|
| CrossHAR-style SSL pretraining | Có, trong `run_crosshar_pretrain.py` |
| Pretraining đồng thời trên source và target | Không được entry point này thực hiện; mỗi `--datasets` được xử lý độc lập |
| Adversarial/domain discrepancy loss | Không thấy trong CrossHAR model/trainer |
| Transfer source → target | Có, bằng nạp encoder checkpoint rồi huấn luyện classifier trên target |
| `CrossHARClassifier` có context module ở downstream | Có class, nhưng runner cross-domain không sử dụng class này |

### 1.2. Lý thuyết, implementation và điểm cần kiểm chứng

- **Lý thuyết tham chiếu:** CrossHAR được mô tả trong repository như một phương
  pháp pretraining phân cấp dựa trên khôi phục tín hiệu cảm biến và regularization
  tương phản. Các khái niệm này được phản ánh trong docstring của
  `datasets/crosshar_dataset.py` và `models/ssl/masked/crosshar_model.py`.
- **Implementation đã xác minh:** Stage A chỉ tối ưu MSM; Stage B tối ưu
  `6 * loss_m + 1 * loss_r`. Encoder được lưu riêng dưới dạng `state_dict`.
- **Chưa xác minh:** chưa có bằng chứng trong code này cho thấy mọi chi tiết là
  “100% đặc tả bài báo”; docstring dùng cách diễn đạt đó nhưng cần đối chiếu
  với bài báo gốc. Đặc biệt, tên `x_neg`/`x_pos` là tên biến code, không đủ để
  kết luận đây là negative/positive theo nghĩa của một loss triplet.

### 1.3. Sơ đồ pipeline thực tế

```text
data/processed/<source>/dataset_all.pt
        |
        | load samples; lọc label 0..4; chuẩn hóa layout về (N, 6, 128)
        v
Physical3DAugmentation.expand_dataset_6x()  [mặc định bật]
        |
        v
CrossHARPretrainDataset.__getitem__()
        |-- x_raw = Channel-wise InstanceNorm(x)
        |-- x_neg = scale Gaussian rồi InstanceNorm
        `-- x_pos = permute 4 segments + jitter rồi InstanceNorm
        |
        +--> MSM: mask x_raw -> encoder -> TransformerSensorDecoder
        |             -> masked MSE loss_m
        |
        `--> CR: x_neg/x_pos -> shared encoder -> context -> projection
                      -> pairwise similarity -> cross-entropy loss_r
                                  |
                                  v
                    loss_total = alpha * loss_m + beta * loss_r
                                  |
                         AdamW + gradient clipping
                                  |
             checkpoints/ssl_pretrain/crosshar/<dataset>/<backbone>/
```

## 2. Cấu trúc mã nguồn liên quan

| File / symbol | Vai trò và quan hệ gọi |
|---|---|
| `scripts/02_pretrain_ssl/run_crosshar_pretrain.py::main` | Parse CLI, chọn dataset/checkpoint directory, load `dataset_all.pt`, tạo augmentation/dataloader và gọi trainer. |
| `config/uci_har_config.py::UCIHARConfig` | Đường dẫn UCI-HAR, mapping nhãn, 6 kênh, window/stride và subject split. |
| `config/motionsense_config.py::MotionSenseConfig` | Đường dẫn MotionSense, channel names, subject split, window/stride. |
| `config/hhar_config.py::HHARConfig` | Đường dẫn phone/watch, 5-class map, subject split, DSP và window parameters. |
| `datasets/crosshar_dataset.py::Physical3DAugmentation` | Hoán vị đồng bộ 3 trục accelerometer và 3 trục gyro; có thể mở rộng một tensor `N` thành `6N`. |
| `datasets/crosshar_dataset.py::channel_instance_norm` | Chuẩn hóa từng channel theo chiều thời gian. |
| `datasets/crosshar_dataset.py::ContrastiveViewGenerator` | Sinh `x_neg`, `x_pos` tại thời điểm `__getitem__`. |
| `datasets/crosshar_dataset.py::CrossHARPretrainDataset` | Trả `(x_raw, x_neg, x_pos)`. Không trả label. |
| `engines/pretrain_ssl/crosshar_sequential_trainer.py::train_crosshar_sequential` | Tự suy ra kích thước encoder, chạy hai stage, optimizer/scheduler, validation, checkpoint và CSV/plot loss. |
| `models/ssl/masked/crosshar_model.py::CrossHARPretrainModel` | Ghép encoder, mask generator, reconstruction decoder, context module và projection head; tính hai loss. |
| `models/ssl/masked/mask_generator.py::SegmentMaskGenerator` | Sinh mask segment liên tục, mặc định `mask_ratio=0.15`. |
| `models/encoders/builder.py::build_encoder` | Factory cho `standard`, `cnn_transformer`, `vit_1d`, `tstcc`. CrossHAR CLI mặc định hai backbone đầu. |
| `models/encoders/cnn1d.py::StandardSensorEncoder1D` | 4 block Conv1d/BN/ReLU/MaxPool; đầu ra `(B,128,8)` với input length 128. |
| `models/encoders/cnn_transformer.py::CNNTransformerEncoder` | CNN stem, positional encoding và Transformer; code hiện tại giữ length 128 dù comment mô tả nén còn 32. |
| `scripts/03_cross_domain/run_few_shot.py::run_few_shot` | Nạp checkpoint CrossHAR theo source/backbone, lấy `k` mẫu mỗi lớp ở target, gọi transfer trainer. |
| `scripts/03_cross_domain/run_cross_domain.py::run_fraction_experiment` | Biến thể fraction-based, dùng tỷ lệ mẫu target thay vì k mẫu mỗi lớp. |
| `engines/transfer/finetune_trainer.py::train_and_eval_finetune` | Nạp encoder state dict, thiết lập LP/FT, CrossEntropy, validation Macro F1 và test. |
| `models/har_classifier.py::HARClassifier` | Ghép encoder với classifier downstream; được transfer trainer dùng. |
| `models/heads/classifier.py::ClassifierHead` | Mean-pool nếu encoder trả tensor 3D rồi linear classifier. |
| `engines/evaluation/evaluator.py::ModelEvaluator` | Inference không gradient, accuracy, macro/weighted F1, classification report và confusion matrix. |

`CrossHARClassifier` trong `crosshar_model.py` là một class hợp lệ gồm encoder,
`TransformerContextModule` và `FeatureMixerClassificationHead`, nhưng không được
import/gọi trong hai script cross-domain nêu trên. Đây là khác biệt quan trọng
giữa “class có trong repository” và “đường chạy thực tế”.

## 3. Dữ liệu đầu vào và cấu trúc tensor

### 3.1. File `.pt` và domain

`run_crosshar_pretrain.py` ánh xạ dataset như sau (dòng 37–42):

```text
motionsense  -> data/processed/motionsense/dataset_all.pt
uci_har      -> data/processed/uci_har/dataset_all.pt
hhar_phone   -> data/processed/hhar_phone/dataset_all.pt
hhar_watch   -> data/processed/hhar_watch/dataset_all.pt
```

Mỗi file được kỳ vọng là dictionary có ít nhất:

```python
{
    "samples": Tensor,
    "labels": Tensor,
    # "subjects": Tensor, nếu được tạo bởi preprocessing
}
```

`load_dataset_all_tensor()` lấy `data["samples"]`, squeeze `data["labels"]`,
giữ nhãn `0 <= y < 5`, ép `float`, và nếu layout là `(N,128,6)` thì chuyển thành
`(N,6,128)` (`run_crosshar_pretrain.py::load_dataset_all_tensor`, dòng 44–63).
Label sau bước lọc không được đưa vào `CrossHARPretrainDataset`; nó chỉ được dùng
để loại `Laying` của UCI-HAR và `Jogging` của MotionSense.

Tất cả dataset được code này giả định có 6 channel:

```text
sample: (C, T) = (6, 128)
batch : (B, C, T) = (B, 6, 128)
C=0..2: accelerometer / total_acc
C=3..5: gyroscope hoặc rotation-rate
```

Tên chính xác của channel đến từ config/preprocessing:

- UCI-HAR: `total_acc_x/y/z`, `body_gyro_x/y/z`
  (`config/uci_har_config.py`, dòng 60–65).
- MotionSense: `total_acc_x/y/z`, `rotationRate.x/y/z`
  (`config/motionsense_config.py`, dòng 58–62).
- HHAR: code gộp 3 trục accelerometer và 3 trục gyro thành 6 channel
  (`scripts/00_preprocess/preprocess_hhar.py`, dòng 159–167).

### 3.2. Dữ liệu được tạo ra từ raw data

- **UCI-HAR:** đọc sáu file trong `Inertial Signals/`, stack thành
  `(N,6,128)`, map nhãn 1–6 về 0–5, tách validation theo subject và lưu
  `train.pt`, `val.pt`, `test.pt`, `dataset_all.pt`
  (`scripts/00_preprocess/preprocess_uci_har.py::load_signals`,
  `process_raw_subset`, `main`, dòng 33–121).
- **MotionSense:** đọc CSV theo activity/subject, tính
  `total_acc = userAcceleration + gravity`, ghép với rotation rate, trượt
  window 128 stride 64, rồi lưu bốn file `.pt`
  (`scripts/00_preprocess/preprocess_motionsense.py::process_subject_subset`,
  dòng 44–102; `main`, dòng 105–128).
- **HHAR:** đọc accelerometer/gyroscope CSV, lọc 5 nhãn chung, group theo
  `(User, Device, gt)`, cắt gap trên 1 giây, low-pass khi cần, nội suy về 50 Hz,
  giữ recording dài ít nhất 256 mẫu, rồi window 128 stride 64. Phone và watch
  được lưu tách riêng (`scripts/00_preprocess/preprocess_hhar.py`, dòng 44–220).

### 3.3. Split trong pretraining và cross-domain

Pretraining CrossHAR không đọc `train.pt`/`val.pt`/`test.pt`. Script đọc
`dataset_all.pt` rồi gán:

```python
X_train = X_all
X_val = X_all
```

(`run_crosshar_pretrain.py`, dòng 82–88). Vì vậy `val_loader` không phải
validation độc lập; nó chỉ là cùng tập mẫu chưa expand để theo dõi loss.

Ở downstream cross-domain, `run_few_shot.py` và `run_cross_domain.py` đọc
`train.pt`, `val.pt`, `test.pt` của **target**. Source được xác định qua đường
dẫn checkpoint, không được dùng làm batch supervised trong transfer:

```text
source dataset -> SSL checkpoint
target train    -> k-shot hoặc fraction, có nhãn
target val      -> chọn epoch tốt nhất theo Macro F1
target test     -> đánh giá cuối
```

Hai script lọc target về nhãn 0–4 và đổi layout `(N,128,6)` thành `(N,6,128)`.
Kịch bản này là transfer learning sau SSL; không có target unlabeled loss trong
`train_and_eval_finetune`.

## 4. Tiền xử lý và tạo cửa sổ

CrossHAR pretraining không thực hiện lại raw preprocessing. Nó tiêu thụ các
tensor đã đóng gói. Chuỗi biến đổi chính là:

1. Raw preprocessing tạo window 128 và lưu tensor `(N,6,128)`.
2. `load_dataset_all_tensor` lọc lớp và chuẩn hóa layout.
3. `Physical3DAugmentation.expand_dataset_6x` (mặc định `--expand_6x`) tạo
   `(6N,6,128)`.
4. `CrossHARPretrainDataset` giữ tensor trong RAM và sinh views khi lấy item.
5. `DataLoader` dùng batch size mặc định 64, shuffle train; validation không
   shuffle (`run_crosshar_pretrain.py`, dòng 95–102).

Không có xử lý missing value trong CrossHAR dataset wrapper. Missing value,
lọc bản ghi, resampling và low-pass (HHAR) nằm ở preprocessing tương ứng.
`subjects` được lưu trong `.pt` nhưng không được dùng trong pretraining CrossHAR
để tạo split hoặc loss.

## 5. Data augmentation

### 5.1. Physical 3D permutation

`Physical3DAugmentation` khai báo sáu permutation của `[0,1,2]`
(`datasets/crosshar_dataset.py`, dòng 30–44). Với một sample `(6,T)`, cùng
permutation được áp dụng cho:

```text
accelerometer: x_aug[0:3] = x[perm]
gyro:          x_aug[3:6] = x[perm + 3]
```

`expand_dataset_6x` áp dụng cả sáu permutation cho toàn bộ `X` và nối theo
batch dimension, nên số sample tăng từ `N` lên `6N`; shape không đổi ở mỗi sample.
Đây là augmentation pretraining, không được gọi trong downstream script.

### 5.2. Channel-wise Instance Normalization

`channel_instance_norm` tính mean/std riêng cho mỗi channel trên chiều thời gian:

$$
\operatorname{IN}(x_{c,t}) =
\frac{x_{c,t}-\mu_c}{\sigma_c+\epsilon},\qquad \epsilon=10^{-6}.
$$

Nó hỗ trợ `(C,T)` và `(B,C,T)`. `CrossHARPretrainDataset.__getitem__` chuẩn hóa
`x_raw` trước khi tạo view; `ContrastiveViewGenerator` chuẩn hóa lại cả hai view
sau augmentation (`datasets/crosshar_dataset.py`, dòng 80–96 và 137–166).

### 5.3. Hai contrastive views

Với `x` shape `(6,128)` và `num_segments=4`:

- `x_neg`: nhân toàn bộ tín hiệu với một scalar
  `scale_factor ~ N(2.0, 1.1)` rồi InstanceNorm.
- `x_pos`: chia chuỗi thành bốn đoạn dài 32, xáo trộn thứ tự bằng
  `torch.randperm(4)`, cộng jitter `N(0, 0.05)` từng phần tử, rồi InstanceNorm.

Hai view đều giữ shape `(6,128)`. Tên `x_neg`/`x_pos` phản ánh tên biến, nhưng
loss thực tế coi cặp này là positive pair; mọi phần tử ngoài đường chéo self
được đưa vào nhóm negative trong similarity matrix. Không có sampling negative
từ dataset riêng.

## 6. Kiến trúc mô hình

### 6.1. Encoder

`build_encoder()` mặc định `feature_dim/d_model=128`. Hai lựa chọn được
`run_crosshar_pretrain.py` cho phép là `standard` và `cnn_transformer`.

**Standard.** `StandardSensorEncoder1D` (`models/encoders/cnn1d.py`) nhận
`(B,6,128)` và đi qua:

```text
(B, 6, 128)
  -> Conv1d 32 + BN + ReLU + MaxPool -> (B, 32, 64)
  -> Conv1d 64 + BN + ReLU + MaxPool -> (B, 64, 32)
  -> Conv1d 96 + BN + ReLU + MaxPool -> (B, 96, 16)
  -> Conv1d 128 + BN + ReLU + MaxPool -> (B, 128, 8)
```

**CNN-Transformer.** `CNNTransformerEncoder` (`models/encoders/cnn_transformer.py`)
gồm Conv1d `6→64`, Conv1d `64→128`, positional encoding, dropout và
TransformerEncoder 2 layers, 4 heads, feed-forward 256. Các convolution có
stride 1 và các `MaxPool1d` trong comment bị comment-out; do đó implementation
thực tế trả `(B,128,128)` cho input `(B,6,128)`, bất kể comment đầu file ghi
`(B,128,32)`. Trainer không hard-code chiều dài mà đo bằng dummy input.

### 6.2. MSM branch

`CrossHARPretrainModel.forward_msm`:

```text
x_raw (B,6,128)
   -> SegmentMaskGenerator
   -> x_masked (vị trí mask đặt 0)
   -> encoder z (B,128,Lf)
   -> TransformerSensorDecoder
   -> x_recon (B,6,Lf), interpolate về 128 nếu cần
```

`TransformerSensorDecoder` dùng positional encoding `feature_dim`, Transformer
Encoder 2 layers (đóng vai trò decoder theo implementation), rồi
`Linear(128, out_channels=6)`. Output được permute về `(B,6,Lf)`
(`models/ssl/masked/crosshar_model.py`, dòng 68–99).

### 6.3. Context, projection và classifier

`TransformerContextModule` đổi `(B,128,Lf)` thành `(B,Lf,128)`, cộng positional
encoding, chạy một TransformerEncoder layer (4 heads), rồi đổi lại
`(B,128,Lf)` (`crosshar_model.py`, dòng 41–65).

`ContrastiveProjectionHead` adaptive-average-pool theo thời gian, flatten,
`Linear(128,128)`, BatchNorm, ReLU, `Linear(128,64)`, sau đó L2-normalize.
Projection output là `(B,64)` (`crosshar_model.py`, dòng 101–115).

`CrossHARClassifier` (dòng 225–246) là kiến trúc downstream riêng:

```text
x -> encoder (B,128,Lf)
  -> TransformerContextModule (B,128,Lf)
  -> FeatureMixerClassificationHead
  -> logits (B,5)
```

Tuy nhiên transfer runner thực tế tạo `HARClassifier` với `ClassifierHead`, không
phải `CrossHARClassifier`; xem mục 9.

## 7. Self-supervised pretraining

### 7.1. Một batch

`CrossHARPretrainDataset` trả:

```text
x_raw, x_neg, x_pos: mỗi tensor (6,128)
DataLoader batch:     mỗi tensor (B,6,128)
```

Trong trainer (`crosshar_sequential_trainer.py`, dòng 86–121):

1. Đưa ba tensor lên `device`.
2. `optimizer.zero_grad()`.
3. Gọi `model(x_raw, x_neg, x_pos, alpha, beta)`.
4. Lấy `out["loss_total"]`.
5. `loss.backward()`.
6. `clip_grad_norm_(model.parameters(), max_norm=2.0)`.
7. `optimizer.step()`.

Khi `beta > 0`, gradient đi qua cả encoder, decoder, mask branch (qua dữ liệu
masked), context module và projection head. Khi Stage A có `beta=0`, nhánh CR
không được tính và không nhận gradient.

### 7.2. Mask generation và MSM loss

`SegmentMaskGenerator(mask_ratio=0.15, max_segment_len=16,
min_segment_len=4)` đặt `int(L * 0.15)` vị trí mask trên mỗi sample. Nó cố
chọn các đoạn không giao nhau; sau tối đa 200 lần thử, phần thiếu được lấp bằng
vị trí ngẫu nhiên (`models/ssl/masked/mask_generator.py`, dòng 30–71).
Mask được broadcast cho tất cả 6 channel.

Với mask nhị phân $m_{b,c,t}$ và `x_recon`, loss code là:

$$
L_m =
\frac{\sum_{b,c,t}m_{b,c,t}
    (\hat{x}_{b,c,t}-x_{b,c,t})^2}
     {\sum_{b,c,t}m_{b,c,t}+10^{-8}}.
$$

Chỉ vị trí bị mask đóng góp vào MSE; decoder vẫn tạo reconstruction ở toàn bộ
window. Loss nằm trong `CrossHARPretrainModel.forward_msm`, dòng 160–179.

### 7.3. Contrastive regularization

`forward_contrastive` dùng cùng `self.encoder` cho `x_neg` và `x_pos`, sau đó
đưa mỗi output qua cùng `reg_head` và `proj_head`. Hai projection được nối:

```text
representations = [s_neg; s_pos]  # (2B,64)
sim_matrix = representations @ representations.T / temperature  # (2B,2B)
```

Với batch size `B`, positive của hàng `i` là phần tử lệch `+B`, còn positive
của hàng `B+i` là phần tử lệch `-B`. Self similarity bị loại bằng identity
mask; mọi phần tử còn lại được flatten thành negative logits. Positive được đặt
ở cột 0, nhãn của mọi hàng là 0, nên:

$$
L_r = \operatorname{CE}
\left(
\begin{bmatrix}s^+ & s^-_1 & \cdots & s^-_{2B-2}\end{bmatrix},
\; y=0
\right),
\quad s = \frac{q_i^\top q_j}{\tau},
\quad \tau=0.2.
$$

Đây là cách tính trực tiếp tại `crosshar_model.py::forward_contrastive`,
dòng 181–204. Không có `detach`, momentum encoder, queue hoặc prototype.

### 7.4. Sequential stages, optimizer và checkpoint

`train_crosshar_sequential` mặc định:

| Giai đoạn | Epoch | `alpha` | `beta` | Objective |
|---|---:|---:|---:|---|
| Stage A (MSM Only) | `1..warmup_msm_epochs` | 6.0 | 0.0 | `6 * L_m` |
| Stage B (Joint) | còn lại | 6.0 | 1.0 | `6 * L_m + L_r` |

Stage B reset `best_val_loss` về vô cùng. Trong Stage A, checkpoint được chọn
theo `avg_val_m`; trong Stage B theo `avg_val_total`
(`crosshar_sequential_trainer.py`, dòng 86–99 và 162–166).

Optimizer là `AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)` và
`CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`. Không có early stopping; vòng
lặp luôn chạy đủ epoch. Sau mỗi epoch scheduler step, gradient được clip L2
2.0. Checkpoint được lưu bằng:

```python
torch.save(model.encoder.state_dict(), checkpoint_path)
```

Do chỉ lưu encoder, decoder/context/projection không được giữ để downstream.

## 8. Chi tiết riêng của CrossHAR

Implementation có cấu trúc phân cấp theo thứ tự:

```text
raw sensor reconstruction (MSM)
        -> representation encoder
        -> temporal context transformer
        -> normalized projection
        -> in-batch contrastive regularization
```

Những điểm **có trong code**:

- physical axis permutation 6x;
- channel-wise instance normalization;
- contiguous segment masking;
- reconstruction decoder có temporal self-attention;
- context Transformer và projection head;
- loss schedule MSM warm-up rồi joint MSM+CR.

Những điểm **không thấy trong code CrossHAR này**:

- không có discriminator hoặc domain classifier;
- không có source/target batch đồng thời trong pretraining;
- không có Sinkhorn-Knopp/prototype assignment;
- không có EMA teacher;
- không có negative queue hoặc cross-batch memory;
- không có supervised label loss trong pretraining.

Do đó “cross-domain” của thí nghiệm không phải domain adaptation loss trong
`CrossHARPretrainModel`; nó là transfer evaluation qua checkpoint source.

## 9. Downstream: Linear Probing và Fine-tuning

### 9.1. Checkpoint và mô hình downstream

`run_few_shot.py` tạo path:

```text
checkpoints/ssl_pretrain/crosshar/<source>/<backbone>/
crosshar_<backbone>_encoder_pretrained_<source>.pt
```

`train_and_eval_finetune` tải checkpoint bằng `torch.load(..., weights_only=True)`.
Nếu object là dictionary có `encoder_state_dict` hoặc `encoder`, nó lấy key đó;
nếu không, coi toàn object là encoder state dict. Sau đó gọi
`model.encoder.load_state_dict(..., strict=True)` (`finetune_trainer.py`,
dòng 77–113).

Runner tạo:

```text
build_encoder(target_backbone, in_channels=6)
        -> HARClassifier(encoder, ClassifierHead(num_classes=5))
```

`ClassifierHead` nhận `(B,128,Lf)`, mean-pool chiều cuối thành `(B,128)` rồi
linear thành `(B,5)`. Vì vậy checkpoint phải được pretrain với cùng backbone và
channel count với transfer run.

### 9.2. Linear Probing (LP)

Với `freeze_backbone=True`, trainer đặt `requires_grad=False` cho toàn bộ
`model.encoder`, tạo `Adam(model.classifier.parameters(), lr=head_lr=1e-3)`.
Loss supervised là `nn.CrossEntropyLoss()`. Context/projection/decoder của
CrossHAR không tồn tại trong mô hình downstream này.

### 9.3. Full Fine-tuning (FT)

Với `freeze_backbone=False`, toàn bộ encoder mở khóa. Optimizer Adam có hai
parameter group:

```text
encoder:     backbone_lr=1e-4
classifier:  head_lr=1e-3
```

Validation được chạy sau mỗi epoch; `ReduceLROnPlateau(mode="max",
factor=0.5, patience=5)` theo validation Macro F1. State tốt nhất theo F1 được
giữ trong RAM, nạp lại trước test. Checkpoint downstream chỉ được lưu nếu
`save_model_path` được truyền; hai runner cross-domain hiện không truyền tham
số này nên việc lưu state downstream không xảy ra ở đường chạy mặc định.

### 9.4. Few-shot và fraction

- `run_few_shot.py::sample_k_shot` lấy đúng `k` mẫu cho mỗi 5 lớp bằng
  `numpy.random.default_rng(seed)`; tổng train là `5k`. Val/test giữ toàn bộ.
- `run_cross_domain.py` dùng `sample_subset_by_ratio` để lấy fraction theo seed;
  tổng số mẫu phụ thuộc phân bố và utility sampling.
- Cả hai chạy các protocol `linear_probing` và `full_finetuning`, sau đó test
  trên toàn bộ target test.

## 10. Cross-domain training và evaluation

### 10.1. Domain và label space

Các tên domain hợp lệ trong runner là `motionsense`, `uci_har`,
`hhar_phone`, `hhar_watch`. Tất cả được đưa về 6 channel/128 timestep và năm
lớp chung:

```text
0 Walking
1 Upstairs
2 Downstairs
3 Sitting
4 Standing
```

`uci_har` có lớp `Laying` và MotionSense có `Jogging`, nhưng hai runner lọc
`labels < 5`, nên các lớp đó bị loại khỏi transfer benchmark.

### 10.2. Hai loại thí nghiệm

`run_few_shot.py` tạo:

```text
outputs_evaluation/cross_k_shot/
  crosshar/<backbone>/<source>_to_<target>/<k>_shot/<protocol>/
```

Mỗi protocol có `detailed_all_seeds.json`, `summary_*.json` và
`aggregated_confusion_matrix.png`. `run_cross_domain.py` tương tự nhưng dùng:

```text
outputs_evaluation/cross_fraction/
  crosshar/<backbone>/<source>_to_<target>/<fraction>/<protocol>/
```

Đây là **cross-domain transfer learning**: encoder học không nhãn ở source,
sau đó classifier được huấn luyện bằng target labels. Code không dùng nhãn
target cho SSL pretraining CrossHAR, nhưng `dataset_all.pt` source có thể gồm
các sample thuộc split test vì script pretraining cố ý đọc toàn bộ file.

`hhar_watch` là target có khác biệt vị trí cảm biến cổ tay so với pocket/phone.
Repository có đủ path để đánh giá riêng domain này, nhưng không có module đặc
biệt xử lý domain shift ngoài việc nạp checkpoint và fine-tune như các target
khác.

## 11. Metrics và output

### 11.1. Trong pretraining

Trainer ghi mỗi epoch:

```text
train_loss_total, train_loss_m, train_loss_r
val_loss_total, val_loss_m, val_loss_r, lr
```

vào `pretrain_loss_history.csv`, và tạo `pretrain_loss_curve.png` trong thư mục
checkpoint. Không tính accuracy/F1 trong SSL pretraining.

### 11.2. Trong downstream

`train_and_eval_finetune` dùng accuracy và Macro F1 trên validation/test để
chọn model và báo cáo. Các runner sau đó tính thêm:

- accuracy;
- macro precision, macro recall;
- macro F1;
- Cohen's kappa;
- F1 từng lớp;
- confusion matrix.

`ModelEvaluator.evaluate` còn tính weighted F1 và in
`classification_report`; nó dùng `argmax(logits, dim=1)` trong
`run_inference` (`engines/evaluation/evaluator.py`, dòng 57–86), và có thể lưu
ảnh confusion matrix. Summary runner lưu mean/std qua các seed; `detailed_all_seeds`
lưu metric và confusion matrix từng seed trong k-shot runner.

## 12. Checkpoint và khả năng tái sử dụng

### 12.1. Pretraining checkpoint

File checkpoint CrossHAR là **plain encoder state dict**, không có:

- optimizer state;
- scheduler state;
- epoch;
- loss history;
- decoder/context/projection state;
- cấu hình được serialize.

Tên mặc định:

```text
checkpoints/ssl_pretrain/crosshar/<dataset>/<backbone>/
crosshar_<backbone>_encoder_pretrained_<dataset>.pt
```

`pretrain_loss_history.csv` và `pretrain_loss_curve.png` là artifact riêng.

### 12.2. Tương thích

Việc load downstream dùng `strict=True`, nên cần khớp kiến trúc encoder và
`in_channels`. `feature_length` được tự suy ra trong pretraining nhưng
downstream chỉ tạo encoder; classifier mean-pool được cả output length 8 của
standard và length 128 của CNN-Transformer. Decoder và context module không
được chuyển sang downstream.

## 13. Cài đặt và chạy code

### 13.1. Môi trường

`requirements.txt` khai báo Python package nhưng không khóa Python/PyTorch/CUDA:

```text
torch>=2.0.0, torchvision, numpy>=1.21.0, pandas>=1.3.0,
scikit-learn>=1.0.0, matplotlib>=3.4.0, seaborn>=0.11.0,
scipy>=1.7.0, tqdm
```

Có thể tạo môi trường theo cách thông thường:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Lệnh trên là hướng dẫn cài đặt suy ra từ `requirements.txt`; chưa được chạy
trong quá trình viết tài liệu này. Device được chọn tự động là `cuda` nếu
`torch.cuda.is_available()`, ngược lại `cpu`.

### 13.2. Chuẩn bị dữ liệu

Raw preprocessing đã xác minh qua các entry point:

```bash
python scripts/00_preprocess/preprocess_uci_har.py
python scripts/00_preprocess/preprocess_motionsense.py
python scripts/00_preprocess/preprocess_hhar.py
```

Các lệnh trên cần raw data đúng cấu trúc mà config/script đọc; chúng được suy
ra từ `if __name__ == "__main__"` và chưa được thực thi trong khảo sát này.
Output local mặc định nằm dưới `data/processed/`. Khi biến môi trường Kaggle
được phát hiện, một số config chuyển sang `/kaggle/input/...`.

### 13.3. Pretrain CrossHAR

Lệnh trực tiếp từ entry point:

```bash
python scripts/02_pretrain_ssl/run_crosshar_pretrain.py \
  --datasets motionsense \
  --backbone standard \
  --epochs 60 \
  --warmup_msm_epochs 15 \
  --batch_size 64 \
  --lr 5e-4 \
  --seed 42
```

Các argument trên đều tồn tại trong parser (dòng 24–32). `--expand_6x` là
`store_true` với `default=True`, nên hiện không có cờ CLI để tắt theo cách
thông thường; code sẽ expand 6x. Đầu ra:

```text
checkpoints/ssl_pretrain/crosshar/motionsense/standard/
  crosshar_standard_encoder_pretrained_motionsense.pt
  pretrain_loss_history.csv
  pretrain_loss_curve.png
```

Đổi `motionsense` thành `uci_har`, `hhar_phone` hoặc `hhar_watch` nếu file
`dataset_all.pt` tương ứng tồn tại. Lệnh chưa được chạy kiểm chứng trong phiên
khảo sát này.

### 13.4. Few-shot transfer

Entry point và argument đã xác minh:

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method crosshar \
  --backbone standard \
  --epochs 40 \
  --batch_size 16 \
  --seeds 42 100 2024 7 99 \
  --k_shots 1 5 10 20 \
  --pairs motionsense:uci_har
```

Script yêu cầu checkpoint source tương ứng và ghi dưới
`outputs_evaluation/cross_k_shot/crosshar/standard/motionsense_to_uci_har/`.

### 13.5. Fraction transfer

```bash
python scripts/03_cross_domain/run_cross_domain.py \
  --method crosshar \
  --backbone standard \
  --epochs 40 \
  --batch_size 16 \
  --seeds 42 100 2024 7 99 \
  --fractions 0.01 0.05 0.10 \
  --pairs motionsense:uci_har
```

Output nằm dưới `outputs_evaluation/cross_fraction/`. Hai lệnh transfer ở trên
được đối chiếu trực tiếp với parser và path trong code, nhưng chưa được thực
thi trong khảo sát này.

## 14. Hyperparameters

| Hyperparameter | Giá trị mặc định | Vị trí | Ý nghĩa |
|---|---:|---|---|
| Input channels | 6 | config và CLI map | 3 accelerometer + 3 gyro |
| Window length | 128 | `config/*_config.py` | Số timestep/window |
| 3D permutations | 6 | `Physical3DAugmentation` | Mở rộng theo permutation trục |
| Contrastive segments | 4 | `ContrastiveViewGenerator` | Mỗi segment dài 32 với window 128 |
| Mask ratio | 0.15 | `train_crosshar_sequential` | Tỷ lệ timestep bị che |
| Mask segment length | 4–16 | `SegmentMaskGenerator` | Khoảng độ dài mỗi segment |
| Feature dim | 128 | `build_encoder` | Chiều channel latent |
| Projection dim | 64 | `ContrastiveProjectionHead` | Chiều embedding tương phản |
| Temperature | 0.2 | `train_crosshar_sequential`/model | Chia similarity |
| Total epochs | 60 | `run_crosshar_pretrain.py` | Số epoch SSL |
| MSM warm-up | 15 | `run_crosshar_pretrain.py` | Stage A |
| Batch size | 64 | `run_crosshar_pretrain.py` | Batch pretraining |
| Learning rate | 5e-4 | trainer/CLI | AdamW learning rate |
| Weight decay | 1e-4 | trainer | AdamW regularization |
| Scheduler | CosineAnnealingLR | trainer | `eta_min=1e-6`, `T_max=epochs` |
| Loss weights | `alpha=6`, `beta=0/1` | trainer | MSM/CR combination |
| Gradient clip | 2.0 | trainer | `clip_grad_norm_` |
| Transfer epochs | 40 | cross-domain scripts | Supervised target training |
| LP/FT LR | head `1e-3`, encoder `1e-4` | `finetune_trainer.py` | FT uses differential LR |

Các giá trị CLI có thể override giá trị mặc định parser; `mask_ratio`,
`temperature`, `weight_decay` hiện không có argument CLI trong runner CrossHAR
và giữ giá trị mặc định của trainer.

## 15. Checklist kiểm tra tính đúng đắn

### Đã xác minh

- [x] Input CrossHAR được chuẩn hóa về `(B,6,128)` trước model.
- [x] Pretraining không lấy `labels` vào Dataset/loss; labels chỉ dùng để lọc
  năm lớp trong loader.
- [x] `x_neg` và `x_pos` dùng chung encoder/context/projection parameters.
- [x] MSM loss chỉ cộng reconstruction error tại vị trí mask.
- [x] Stage A tắt CR bằng `beta=0`; Stage B dùng `6*MSM + CR`.
- [x] Pretrained file lưu riêng `model.encoder.state_dict()`.
- [x] LP đóng băng encoder; FT mở encoder với learning rate thấp hơn head.
- [x] Target val dùng để chọn state tốt nhất, target test chỉ đánh giá sau đó.
- [x] K-shot sampling lấy cùng số mẫu cho mỗi lớp theo seed.

### Phát hiện cần chú ý

1. **Validation leakage trong SSL pretraining.** `X_val = X_all` và
   `X_train = X_all` trong `run_crosshar_pretrain.py`, nên `val_loss` không đo
   tổng quát hóa trên mẫu chưa thấy. Cần kiểm tra lại nếu dùng “best checkpoint”
   như bằng chứng về generalization.
2. **`dataset_all.pt` có thể chứa test samples.** UCI-HAR ghép train và test
   vào `dataset_all`; MotionSense ghép train/val/test. Điều này không dùng nhãn
   trong loss, nhưng là transductive exposure nếu báo cáo transfer test như hoàn
   toàn chưa thấy trong pretraining.
3. **Comment/shape CNN-Transformer không khớp code.** `cnn_transformer.py`
   ghi length 32 nhưng các pooling bị comment; trainer phát hiện động nên cần
   test output thực tế trước khi diễn giải kiến trúc.
4. **Tên `x_neg` có thể gây hiểu nhầm.** Nó là một view của positive pair trong
   loss, không phải sample negative độc lập.
5. **`CrossHARClassifier` không nằm trên đường chạy transfer.** Downstream dùng
   `HARClassifier + ClassifierHead`; nếu mục tiêu là đánh giá context module của
   CrossHAR, cần viết test/runner riêng, không suy ra từ kết quả hiện tại.
6. **Seed chưa hoàn toàn được kiểm soát.** Pretraining chỉ gọi
   `torch.manual_seed(args.seed)`; không thấy seed Python/NumPy/CUDA hoặc
   DataLoader generator trong entry point này.
7. **Checkpoint không mang metadata.** Load strict phụ thuộc người chạy truyền
   đúng source/backbone/in_channels; file không tự ghi các cấu hình đó.
8. **Pretraining theo từng dataset, không phải cross-domain pretraining.**
   Vòng `for ds_name` chạy mỗi dataset độc lập và không trộn domain.

### Cần kiểm chứng thêm

- Cần chạy forward thực tế trên từng backbone để xác nhận mọi output length,
  đặc biệt `standard` (`L_f=8`) và `cnn_transformer` (`L_f` theo code hiện tại).
- Cần đối chiếu công thức `forward_contrastive` với định nghĩa loss của tài liệu
  CrossHAR tham chiếu, nhất là ý nghĩa của `x^-`, `x^+` và việc dùng toàn bộ
  in-batch non-self entries.
- Cần kiểm tra raw files có đúng channel semantics, đơn vị và sampling rate
  trước khi kết luận InstanceNorm loại bỏ/giữ thông tin vật lý phù hợp.
- Cần kiểm tra số lượng mỗi lớp sau lọc `labels < 5` và khả năng thiếu lớp trong
  target; `sample_k_shot` sẽ raise nếu một lớp không đủ `k` mẫu.
- Cần kiểm tra output `classification_report` và confusion matrix khi một lớp
  vắng trong prediction; code dùng `zero_division=0` nhưng vẫn cần diễn giải.
- Cần kiểm tra compatibility nếu checkpoint CrossHAR được nạp vào backbone
  downstream khác hoặc nếu thay đổi channel count.
- Cần đối chiếu đường dẫn Kaggle/local và quyền ghi output trước khi chạy các
  script; `OUTPUT_ROOT` của cross-domain đổi sang `/kaggle/working` trên Kaggle.

## 16. Tài liệu tham khảo và bằng chứng trong repository

Các nguồn nội bộ dùng để lập tài liệu:

- `README.md`: mô tả mục tiêu Cross-Domain HAR, các phương pháp và pipeline tổng thể.
- `scripts/02_pretrain_ssl/run_crosshar_pretrain.py`: entry point CrossHAR.
- `engines/pretrain_ssl/crosshar_sequential_trainer.py`: training schedule,
  optimizer, validation và checkpoint.
- `models/ssl/masked/crosshar_model.py`: kiến trúc và loss.
- `datasets/crosshar_dataset.py`: physical augmentation, normalization và views.
- `scripts/03_cross_domain/run_few_shot.py`,
  `scripts/03_cross_domain/run_cross_domain.py`: transfer evaluation.
- `engines/transfer/finetune_trainer.py`,
  `engines/evaluation/evaluator.py`: downstream optimization và metrics.
- Các file preprocessing/config trong `scripts/00_preprocess/` và `config/`.

Repository có tên tài liệu tham chiếu CrossHAR trong thư mục `document/references`
ở trạng thái lịch sử, nhưng tại thời điểm khảo sát file đó không còn trong
working tree. Vì vậy tài liệu này không tự thêm citation tác giả/DOI/URL và
không dùng bài báo để suy ra những thành phần không tồn tại trong code.
