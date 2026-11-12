# CrossDomain-HAR-SSL

Framework thực nghiệm cho **Self-Supervised Learning (SSL)** và **few-shot
cross-domain Human Activity Recognition (HAR)** trên dữ liệu cảm biến quán
tính. Repository bao gồm tiền xử lý dữ liệu, pre-training SSL, chuyển giao
sang miền đích, đánh giá linear probing/full fine-tuning và các công cụ tổng
hợp kết quả.

> **Trạng thái hiện tại:** backbone chính của dự án là `standard`, tức
> **1D-CNN gồm 4 convolution block**. Đây cũng là backbone được sử dụng trong
> các thí nghiệm TS-TCC/TSTCC chính (`--backbone standard`). Các backbone khác
> chỉ dùng cho ablation, thử nghiệm hoặc tương thích checkpoint.

## Mục lục

- [1. Mục tiêu và pipeline](#1-mục-tiêu-và-pipeline)
- [2. Dữ liệu và bài toán](#2-dữ-liệu-và-bài-toán)
- [3. Backbone và các phương pháp SSL](#3-backbone-và-các-phương-pháp-ssl)
- [4. Cấu trúc repository](#4-cấu-trúc-repository)
- [5. Cài đặt](#5-cài-đặt)
- [6. Chuẩn bị dữ liệu](#6-chuẩn-bị-dữ-liệu)
- [7. Pre-training SSL](#7-pre-training-ssl)
- [8. Few-shot cross-domain transfer](#8-few-shot-cross-domain-transfer)
- [9. Đánh giá, aggregate và báo cáo](#9-đánh-giá-aggregate-và-báo-cáo)
- [10. Checkpoint và output](#10-checkpoint-và-output)
- [11. Tái lập thí nghiệm](#11-tái-lập-thí-nghiệm)
- [12. Tài liệu tham khảo trong repository](#12-tài-liệu-tham-khảo-trong-repository)

## 1. Mục tiêu và pipeline

Mục tiêu là học biểu diễn từ dữ liệu chưa gán nhãn ở **source domain**, sau đó
chuyển biểu diễn sang **target domain** với ít mẫu nhãn. Pipeline chính:

```text
Raw dataset
    │
    ▼
Preprocess: chuẩn hóa nhãn + resample/lọc + cửa sổ trượt
    │
    ▼
SSL pre-training trên dataset_all.pt của source
    │
    ▼
Nạp encoder vào HARClassifier
    │
    ├── Linear probing: đóng băng toàn bộ encoder
    ├── Partial fine-tuning: mở một số block cuối
    └── Full fine-tuning: mở toàn bộ encoder
    │
    ▼
Macro F1, accuracy, confusion matrix và báo cáo theo seed
```

Các câu hỏi được thiết kế để trả lời:

1. Phương pháp SSL nào chuyển giao tốt hơn giữa các sensor/domain?
2. Biểu diễn học được cần bao nhiêu mẫu nhãn ở target?
3. Linear probing có phản ánh chất lượng biểu diễn trước khi fine-tuning hay
   không?
4. Domain `hhar_watch` khác pocket/phone về mặt động học đến mức nào?
5. Chi phí mô hình thay đổi ra sao giữa các backbone?

## 2. Dữ liệu và bài toán

### 2.1. Input và cửa sổ thời gian

- Input chuẩn: `(batch, 6, 128)`.
- 6 kênh: 3 trục accelerometer và 3 trục gyroscope/rotation rate.
- Window length: 128 timestep.
- Stride: 64 timestep, tương đương overlap 50%.
- Tần số mục tiêu: 50 Hz, mỗi cửa sổ dài khoảng 2,56 giây.

### 2.2. Dataset được đăng ký trong code

| Tên CLI | Dataset/domain | Đặc điểm |
|---|---|---|
| `uci_har` | UCI-HAR | 6 lớp gốc, sensor ở eo; downstream cross-domain dùng 5 lớp |
| `motion_sense` | MotionSense | 6 lớp gốc, iPhone ở eo; bỏ `Jogging` khi benchmark chung |
| `hhar_phone` | HHAR phone | điện thoại, dữ liệu timestamp không đều được resample/lọc |
| `hhar_watch` | HHAR watch | đồng hồ đeo cổ tay, được phân tích như bottleneck vật lý riêng |

Năm lớp chung trong cross-domain benchmark:

| ID | Lớp |
|---:|---|
| 0 | Walking |
| 1 | Upstairs |
| 2 | Downstairs |
| 3 | Sitting |
| 4 | Standing |

`Laying` của UCI-HAR và `Jogging` của MotionSense vẫn được giữ trong cấu hình
dataset, nhưng bị lọc khỏi bài toán năm lớp để nhãn giữa các domain tương thích.

### 2.3. Subject split và xử lý tín hiệu

- **MotionSense:** train subjects 1--14, validation 15--18, test 19--24.
- **UCI-HAR:** dùng split train/test gốc và tách subjects 27--30 từ train làm
  validation.
- **HHAR:** 9 subjects được tách train/validation/test theo cấu hình
  `HHARConfig`: `[a,b,d,e,h]` / `[f,g]` / `[c,i]`.
- **HHAR preprocessing:** resample về 50 Hz, cắt đoạn khi gap lớn hơn 1 giây,
  low-pass Butterworth cutoff 10 Hz, order 4.

Mỗi file `.pt` lưu sample, label và thông tin subject để kiểm tra data leakage.

## 3. Backbone và các phương pháp SSL

### 3.1. Backbone chính: `standard` 1D-CNN 4 block

`models/encoders/cnn1d.py` định nghĩa backbone dùng cho pipeline chính và
được chọn mặc định bởi `build_encoder()` cũng như transfer model.

| Block | Input channels | Output channels | Kernel | Pool | Output shape với input `(B,6,128)` |
|---:|---:|---:|---:|---:|---|
| 1 | 6 | 32 | 7 | 2 | `(B,32,64)` |
| 2 | 32 | 64 | 5 | 2 | `(B,64,32)` |
| 3 | 64 | 96 | 5 | 2 | `(B,96,16)` |
| 4 | 96 | 128 | 3 | 2 | `(B,128,8)` |

Mỗi block gồm `Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d`. Encoder trả về
feature map `(B,128,8)`; classifier/projection head xử lý feature này theo
kiến trúc SSL hoặc downstream.

**Lưu ý quan trọng:** TS-TCC/TSTCC không thay backbone chính bằng encoder
khác. Khi chạy TSTCC với `--backbone standard`, temporal contrasting và
contextual contrasting đều được tối ưu trên encoder **standard 4 block**.

### 3.2. Backbone tùy chọn

`models/encoders/builder.py` hiện hỗ trợ:

- `standard`: backbone chính, 4 block 1D-CNN.
- `cnn_transformer`: CNN kết hợp Transformer.
- `vit_1d`: patch embedding và Transformer cho chuỗi 1D.
- `cnn_lstm`: 1D-CNN kết hợp một LSTM một chiều.
- `tstcc`: encoder 1D-CNN theo implementation TS-TCC gốc, 3 block.

`tstcc` là lựa chọn kiến trúc tùy chọn/legacy; không phải backbone chuẩn của
các thí nghiệm chính. Không được trộn checkpoint giữa các backbone.

### 3.3. Các phương pháp SSL

| CLI `--method` | Thành phần chính | Checkpoint method |
|---|---|---|
| `tstcc` | Temporal Contrasting + Contextual Contrasting (TS-TCC) | `tstcc` |
| `swav` | weak/strong views, prototype, Sinkhorn và swapped prediction | `swav` |
| `crosshar` | augmentation hoán vị 3D và sequential domain-aware training | `crosshar` |
| `tstcc_swav` | hybrid TS-TCC + SwAV; có thể bật thêm contextual contrastive loss | `tstcc_swav` |

Thông số TS-TCC mặc định nằm trong
[`config/methods/tstcc_config.py`](config/methods/tstcc_config.py): 3
timesteps dự đoán, hidden dimension 64, Transformer depth 4, 4 attention
heads, temperature 0.2, 60 epoch, batch size 64. Thông số hybrid nằm trong
[`config/methods/tstcc_swav_config.py`](config/methods/tstcc_swav_config.py).

Tài liệu triển khai chi tiết:

- [`document/ssl_implementation_docs/TS_TCC.md`](document/ssl_implementation_docs/TS_TCC.md)
- [`document/ssl_implementation_docs/SwAV_SSL.md`](document/ssl_implementation_docs/SwAV_SSL.md)
- [`document/ssl_implementation_docs/CROSSHAR.md`](document/ssl_implementation_docs/CROSSHAR.md)
- [`document/ssl_implementation_docs/TSTCC_SWAV.md`](document/ssl_implementation_docs/TSTCC_SWAV.md)

## 4. Cấu trúc repository

```text
CrossDomain-HAR-SSL/
├── config/
│   ├── data_paths.py
│   ├── datasets/                 # registry và cấu hình UCI/MotionSense/HHAR
│   └── methods/                  # hyperparameter SSL
├── data/
│   ├── raw/                      # dữ liệu gốc, không commit
│   ├── processed/                # train.pt/val.pt/test.pt/dataset_all.pt
│   └── explore_data/             # notebook khảo sát dữ liệu
├── datasets/                     # dataset wrapper và augmentation
├── engines/
│   ├── pretrain_ssl/             # trainer TSTCC, SwAV, CrossHAR, hybrid
│   ├── transfer/                 # train/evaluate downstream
│   ├── evaluation/               # metric và few-shot report
│   └── supervised/               # baseline supervised
├── models/
│   ├── encoders/                 # standard 4 block và backbone tùy chọn
│   ├── heads/                    # classifier/projection head
│   └── ssl/                      # model TSTCC, SwAV, CrossHAR, hybrid
├── scripts/
│   ├── 00_preprocess/
│   ├── 01_baseline/
│   ├── 02_pretrain_ssl/
│   ├── 03_cross_domain/
│   ├── 04_results/
│   └── 05_analysis/
├── checkpoints/                  # checkpoint SSL, không commit file lớn
├── outputs_evaluation/           # JSON, CSV, confusion matrix, aggregate
├── document/                     # báo cáo và tài liệu triển khai
├── requirements.txt
└── README.md
```

## 5. Cài đặt

Yêu cầu khuyến nghị: Linux, Python 3.11, PyTorch 2.x và GPU CUDA cho
pre-training. Cài dependency:

```bash
git clone https://github.com/Dungx1107/CrossDomain-HAR-SSL.git
cd CrossDomain-HAR-SSL

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Hoặc dùng `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Kiểm tra thiết bị:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

Khi chạy trên Kaggle, code tự dùng `/kaggle/working` làm output root nếu
`KAGGLE_KERNEL_RUN_TYPE` tồn tại.

## 6. Chuẩn bị dữ liệu

Đặt dữ liệu gốc dưới `data/raw/` theo cấu hình trong `config/datasets/`:

```text
data/raw/
├── uci_har/
├── motion_sense/
└── hhar/
```

Chạy tiền xử lý:

```bash
python scripts/00_preprocess/preprocess_uci_har.py
python scripts/00_preprocess/preprocess_motionsense.py
python scripts/00_preprocess/preprocess_hhar.py
python scripts/00_preprocess/check_data_quality.py
```

Kết quả tối thiểu:

```text
data/processed/
├── uci_har/{train,val,test,dataset_all}.pt
├── motion_sense/{train,val,test,dataset_all}.pt
└── hhar/
    ├── hhar_phone/{train,val,test,dataset_all}.pt
    └── hhar_watch/{train,val,test,dataset_all}.pt
```

Đường dẫn thực tế được tạo tập trung trong
[`config/datasets/registry.py`](config/datasets/registry.py); không hard-code
đường dẫn dataset trong script mới.

## 7. Pre-training SSL

### 7.1. TS-TCC/TSTCC với backbone standard 4 block

```bash
python scripts/02_pretrain_ssl/run_pretrain_tstcc.py \
  --backbone standard \
  --datasets uci_har motion_sense \
  --epochs 60 \
  --batch_size 64 \
  --tag bs64
```

`--tag` là tùy chọn để tách các lần chạy. Nếu bỏ tag, checkpoint dùng đường
dẫn mặc định.

### 7.2. SwAV

```bash
python scripts/02_pretrain_ssl/run_pretrain_swav.py \
  --backbone standard \
  --datasets uci_har motion_sense \
  --epochs 60 \
  --batch_size 64 \
  --tag bs64
```

### 7.3. CrossHAR

CrossHAR hiện hỗ trợ `standard` và `cnn_transformer`; backbone chính vẫn có
thể chạy bằng `--backbone standard`.

```bash
python scripts/02_pretrain_ssl/run_pretrain_crosshar.py \
  --backbone standard \
  --datasets motion_sense uci_har \
  --epochs 60 \
  --warmup_msm_epochs 15 \
  --batch_size 64
```

Script mặc định mở rộng dữ liệu bằng 6 phép hoán vị 3D (`--expand_6x`).

### 7.4. Hybrid TS-TCC x SwAV

```bash
python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
  --backbone standard \
  --datasets uci_har motion_sense \
  --epochs 60 \
  --batch_size 64 \
  --complexity
```

Chạy ablation có contextual contrastive loss:

```bash
python scripts/02_pretrain_ssl/run_pretrain_tstcc_swav.py \
  --backbone standard \
  --variant cc \
  --lambda_cc 0.7 \
  --datasets uci_har motion_sense
```

Trước khi chạy thật, có thể dùng `--epochs 1 --datasets uci_har` để kiểm tra
đường dẫn dữ liệu, kích thước tensor và checkpoint.

## 8. Few-shot cross-domain transfer

`run_few_shot.py` hỗ trợ các protocol:

- `full`: linear probing, đóng băng toàn bộ encoder;
- `partial:N`: partial fine-tuning, mở `N` block cuối;
- `none`: full fine-tuning, mở toàn bộ encoder.

Ví dụ TSTCC với backbone **standard 4 block**:

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method tstcc \
  --backbone standard \
  --pretrain_tag bs64 \
  --pairs motion_sense:uci_har uci_har:motion_sense \
  --k_shots 10 30 50 100 150 200 \
  --seeds 42 100 2024 7 99 \
  --epochs 40 \
  --batch_size 16 \
  --freeze_modes full partial:2 none
```

Ví dụ đánh giá bottleneck wrist:

```bash
python scripts/03_cross_domain/run_few_shot.py \
  --method swav \
  --backbone standard \
  --pairs motion_sense:hhar_watch uci_har:hhar_watch \
  --k_shots 10 30 50 100 \
  --freeze_modes full none
```

Các method hợp lệ trong CLI hiện tại là `tstcc`, `swav`, `crosshar` và
`tstcc_swav`. Backbone hợp lệ là `standard`, `cnn_transformer`, `vit_1d`,
`tstcc` và `cnn_lstm`.

Fraction-based adaptation là pipeline legacy và hiện nhận các method
`contrastive`, `prototype` hoặc `crosshar`:

```bash
python scripts/03_cross_domain/run_cross_domain.py \
  --method contrastive \
  --backbone standard \
  --fractions 0.01 0.05 0.10 \
  --pairs motion_sense:uci_har \
  --seeds 42 100 2024 7 99
```

## 9. Đánh giá, aggregate và báo cáo

Tổng hợp K-shot và fraction:

```bash
python scripts/04_results/aggregate_k_shot.py
python scripts/04_results/aggregate_fraction.py
```

Sinh báo cáo Markdown từ cây kết quả:

```bash
python scripts/04_results/cross_domain_har_reporter.py \
  --input outputs_evaluation/cross_k_shot \
  --output section/Bao_Cao_Cross_Domain_HAR.md
```

Các công cụ trực quan hóa và phân tích:

```bash
python scripts/04_results/plot_charts.py
python scripts/04_results/plot_hybrid_grid.py
python scripts/05_analysis/plot_k_shot_evaluation.py
python scripts/05_analysis/plot_protocol_split.py
python scripts/05_analysis/plot_tsne.py
python scripts/05_analysis/visualize_cross_k_shot.py
python scripts/05_analysis/print_k_shot_table.py
```

Khi so sánh nhiều method, chỉ dùng các source-target pair có cùng shot,
protocol và backbone. `N/A` là thiếu record, không phải điểm 0. Báo cáo nên
giữ `mean ± std` theo seed, tách riêng target `hhar_watch`, và kiểm tra các
trường hợp `std > 5` điểm phần trăm hoặc linear probing vượt full
fine-tuning.

Các biểu đồ nên có trong báo cáo:

1. **100-shot grouped bars:** x = method, y = macro F1, hue = backbone,
   error bar = std.
2. **Shot learning curve:** x = số shot, y = macro F1, nhóm theo
   method/backbone/protocol.
3. **Source-target heatmap:** hàng = source, cột = target, giá trị = cấu hình
   tốt nhất.
4. **Seed-level boxplot:** x = method, y = metric từng seed, hue = shot.

Nếu file seed-level không tồn tại thì phải ghi rõ boxplot không khả dụng,
không thay bằng một proxy.

## 10. Checkpoint và output

Checkpoint dùng quy ước tập trung trong
[`utils/checkpoint_paths.py`](utils/checkpoint_paths.py):

```text
checkpoints/ssl_pretrain/<method>/<dataset>/<backbone>[_<tag>]/
└── <method>_<backbone>_encoder_pretrained_<dataset>.pt
```

Ví dụ:

```text
checkpoints/ssl_pretrain/tstcc/uci_har/standard_bs64/
└── tstcc_standard_encoder_pretrained_uci_har.pt
```

Kết quả few-shot:

```text
outputs_evaluation/cross_k_shot/<method>/<backbone>[_<tag>]/
└── <source>_to_<target>/
    ├── experiment_metadata.json
    └── <k>_shot/
        ├── linear_probing/
        ├── partial_<N>blocks/
        └── full_finetuning/
```

Mỗi protocol thường gồm `summary_*.json`,
`detailed_all_seeds.json` và `aggregated_confusion_matrix.png`.

## 11. Tái lập thí nghiệm

1. Ghi lại commit Git, Python, PyTorch, CUDA, GPU và device.
2. Giữ nguyên dataset, source-target pairs, method, **backbone**, tag, shot,
   protocol, epoch, batch size và seed.
3. Kiểm tra checkpoint có đúng method/dataset/backbone trước khi transfer.
4. Không dùng `Laying` hoặc `Jogging` trong benchmark năm lớp.
5. So sánh method trên cùng giao source-target pair.
6. Giữ `experiment_metadata.json`, summary và detailed seed records cùng nhau.
7. Không commit raw data, checkpoint lớn hoặc token/API credentials.

## 12. Tài liệu tham khảo trong repository

- [`document/ssl_implementation_docs/`](document/ssl_implementation_docs/) -
  tài liệu triển khai SSL.
- [`document/0_reports/`](document/0_reports/) - báo cáo học thuật và kết quả.
- [`scripts/05_analysis/`](scripts/05_analysis/) - notebook/script phân tích.
- [`config/datasets/`](config/datasets/) - nguồn sự thật cho nhãn, split và
  đường dẫn dữ liệu.
- [`models/encoders/cnn1d.py`](models/encoders/cnn1d.py) - backbone standard
  4 block.
- [`models/encoders/builder.py`](models/encoders/builder.py) - factory của
  toàn bộ backbone.

Khi sử dụng repository, cần trích dẫn dataset và bài báo gốc tương ứng, đồng
thời ghi lại commit đã dùng để tạo kết quả.
