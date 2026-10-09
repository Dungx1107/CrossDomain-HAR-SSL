# Báo cáo Cross-Domain HAR K-shot

## 1. Tóm tắt điều hành
Kết quả gồm 312 bản ghi từ 96 cấu hình chuyển giao. Phân tích overall chỉ sử dụng các cặp pocket-level và loại bỏ 76 bản ghi có target `hhar_watch`.
So sánh phương pháp được tính trên giao các cặp chuyển giao có đủ dữ liệu cho toàn bộ phương pháp, không nội suy missing data.
Linear probing được dùng như phép đo chất lượng biểu diễn frozen features; full fine-tuning được diễn giải riêng vì có khả năng thích nghi representation.
Hai phát hiện cần lưu ý là standard vẫn đạt 82.16% FT ở pocket-level trong khi cnn_transformer đạt 80.37%, và target hhar_watch chỉ đạt 33.70% so với 70.38% ở pocket-level.
Ngoài ra, dữ liệu quan sát được không cho phép kết luận overfitting hoặc ý nghĩa thống kê; các độ lệch chuẩn được dùng như mô tả độ ổn định, không thay thế kiểm định giả thuyết.
Các cảnh báo về LP > FT và độ ổn định seed được giữ nguyên trong mục self-validation.

## 2. Bảng so sánh tổng hợp

| Method | Backbone | Shot | Protocol | Macro F1 (%) |
| --- | --- | --- | --- | --- |
| contrastive | cnn_transformer | 100_shot | full_finetuning | 79.65 ± 2.86 |
| contrastive | cnn_transformer | 100_shot | linear_probing | 54.61 ± 2.65 |
| contrastive | standard | 100_shot | full_finetuning | 82.05 ± 2.58 |
| contrastive | standard | 100_shot | linear_probing | 72.20 ± 2.64 |
| crosshar | cnn_transformer | 100_shot | full_finetuning | 80.33 ± 2.77 |
| crosshar | cnn_transformer | 100_shot | linear_probing | 61.79 ± 1.89 |
| crosshar | cnn_transformer | 50_shot | full_finetuning | 76.55 ± 2.27 |
| crosshar | cnn_transformer | 50_shot | linear_probing | 58.28 ± 2.25 |
| crosshar | standard | 100_shot | full_finetuning | 82.11 ± 2.17 |
| crosshar | standard | 100_shot | linear_probing | 73.68 ± 1.68 |
| crosshar | standard | 50_shot | full_finetuning | 79.46 ± 2.92 |
| crosshar | standard | 50_shot | linear_probing | 69.81 ± 1.63 |
| prototype | cnn_transformer | 100_shot | full_finetuning | 81.13 ± 2.44 |
| prototype | cnn_transformer | 100_shot | linear_probing | 69.09 ± 2.50 |
| prototype | cnn_transformer | 10_shot | full_finetuning | 65.63 ± 5.06 |
| prototype | cnn_transformer | 10_shot | linear_probing | 55.13 ± 3.87 |
| prototype | cnn_transformer | 20_shot | full_finetuning | 73.37 ± 4.36 |
| prototype | cnn_transformer | 20_shot | linear_probing | 60.94 ± 3.33 |
| prototype | cnn_transformer | 50_shot | full_finetuning | 77.59 ± 2.91 |
| prototype | cnn_transformer | 50_shot | linear_probing | 65.64 ± 2.93 |
| prototype | standard | 100_shot | full_finetuning | 82.32 ± 2.44 |
| prototype | standard | 100_shot | linear_probing | 70.72 ± 2.61 |
| prototype | standard | 10_shot | full_finetuning | 65.24 ± 4.84 |
| prototype | standard | 10_shot | linear_probing | 56.27 ± 4.39 |
| prototype | standard | 20_shot | full_finetuning | 74.42 ± 3.21 |
| prototype | standard | 20_shot | linear_probing | 61.07 ± 3.17 |
| prototype | standard | 50_shot | full_finetuning | 79.79 ± 2.88 |
| prototype | standard | 50_shot | linear_probing | 65.96 ± 3.04 |

## 3. Phân tích ba trục

### 3.1 Trục phương pháp
Các bảng dưới đây chỉ dùng giao cặp pocket-level của toàn bộ phương pháp hiện diện.

**standard, 10_shot, full_finetuning: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 10_shot, linear_probing: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 20_shot, full_finetuning: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 20_shot, linear_probing: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 50_shot, full_finetuning: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 50_shot, linear_probing: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 100_shot, full_finetuning: 7/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**standard, 100_shot, linear_probing: 7/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 10_shot, full_finetuning: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 10_shot, linear_probing: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 20_shot, full_finetuning: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 20_shot, linear_probing: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 50_shot, full_finetuning: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 50_shot, linear_probing: 0/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 100_shot, full_finetuning: 7/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

**cnn_transformer, 100_shot, linear_probing: 7/9 cặp đủ dữ liệu**

| Method | F1 (%) |
| --- | --- |
| crosshar | N/A |
| prototype | N/A |
| contrastive | N/A |

Nhận xét: CrossHAR, Prototype và Contrastive được so sánh trên cùng transfer pairs; LP là trục chính để đánh giá frozen features, còn FT phản ánh cả khả năng thích nghi.

### 3.2 Trục kiến trúc

| Backbone | Shot | Protocol | F1 pocket (%) |
| --- | --- | --- | --- |
| standard | 10_shot | full_finetuning | 65.24 |
| standard | 10_shot | linear_probing | 56.27 |
| standard | 20_shot | full_finetuning | 74.42 |
| standard | 20_shot | linear_probing | 61.07 |
| standard | 50_shot | full_finetuning | 79.65 |
| standard | 50_shot | linear_probing | 67.64 |
| standard | 100_shot | full_finetuning | 82.16 |
| standard | 100_shot | linear_probing | 72.08 |
| cnn_transformer | 10_shot | full_finetuning | 65.63 |
| cnn_transformer | 10_shot | linear_probing | 55.13 |
| cnn_transformer | 20_shot | full_finetuning | 73.37 |
| cnn_transformer | 20_shot | linear_probing | 60.94 |
| cnn_transformer | 50_shot | full_finetuning | 77.14 |
| cnn_transformer | 50_shot | linear_probing | 62.42 |
| cnn_transformer | 100_shot | full_finetuning | 80.37 |
| cnn_transformer | 100_shot | linear_probing | 61.83 |

Nhận xét: cần kiểm tra chênh lệch `standard` và `cnn_transformer` tại 10-shot và LP; không kết luận Transformer mong manh nếu giao cặp hoặc seed không đủ.

### 3.3 Trục giao thức

| Method | Backbone | Shot | Matched pairs | FT F1 (%) | LP F1 (%) | Delta (FT-LP) |
| --- | --- | --- | --- | --- | --- | --- |
| contrastive | cnn_transformer | 100_shot | 9 | 79.65 | 54.61 | +25.04 |
| contrastive | standard | 100_shot | 9 | 82.05 | 72.20 | +9.85 |
| crosshar | cnn_transformer | 100_shot | 7 | 80.33 | 61.79 | +18.55 |
| crosshar | cnn_transformer | 50_shot | 7 | 76.55 | 58.28 | +18.26 |
| crosshar | standard | 100_shot | 7 | 82.11 | 73.68 | +8.43 |
| crosshar | standard | 50_shot | 7 | 79.46 | 69.81 | +9.65 |
| prototype | cnn_transformer | 100_shot | 9 | 81.13 | 69.09 | +12.05 |
| prototype | cnn_transformer | 10_shot | 9 | 65.63 | 55.13 | +10.50 |
| prototype | cnn_transformer | 20_shot | 9 | 73.37 | 60.94 | +12.43 |
| prototype | cnn_transformer | 50_shot | 9 | 77.59 | 65.64 | +11.95 |
| prototype | standard | 100_shot | 9 | 82.32 | 70.72 | +11.61 |
| prototype | standard | 10_shot | 9 | 65.24 | 56.27 | +8.98 |
| prototype | standard | 20_shot | 9 | 74.42 | 61.07 | +13.35 |
| prototype | standard | 50_shot | 9 | 79.79 | 65.96 | +13.83 |

Delta được tính trên cùng source-target pair có đủ cả FT và LP. Giá trị dương cho thấy full fine-tuning tốt hơn linear probing; vì vậy không bị ảnh hưởng bởi khác biệt coverage.

### 3.4 Xu hướng theo shot

| Method | Backbone | Shot | Protocol | Records | Mean F1 (%) |
| --- | --- | --- | --- | --- | --- |
| crosshar | standard | 50_shot | full_finetuning | 7 | 79.46 |
| crosshar | standard | 50_shot | linear_probing | 7 | 69.81 |
| crosshar | standard | 100_shot | full_finetuning | 7 | 82.11 |
| crosshar | standard | 100_shot | linear_probing | 7 | 73.68 |
| crosshar | cnn_transformer | 50_shot | full_finetuning | 7 | 76.55 |
| crosshar | cnn_transformer | 50_shot | linear_probing | 7 | 58.28 |
| crosshar | cnn_transformer | 100_shot | full_finetuning | 7 | 80.33 |
| crosshar | cnn_transformer | 100_shot | linear_probing | 7 | 61.79 |
| prototype | standard | 10_shot | full_finetuning | 9 | 65.24 |
| prototype | standard | 10_shot | linear_probing | 9 | 56.27 |
| prototype | standard | 20_shot | full_finetuning | 9 | 74.42 |
| prototype | standard | 20_shot | linear_probing | 9 | 61.07 |
| prototype | standard | 50_shot | full_finetuning | 9 | 79.79 |
| prototype | standard | 50_shot | linear_probing | 9 | 65.96 |
| prototype | standard | 100_shot | full_finetuning | 9 | 82.32 |
| prototype | standard | 100_shot | linear_probing | 9 | 70.72 |
| prototype | cnn_transformer | 10_shot | full_finetuning | 9 | 65.63 |
| prototype | cnn_transformer | 10_shot | linear_probing | 9 | 55.13 |
| prototype | cnn_transformer | 20_shot | full_finetuning | 9 | 73.37 |
| prototype | cnn_transformer | 20_shot | linear_probing | 9 | 60.94 |
| prototype | cnn_transformer | 50_shot | full_finetuning | 9 | 77.59 |
| prototype | cnn_transformer | 50_shot | linear_probing | 9 | 65.64 |
| prototype | cnn_transformer | 100_shot | full_finetuning | 9 | 81.13 |
| prototype | cnn_transformer | 100_shot | linear_probing | 9 | 69.09 |
| contrastive | standard | 100_shot | full_finetuning | 9 | 82.05 |
| contrastive | standard | 100_shot | linear_probing | 9 | 72.20 |
| contrastive | cnn_transformer | 100_shot | full_finetuning | 9 | 79.65 |
| contrastive | cnn_transformer | 100_shot | linear_probing | 9 | 54.61 |

Xu hướng được mô tả theo các mức shot quan sát được. Không kết luận overfitting chỉ từ việc F1 giảm ở một shot; cần learning curve theo seed và validation độc lập.

### 3.5 Trục nút thắt miền

| Cluster | Records | Mean F1 (%) |
| --- | --- | --- |
| Pocket-level (overall) | 236 | 70.38 |
| Target contains hhar_watch | 76 | 33.70 |

Target `hhar_watch` được tách độc lập vì cảm biến đeo cổ tay có động học khác cảm biến ở đùi/hông; chênh lệch này là physical domain shift.

## 4. Phân tích theo cặp chuyển giao

| Transfer pair | Records | Mean F1 (%) |
| --- | --- | --- |
| motionsense → uci_har | 28 | 79.71 |
| hhar_phone → uci_har | 20 | 78.23 |
| uci_har → motionsense | 28 | 76.68 |
| hhar_watch → motionsense | 28 | 76.65 |
| hhar_watch → uci_har | 28 | 76.25 |
| hhar_phone → motionsense | 20 | 71.82 |
| hhar_watch → hhar_phone | 28 | 59.47 |
| motionsense → hhar_phone | 28 | 59.27 |
| uci_har → hhar_phone | 28 | 57.99 |

Cặp đứng đầu được xem là dễ chuyển giao nhất trong các bản ghi pocket-level; cặp cuối được xem là khó nhất theo macro F1 trung bình. Xếp hạng này gộp method, backbone, shot và protocol nên không thay thế so sánh matched configuration.

## 5. Phân tích nút thắt vật lý

| Method | Backbone | Shot | Protocol | Target cluster | Macro F1 (%) |
| --- | --- | --- | --- | --- | --- |
| contrastive | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 39.75 ± 3.26 |
| contrastive | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 34.73 ± 2.83 |
| contrastive | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 36.22 ± 3.76 |
| contrastive | cnn_transformer | 100_shot | linear_probing | hhar_watch | 25.25 ± 3.19 |
| contrastive | cnn_transformer | 100_shot | linear_probing | hhar_watch | 30.10 ± 5.23 |
| contrastive | cnn_transformer | 100_shot | linear_probing | hhar_watch | 25.01 ± 4.08 |
| contrastive | standard | 100_shot | full_finetuning | hhar_watch | 36.81 ± 1.90 |
| contrastive | standard | 100_shot | full_finetuning | hhar_watch | 38.03 ± 3.45 |
| contrastive | standard | 100_shot | full_finetuning | hhar_watch | 38.98 ± 2.16 |
| contrastive | standard | 100_shot | linear_probing | hhar_watch | 44.84 ± 1.58 |
| contrastive | standard | 100_shot | linear_probing | hhar_watch | 31.90 ± 5.04 |
| contrastive | standard | 100_shot | linear_probing | hhar_watch | 36.66 ± 1.59 |
| crosshar | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 41.79 ± 4.18 |
| crosshar | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 39.82 ± 1.94 |
| crosshar | cnn_transformer | 100_shot | linear_probing | hhar_watch | 41.55 ± 13.22 |
| crosshar | cnn_transformer | 100_shot | linear_probing | hhar_watch | 33.48 ± 4.04 |
| crosshar | standard | 100_shot | full_finetuning | hhar_watch | 38.12 ± 1.38 |
| crosshar | standard | 100_shot | full_finetuning | hhar_watch | 36.52 ± 1.55 |
| crosshar | standard | 100_shot | linear_probing | hhar_watch | 32.42 ± 3.33 |
| crosshar | standard | 100_shot | linear_probing | hhar_watch | 38.75 ± 2.02 |
| prototype | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 40.82 ± 5.88 |
| prototype | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 37.75 ± 2.25 |
| prototype | cnn_transformer | 100_shot | full_finetuning | hhar_watch | 38.10 ± 2.49 |
| prototype | cnn_transformer | 100_shot | linear_probing | hhar_watch | 50.48 ± 14.05 |
| prototype | cnn_transformer | 100_shot | linear_probing | hhar_watch | 31.42 ± 5.67 |
| prototype | cnn_transformer | 100_shot | linear_probing | hhar_watch | 26.05 ± 6.02 |
| prototype | standard | 100_shot | full_finetuning | hhar_watch | 36.51 ± 1.07 |
| prototype | standard | 100_shot | full_finetuning | hhar_watch | 39.23 ± 1.75 |
| prototype | standard | 100_shot | full_finetuning | hhar_watch | 38.67 ± 2.51 |
| prototype | standard | 100_shot | linear_probing | hhar_watch | 46.16 ± 7.74 |
| prototype | standard | 100_shot | linear_probing | hhar_watch | 23.20 ± 3.05 |
| prototype | standard | 100_shot | linear_probing | hhar_watch | 32.48 ± 2.82 |
| prototype | cnn_transformer | 10_shot | full_finetuning | hhar_watch | 39.06 ± 13.04 |
| prototype | cnn_transformer | 10_shot | full_finetuning | hhar_watch | 19.11 ± 8.22 |
| prototype | cnn_transformer | 10_shot | full_finetuning | hhar_watch | 25.04 ± 4.43 |
| prototype | cnn_transformer | 10_shot | linear_probing | hhar_watch | 40.13 ± 6.53 |
| prototype | cnn_transformer | 10_shot | linear_probing | hhar_watch | 17.42 ± 6.02 |
| prototype | cnn_transformer | 10_shot | linear_probing | hhar_watch | 20.56 ± 10.56 |
| prototype | standard | 10_shot | full_finetuning | hhar_watch | 25.83 ± 7.95 |
| prototype | standard | 10_shot | full_finetuning | hhar_watch | 16.01 ± 5.20 |
| prototype | standard | 10_shot | full_finetuning | hhar_watch | 19.56 ± 3.40 |
| prototype | standard | 10_shot | linear_probing | hhar_watch | 24.52 ± 6.25 |
| prototype | standard | 10_shot | linear_probing | hhar_watch | 14.07 ± 2.42 |
| prototype | standard | 10_shot | linear_probing | hhar_watch | 16.88 ± 3.85 |
| prototype | cnn_transformer | 20_shot | full_finetuning | hhar_watch | 49.20 ± 8.91 |
| prototype | cnn_transformer | 20_shot | full_finetuning | hhar_watch | 36.46 ± 8.23 |
| prototype | cnn_transformer | 20_shot | full_finetuning | hhar_watch | 36.87 ± 2.99 |
| prototype | cnn_transformer | 20_shot | linear_probing | hhar_watch | 36.12 ± 7.07 |
| prototype | cnn_transformer | 20_shot | linear_probing | hhar_watch | 20.15 ± 10.21 |
| prototype | cnn_transformer | 20_shot | linear_probing | hhar_watch | 23.13 ± 7.57 |
| prototype | standard | 20_shot | full_finetuning | hhar_watch | 34.52 ± 14.49 |
| prototype | standard | 20_shot | full_finetuning | hhar_watch | 29.04 ± 12.22 |
| prototype | standard | 20_shot | full_finetuning | hhar_watch | 35.86 ± 7.00 |
| prototype | standard | 20_shot | linear_probing | hhar_watch | 24.25 ± 6.61 |
| prototype | standard | 20_shot | linear_probing | hhar_watch | 15.46 ± 3.72 |
| prototype | standard | 20_shot | linear_probing | hhar_watch | 24.66 ± 6.74 |
| crosshar | cnn_transformer | 50_shot | full_finetuning | hhar_watch | 47.12 ± 5.90 |
| crosshar | cnn_transformer | 50_shot | full_finetuning | hhar_watch | 41.22 ± 1.14 |
| crosshar | cnn_transformer | 50_shot | linear_probing | hhar_watch | 40.30 ± 14.39 |
| crosshar | cnn_transformer | 50_shot | linear_probing | hhar_watch | 33.75 ± 2.76 |
| crosshar | standard | 50_shot | full_finetuning | hhar_watch | 40.96 ± 2.16 |
| crosshar | standard | 50_shot | full_finetuning | hhar_watch | 39.53 ± 0.70 |
| crosshar | standard | 50_shot | linear_probing | hhar_watch | 33.27 ± 2.90 |
| crosshar | standard | 50_shot | linear_probing | hhar_watch | 32.57 ± 8.93 |
| prototype | cnn_transformer | 50_shot | full_finetuning | hhar_watch | 40.77 ± 4.20 |
| prototype | cnn_transformer | 50_shot | full_finetuning | hhar_watch | 37.06 ± 2.49 |
| prototype | cnn_transformer | 50_shot | full_finetuning | hhar_watch | 37.43 ± 1.90 |
| prototype | cnn_transformer | 50_shot | linear_probing | hhar_watch | 51.23 ± 11.55 |
| prototype | cnn_transformer | 50_shot | linear_probing | hhar_watch | 26.99 ± 9.64 |
| prototype | cnn_transformer | 50_shot | linear_probing | hhar_watch | 30.65 ± 7.72 |
| prototype | standard | 50_shot | full_finetuning | hhar_watch | 38.61 ± 2.60 |
| prototype | standard | 50_shot | full_finetuning | hhar_watch | 40.61 ± 2.63 |
| prototype | standard | 50_shot | full_finetuning | hhar_watch | 43.29 ± 6.04 |
| prototype | standard | 50_shot | linear_probing | hhar_watch | 47.34 ± 3.95 |
| prototype | standard | 50_shot | linear_probing | hhar_watch | 17.51 ± 4.42 |
| prototype | standard | 50_shot | linear_probing | hhar_watch | 35.80 ± 1.77 |

## 6. Phân tích ma trận nhầm lẫn

Đã kiểm tra 312 file `aggregated_confusion_matrix.png` và ma trận JSON đi kèm. Các hướng nhầm lẫn lớn nhất theo tỷ lệ hàng gộp là Walking → Upstairs, Upstairs → Walking, Downstairs → Walking.
Việc diễn giải được neo vào `aggregated_confusion_matrix.png` và ma trận đếm JSON; các lớp có tỷ lệ ngoài đường chéo cao cần được ưu tiên trong error analysis.

## 7. Self-validation

| Cảnh báo |
| --- |
| `contrastive/cnn_transformer/100_shot/linear_probing` có std 9.13% > 5%. |
| `contrastive/cnn_transformer/100_shot/linear_probing` có std 5.23% > 5%. |
| `contrastive/standard/100_shot/full_finetuning` có std 6.49% > 5%. |
| `contrastive/standard/100_shot/linear_probing` có std 7.02% > 5%. |
| `contrastive/standard/100_shot/linear_probing` có std 5.04% > 5%. |
| `crosshar/cnn_transformer/100_shot/full_finetuning` có std 5.49% > 5%. |
| `crosshar/cnn_transformer/100_shot/linear_probing` có std 13.22% > 5%. |
| `crosshar/cnn_transformer/50_shot/full_finetuning` có std 5.90% > 5%. |
| `crosshar/cnn_transformer/50_shot/linear_probing` có std 14.39% > 5%. |
| `crosshar/standard/50_shot/linear_probing` có std 8.93% > 5%. |
| `prototype/cnn_transformer/100_shot/full_finetuning` có std 5.88% > 5%. |
| `prototype/cnn_transformer/100_shot/linear_probing` có std 14.05% > 5%. |
| `prototype/cnn_transformer/10_shot/full_finetuning` có std 13.04% > 5%. |
| `prototype/cnn_transformer/10_shot/linear_probing` có std 6.53% > 5%. |
| `prototype/cnn_transformer/20_shot/full_finetuning` có std 8.91% > 5%. |
| `prototype/cnn_transformer/20_shot/linear_probing` có std 7.07% > 5%. |
| `prototype/cnn_transformer/50_shot/linear_probing` có std 11.55% > 5%. |
| `prototype/cnn_transformer/10_shot/full_finetuning` có std 6.97% > 5%. |
| `prototype/cnn_transformer/10_shot/linear_probing` có std 6.40% > 5%. |
| `prototype/cnn_transformer/10_shot/full_finetuning` có std 5.67% > 5%. |
| `prototype/cnn_transformer/10_shot/full_finetuning` có std 5.24% > 5%. |
| `prototype/cnn_transformer/20_shot/linear_probing` có std 5.36% > 5%. |
| `prototype/cnn_transformer/50_shot/full_finetuning` có std 5.09% > 5%. |
| `prototype/cnn_transformer/10_shot/full_finetuning` có std 12.66% > 5%. |
| `prototype/cnn_transformer/20_shot/full_finetuning` có std 12.19% > 5%. |
| `prototype/cnn_transformer/100_shot/linear_probing` có std 5.67% > 5%. |
| `prototype/cnn_transformer/10_shot/full_finetuning` có std 8.22% > 5%. |
| `prototype/cnn_transformer/10_shot/linear_probing` có std 6.02% > 5%. |
| `prototype/cnn_transformer/20_shot/full_finetuning` có std 8.23% > 5%. |
| `prototype/cnn_transformer/20_shot/linear_probing` có std 10.21% > 5%. |
| `prototype/cnn_transformer/50_shot/linear_probing` có std 9.64% > 5%. |
| `prototype/cnn_transformer/20_shot/linear_probing` có std 6.32% > 5%. |
| `prototype/cnn_transformer/100_shot/linear_probing` có std 6.02% > 5%. |
| `prototype/cnn_transformer/10_shot/linear_probing` có std 10.56% > 5%. |
| `prototype/cnn_transformer/20_shot/linear_probing` có std 7.57% > 5%. |
| `prototype/cnn_transformer/50_shot/linear_probing` có std 7.72% > 5%. |
| `prototype/standard/100_shot/linear_probing` có std 7.74% > 5%. |
| `prototype/standard/10_shot/full_finetuning` có std 7.95% > 5%. |
| `prototype/standard/10_shot/linear_probing` có std 6.25% > 5%. |
| `prototype/standard/20_shot/full_finetuning` có std 14.49% > 5%. |
| `prototype/standard/20_shot/linear_probing` có std 6.61% > 5%. |
| `prototype/standard/10_shot/linear_probing` có std 6.71% > 5%. |
| `prototype/standard/10_shot/linear_probing` có std 5.43% > 5%. |
| `prototype/standard/20_shot/full_finetuning` có std 5.22% > 5%. |
| `prototype/standard/20_shot/linear_probing` có std 6.04% > 5%. |
| `prototype/standard/10_shot/full_finetuning` có std 6.71% > 5%. |
| `prototype/standard/20_shot/full_finetuning` có std 6.46% > 5%. |
| `prototype/standard/10_shot/full_finetuning` có std 5.20% > 5%. |
| `prototype/standard/20_shot/full_finetuning` có std 12.22% > 5%. |
| `prototype/standard/10_shot/full_finetuning` có std 12.60% > 5%. |
| `prototype/standard/10_shot/linear_probing` có std 7.46% > 5%. |
| `prototype/standard/20_shot/full_finetuning` có std 7.00% > 5%. |
| `prototype/standard/20_shot/linear_probing` có std 6.74% > 5%. |
| `prototype/standard/50_shot/full_finetuning` có std 6.04% > 5%. |
| `prototype/standard/10_shot/linear_probing` có std 5.18% > 5%. |

## 8. Kết luận và đề xuất

- **Best configuration theo accuracy:** `prototype/standard/100_shot/full_finetuning` với accuracy 91.30%.
- **Best trade-off accuracy/compute:** chưa kết luận vì không tìm thấy metadata complexity; cần bổ sung FLOPs, tham số hoặc latency.
- Hướng phát triển: tăng số seed cho cấu hình std > 5%, báo cáo riêng pocket-level và wrist-level, và kiểm định trực tiếp giả thuyết Transformer tại 10-shot/LP.

## 9. Đề xuất visualization

- **Hình 1 — 100-shot comparison:** Grouped bar chart; X = Method × Backbone; Y = Macro F1 (%); hue = Protocol; error bars = std; thông điệp: chênh lệch bốn cấu hình tại 100-shot; đặt sau Bảng tổng hợp.
- **Hình 2 — Learning curve:** Line chart; X = Shot (10, 20, 50, 100); Y = Macro F1 (%); mỗi đường = một Method × Backbone × Protocol; thông điệp: lợi ích biên của nhãn; đặt trong phân tích shot.
- **Hình 3 — Transfer heatmap:** Heatmap Source × Target cho phương pháp tốt nhất; X = Target; Y = Source; màu = F1 (%); thông điệp: bất đối xứng source-target và physical bottleneck; đặt trong mục 4.
- **Hình 4 — Seed stability:** Boxplot F1 qua các seed; X = Method (facet theo backbone/protocol); Y = F1 (%); hue = Shot; thông điệp: độ ổn định và outlier giữa seed; đặt trong Self-validation.
- **Hình 5 — FT versus LP:** Bar chart; X = Method × Backbone; Y = Macro F1 (%); hue = Protocol; error bars = std; thông điệp: mức phụ thuộc vào thích nghi representation; đặt trong mục phân tích protocol.

## 10. Phân tích khoa học chuyên sâu

### 10.1 Bối cảnh, thiết kế và phạm vi suy luận
**Nhận xét.** Cây dữ liệu gồm 2 phương pháp chính (Prototype và Contrastive), cùng các bản ghi CrossHAR khi hiện diện, 2 backbone, 4 mức shot và 2 protocol; mỗi summary được đối chiếu với detailed_all_seeds.json. Các kết quả overall được tính trên 236 bản ghi pocket-level và 76 bản ghi wrist-target được giữ riêng.

**Giải thích.** Đây là bài toán transfer learning few-shot: domain đích vừa có ít nhãn vừa khác phân phối cảm biến. Summary cung cấp ước lượng trung bình và độ lệch chuẩn, còn dữ liệu seed cho phép kiểm tra sự dao động. Vì số cặp hiện diện không đồng đều theo method/shot, mọi nhận định xếp hạng phải nêu coverage; không được xem N/A là điểm bằng không.

### 10.2 Phương pháp: Prototype, Contrastive và CrossHAR
**Nhận xét.** Ở 100-shot pocket-level, các cấu hình đầy đủ cho thấy Prototype/standard đạt 82.32% ± 2.44 F1, Contrastive/standard đạt 82.05% ± 2.58, còn CrossHAR/standard đạt 82.11% ± 2.17 trên các coverage tương ứng. Chênh lệch dưới khoảng 0.3 điểm phần trăm trong nhóm này nhỏ hơn độ lệch chuẩn, do đó chưa đủ bằng chứng để khẳng định một phương pháp thắng thống kê.

**Giải thích.** Prototype tối ưu khoảng cách tới class centroid nên có thể hiệu quả khi mỗi lớp tạo một cụm gọn, nhưng centroid bị kéo lệch bởi outlier hoặc khi domain đích làm biến dạng cụm. Contrastive kéo mẫu dương lại gần và đẩy mẫu âm ra xa, thường tạo không gian trơn hơn; tuy nhiên hiệu quả phụ thuộc negative sampling và mức tương đồng giữa domain nguồn-đích. CrossHAR chỉ được diễn giải ở các cấu hình có dữ liệu, vì vậy không suy rộng kết quả CrossHAR sang shot mà coverage bằng N/A.

### 10.3 Backbone và protocol: kiểm chứng giả thuyết Transformer
**Nhận xét.** Ở pocket-level, standard đạt 82.16% FT và 72.08% LP tại 100-shot, trong khi cnn_transformer đạt 80.37% FT và 61.83% LP. Mức giảm FT→LP tương ứng khoảng 10.08 và 18.55 điểm phần trăm; khoảng giảm lớn hơn của cnn_transformer phù hợp với giả thuyết attention cần thích nghi domain, trong khi CNN giữ được inductive bias locality.

**Giải thích.** 1D-CNN mã hóa các mẫu cục bộ, biên độ và chu kỳ ngắn của tín hiệu nên có prior phù hợp với HAR. Transformer có receptive field rộng và có thể học phụ thuộc xa, nhưng attention pattern cần đủ dữ liệu để ổn định. Khi backbone bị đóng băng, sai lệch sensor không được sửa bởi các lớp attention, làm LP suy giảm mạnh. FT cao hơn LP ở các cấu hình matched vì FT được phép điều chỉnh cả representation; điều này không đồng nghĩa LP kém về compute, vì LP rẻ hơn và ít nguy cơ overfit hơn.

### 10.4 Data efficiency và nguy cơ overfitting
**Nhận xét.** Với Prototype/standard, FT tăng từ 65.24% ở 10-shot lên 74.42% ở 20-shot, 79.79% ở 50-shot và 82.32% ở 100-shot; LP tăng từ 56.27% lên 61.07%, 65.96% và 70.72%. Độ dốc lớn nhất nằm trong 10→20 shot, sau đó lợi ích biên giảm, cho thấy vùng bão hòa chưa hoàn toàn nhưng 50→100 shot đã thu hẹp hơn.

**Giải thích.** Thêm shot làm classifier ước lượng boundary ổn định hơn và giúp FT quan sát nhiều biến thiên người dùng. Không quan sát thấy đường cong giảm đơn điệu trong chuỗi Prototype/standard, vì vậy chưa có bằng chứng trực tiếp về overfitting theo shot. Tuy nhiên, std trên 5% ở nhiều cấu hình cnn_transformer/LP và shot thấp cho thấy variance giữa seed có thể che lấp xu hướng; cần validation theo seed và confidence interval trước khi kết luận.

### 10.5 Physical bottleneck và ý nghĩa ứng dụng
**Nhận xét.** F1 trung bình target `hhar_watch` là 33.70%, thấp hơn pocket-level 70.38% khoảng 36.68 điểm phần trăm. Cặp khó nhất trong bảng pocket-level là `uci_har → hhar_phone` với 57.99%, trong khi `motionsense → uci_har` cao nhất với 79.71%; các hướng tới hhar_phone cũng thấp hơn đáng kể so với hướng tới UCI HAR/MotionSense.

**Giải thích.** Cổ tay có biên độ và tần số dao động lớn hơn, đồng thời chịu chuyển động tay không liên quan trực tiếp tới walking/standing/sitting. Vì vậy cùng một nhãn hoạt động tạo ra tín hiệu cảm biến khác về pha, biên độ và phổ tần. Khoảng giảm 36.68 điểm không nên quy toàn bộ cho thuật toán; cần xem đây là physical domain shift và thiết kế calibration hoặc sensor-specific pretraining. Trong ứng dụng, một model đạt cao ở pocket-level không nên được triển khai trực tiếp cho wrist sensor nếu chưa có dữ liệu thích nghi.

### 10.6 Confusion matrix và cơ chế lỗi
**Nhận xét.** Gộp các ma trận cho thấy các hướng nhầm nổi bật là Walking → Upstairs, Upstairs → Walking và Downstairs → Walking. Đây là lỗi ngoài đường chéo có logic vật lý vì các hoạt động locomotion chia sẻ dao động tuần hoàn; Sitting/Standing có thể khó tách khi orientation hoặc gravity component thay đổi.

**Giải thích.** Macro F1 giảm khi một lớp bị dồn dự đoán sang lớp locomotion gần nhất, dù accuracy tổng thể vẫn có thể được nâng bởi lớp chiếm nhiều mẫu. Error analysis nên dùng confusion matrix chuẩn hóa theo hàng, bổ sung per-class recall và xem riêng từng target sensor. Từ các artifact hiện có, chưa thể khẳng định phương pháp nào giảm nhầm lẫn tốt nhất nếu không đặt các ma trận cùng protocol/shot cạnh nhau; báo cáo vì vậy không suy diễn vượt dữ liệu.

### 10.7 Thảo luận, liên hệ literature và hạn chế
**Nhận xét.** Mẫu hình FT > LP, CNN ổn định hơn Transformer trong few-shot và domain shift vật lý làm giảm mạnh F1 phù hợp với các quan sát phổ biến trong transfer learning chuỗi thời gian: representation cần vừa bất biến với hoạt động vừa thích nghi với vị trí cảm biến. Tuy vậy, không có baseline literature được chuẩn hóa cùng split, seed và nhãn nên không thực hiện so sánh số học trực tiếp với paper khác.

**Giải thích.** Hạn chế chính gồm coverage không cân bằng (CrossHAR thiếu nhiều shot thấp), chỉ số seed có thể chưa đủ cho cấu hình std cao, thiếu metadata FLOPs/parameter/latency, và confusion matrix chưa lưu nhãn class trong từng file. Các std lớn hơn 5% làm giảm độ tin cậy của xếp hạng nhỏ; cần báo cáo CI hoặc bootstrap trên seed. Ngoài ra, gộp nhiều source-target khi xếp hạng pair là mô tả tổng quan, không thay thế paired test.

### 10.8 Kết luận khoa học và đề xuất cải thiện
**Nhận xét.** Cấu hình accuracy cao nhất trong artifact hiện tại là Prototype/standard/100-shot/full_finetuning với accuracy 91.30%; nhưng best trade-off compute chưa thể xác định vì không có FLOPs, số tham số hoặc latency. Nếu chi phí là ưu tiên, Prototype/standard/100-shot/linear_probing đạt 70.72% F1 và giảm chi phí cập nhật representation, nhưng phải chấp nhận khoảng cách so với FT.

**Giải thích.** Lựa chọn triển khai cần tối ưu Pareto chứ không chỉ lấy accuracy cực đại: FT phù hợp khi có ngân sách nhãn và compute; LP phù hợp khi cần cập nhật nhanh hoặc nhiều target. Ba hướng ưu tiên là (1) pretrain domain-invariant kết hợp augmentation theo orientation và frequency, (2) calibration wrist-to-pocket hoặc adapter nhẹ thay vì cập nhật toàn bộ encoder, (3) tăng seed và dùng paired confidence interval cho các cấu hình có std cao. Hai hướng bổ sung là hard-negative mining cho Walking/Upstairs/Downstairs và benchmark latency/FLOPs để xác định trade-off định lượng.

## 11. Mã Python đề xuất cho năm hình

Các đoạn mã dưới đây dùng một DataFrame `df` với các cột `method`, `backbone`, `shot`, `protocol`, `source`, `target`, `f1`, `std`, `seed_f1`. Cần lọc `target` chứa `hhar_watch` khi vẽ overall.

### Hình 1 — Grouped bar chart tại 100-shot
Loại: grouped bar chart; X = method; Y = Macro F1 (%); hue = backbone; error bars = std; thông điệp: cấu hình method-backbone nào dẫn đầu; vị trí: Mục 3.3.
```python
import seaborn as sns
import matplotlib.pyplot as plt
plot = df[(df.shot == '100_shot') & (df.protocol == 'full_finetuning') & ~df.target.str.contains('hhar_watch')]
sns.barplot(data=plot, x='method', y='f1', hue='backbone', errorbar='sd', capsize=.1)
plt.ylabel('Macro F1 (%)'); plt.xlabel('Method'); plt.title('100-shot FT: method × backbone')
plt.tight_layout(); plt.savefig('fig1_100shot.png', dpi=300)
```

### Hình 2 — Learning curve
Loại: line chart; X = shot; Y = Macro F1 (%); mỗi đường = method-backbone-protocol; thông điệp: tốc độ học và saturation; vị trí: Mục 3.4.
```python
shots = ['10_shot', '20_shot', '50_shot', '100_shot']
plot = df[~df.target.str.contains('hhar_watch')].copy()
plot['config'] = plot.method + ' / ' + plot.backbone + ' / ' + plot.protocol
sns.lineplot(data=plot, x='shot', y='f1', hue='config', errorbar='sd', marker='o', sort=False)
plt.xticks(range(4), shots); plt.ylabel('Macro F1 (%)'); plt.xlabel('Shot')
plt.tight_layout(); plt.savefig('fig2_learning_curve.png', dpi=300)
```

### Hình 3 — Source × target heatmap
Loại: heatmap; X = target; Y = source; màu = Macro F1 (%); hue/colorbar = F1; thông điệp: bất đối xứng transfer và pair khó; vị trí: Mục 3.4.
```python
best = plot.sort_values('f1', ascending=False).iloc[0]
subset = df[(df.method == best.method) & (df.backbone == best.backbone) &
            (df.shot == best.shot) & (df.protocol == best.protocol)]
pivot = subset.pivot_table(index='source', columns='target', values='f1', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', vmin=0, vmax=100)
plt.xlabel('Target'); plt.ylabel('Source'); plt.tight_layout()
plt.savefig('fig3_transfer_heatmap.png', dpi=300)
```

### Hình 4 — Boxplot qua seed
Loại: boxplot; X = method × backbone; Y = Macro F1 (%); hue = protocol/shot; thông điệp: hộp hẹp biểu thị ổn định; vị trí: Mục 3.5 hoặc 3.7.
```python
seed = df.explode('seed_f1').copy()
seed['config'] = seed.method + ' / ' + seed.backbone
sns.boxplot(data=seed[~seed.target.str.contains('hhar_watch')],
            x='config', y='seed_f1', hue='protocol', showfliers=True)
plt.xticks(rotation=30, ha='right'); plt.ylabel('Seed Macro F1 (%)')
plt.tight_layout(); plt.savefig('fig4_seed_boxplot.png', dpi=300)
```

### Hình 5 — FT versus LP
Loại: grouped bar chart; X = method × backbone; Y = Macro F1 (%); hue = protocol; error bars = std; thông điệp: mức phụ thuộc vào adaptation; vị trí: Mục 3.4.
```python
plot = df[~df.target.str.contains('hhar_watch')].copy()
plot['config'] = plot.method + ' / ' + plot.backbone
sns.barplot(data=plot, x='config', y='f1', hue='protocol', errorbar='sd', capsize=.1)
plt.xticks(rotation=30, ha='right'); plt.ylabel('Macro F1 (%)')
plt.title('Full fine-tuning versus linear probing')
plt.tight_layout(); plt.savefig('fig5_ft_lp.png', dpi=300)
```

_Ghi chú: N/A biểu thị thiếu dữ liệu và đã bị loại khỏi mọi phép tính trung bình._
