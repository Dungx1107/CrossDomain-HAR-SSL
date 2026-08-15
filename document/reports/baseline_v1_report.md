# BÁO CÁO KỸ THUẬT: XÂY DỰNG MÔ HÌNH BASELINE HỌC CÓ GIÁM SÁT (VERSION 1.0)

**Đề tài:** Nhận dạng hoạt động người dùng (Human Activity Recognition - HAR) tiếp cận Học Tự Giám sát (Self-Supervised Learning - SSL) và Thích nghi miền (Cross-Domain Adaptation)

**Tác giả thực hiện:** [Tên của bạn]

**Ngày hoàn thành:** 15/08/2026

---

## 1. TỔNG QUAN VÀ MỤC TIÊU GIAI ĐOẠN 1

Giai đoạn 1 tập trung xây dựng một đường ống học sâu hoàn chỉnh (End-to-End Deep Learning Pipeline) theo phương pháp luận khoa học, đồng thời thiết lập một mô hình mốc (Baseline) sử dụng phương pháp học có giám sát (Supervised Learning) trên bộ dữ liệu MotionSense [4, 10].

Mô hình Baseline này đóng vai trò là thước đo đối sánh chuẩn cho các kịch bản Self-Supervised Learning (SSL) và Thích nghi miền (Cross-Domain Adaptation) được triển khai trong các giai đoạn tiếp theo.

---

## 2. PHƯƠNG PHÁP LUẬN VÀ TIỀN XỬ LÝ DỮ LIỆU

### 2.1. Bộ dữ liệu MotionSense

Bộ dữ liệu MotionSense được xây dựng từ dữ liệu cảm biến của điện thoại thông minh được đặt tại túi quần trước của 24 người tham gia.

Nghiên cứu sử dụng 6 hoạt động được gán nhãn:

* Downstairs: Đi xuống cầu thang.
* Upstairs: Đi lên cầu thang.
* Walking: Đi bộ.
* Jogging: Chạy bộ.
* Sitting: Ngồi.
* Standing: Đứng.

Trong mô hình Baseline, 6 kênh cảm biến được sử dụng, bao gồm:

* 3 trục gia tốc người dùng: `userAcceleration.x`, `userAcceleration.y`, `userAcceleration.z`.
* 3 trục vận tốc góc: `rotationRate.x`, `rotationRate.y`, `rotationRate.z`.

[4]

### 2.2. Chiến lược chia dữ liệu: 3-Way Subject-Independent Split

Để hạn chế hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các đối tượng và giảm thiên vị trong quá trình lựa chọn mô hình (Model Selection Bias), 24 người dùng được chia độc lập thành ba tập: Train, Validation và Test [4].

| Tập dữ liệu | Subjects            | Số cửa sổ | Số Batch | Mục đích                                             |
| ----------- | ------------------- | --------: | -------: | ---------------------------------------------------- |
| Train       | `sub_1` → `sub_14`  |    12.375 |      193 | Huấn luyện mô hình                                   |
| Validation  | `sub_15` → `sub_18` |     3.711 |       58 | Theo dõi quá trình huấn luyện và lựa chọn checkpoint |
| Test        | `sub_19` → `sub_24` |     5.453 |       86 | Đánh giá độc lập cuối cùng                           |

Tập Test được giữ độc lập trong quá trình huấn luyện và lựa chọn checkpoint. Mô hình chỉ được đánh giá trên tập Test sau khi hoàn tất quá trình lựa chọn mô hình dựa trên tập Validation [4, 6, 9, 10].

### 2.3. Kỹ thuật cắt cửa sổ trượt (Sliding Window)

Dữ liệu cảm biến liên tục được chia thành các cửa sổ thời gian bằng kỹ thuật Sliding Window với các thông số:

* `WINDOW_SIZE = 128` mẫu.
* Tần số lấy mẫu: 50 Hz.
* Độ dài mỗi cửa sổ: 128 / 50 = 2,56 giây.
* Độ chồng lấp (Overlap): 50%.
* `STRIDE = 64` mẫu.

Việc sử dụng cửa sổ chồng lấp giúp tạo ra nhiều mẫu huấn luyện hơn đồng thời duy trì tính liên tục của tín hiệu trong các đoạn thời gian kế tiếp [4].

Sau quá trình tiền xử lý, dữ liệu được đưa về định dạng Tensor chuẩn của PyTorch:

```text
(Batch_Size, Channels, Time_Steps)
```

Với Batch Size bằng 64, tensor đầu vào của mô hình có kích thước:

```text
(B, 6, 128)
```

[6, 7]

---

## 3. KIẾN TRÚC MÔ HÌNH: 1D-CNN BASELINE

Mô hình Baseline được thiết kế theo kiến trúc tách rời Encoder và Classifier Head (Decoupled Architecture). Cách thiết kế này cho phép phần Encoder được tái sử dụng trong các giai đoạn Self-Supervised Learning sau này [3, 7, 8].

### 3.1. Kiến trúc tổng thể

```text
Input
(B, 6, 128)
    |
    v
[Block 1]
Conv1D(6 -> 32, kernel=7, padding=3)
+ BatchNorm1D
+ ReLU
+ MaxPool1D(2)
    |
    v
(B, 32, 64)
    |
    v
[Block 2]
Conv1D(32 -> 64, kernel=5, padding=2)
+ BatchNorm1D
+ ReLU
+ MaxPool1D(2)
    |
    v
(B, 64, 32)
    |
    v
[Block 3]
Conv1D(64 -> 128, kernel=5, padding=2)
+ BatchNorm1D
+ ReLU
+ MaxPool1D(2)
    |
    v
(B, 128, 16)
    |
    v
[Block 4]
Conv1D(128 -> 256, kernel=3, padding=1)
+ BatchNorm1D
+ ReLU
+ MaxPool1D(2)
    |
    v
(B, 256, 8)
    |
    v
[Global Pooling]
AdaptiveAvgPool1D(1)
+ Flatten
    |
    v
(B, 256)
    |
    v
[Projection]
Linear(256 -> 128)
    |
    v
(B, 128)
Feature Vector
    |
    v
[Classifier Head]
Dropout(p=0.2)
+ Linear(128 -> 6)
    |
    v
(B, 6)
Logits
```

### 3.2. Các thành phần chính

Encoder bao gồm 4 khối tích chập 1D. Số lượng kênh được tăng dần từ 6 lên 256, trong khi chiều thời gian được giảm dần thông qua các lớp MaxPooling.

Sau Block 4, tensor có kích thước:

```text
(B, 256, 8)
```

Lớp `AdaptiveAvgPool1d(1)` thực hiện Global Average Pooling trên chiều thời gian, đưa tensor về:

```text
(B, 256, 1)
```

Sau đó `Flatten` tạo ra vector đặc trưng:

```text
(B, 256)
```

Lớp Projection `Linear(256 -> 128)` tiếp tục ánh xạ vector này thành Feature Vector có kích thước:

```text
(B, 128)
```

Feature Vector này được thiết kế để có thể được tái sử dụng trong các giai đoạn SSL và Domain Adaptation.

Classifier Head nhận Feature Vector và ánh xạ thành 6 Logits tương ứng với 6 lớp hoạt động:

```text
(B, 128) -> (B, 6)
```

[7, 8]

### 3.3. Cấu hình huấn luyện

| Thành phần                | Giá trị              |
| ------------------------- | -------------------- |
| Kiến trúc                 | 1D-CNN               |
| Input                     | `(B, 6, 128)`        |
| Feature Dimension         | 128                  |
| Số lớp phân loại          | 6                    |
| Loss Function             | `CrossEntropyLoss()` |
| Optimizer                 | `Adam`               |
| Learning Rate             | `0.001`              |
| Batch Size                | 64                   |
| Số Epoch                  | 30                   |
| Dropout                   | 0.2                  |
| Số tham số                | ~178.694             |
| Kích thước tham số xấp xỉ | ~0.68 MB             |

[4, 7, 10]

---

## 4. KẾT QUẢ THỰC NGHIỆM

### 4.1. Lựa chọn mô hình dựa trên Validation Set

Trong quá trình huấn luyện 30 Epoch, mô hình đạt độ chính xác Validation cao nhất tại:

```text
Epoch: 17
Validation Accuracy: 87.79%
```

Checkpoint tại Epoch 17 được lựa chọn làm mô hình cuối cùng để đánh giá trên tập Test độc lập [10].

Điều này đảm bảo tập Test không được sử dụng để lựa chọn Epoch hoặc điều chỉnh mô hình.

### 4.2. Classification Report trên Test Set

Khi sử dụng checkpoint tốt nhất tại Epoch 17 để đánh giá trên các Subjects 19–24, mô hình đạt:

* Overall Accuracy: **81.68%**
* Macro F1-Score: **81.68%**
* Weighted F1-Score: **82.20%**

| Activity   | Precision | Recall | F1-Score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| Downstairs |    0.4818 | 0.8169 |   0.6061 |     486 |
| Upstairs   |    0.8163 | 0.8658 |   0.8403 |     626 |
| Walking    |    0.9821 | 0.6996 |   0.8171 |   1,335 |
| Jogging    |    0.9755 | 0.9792 |   0.9773 |     528 |
| Sitting    |    0.7450 | 0.9831 |   0.8477 |   1,183 |
| Standing   |    0.9762 | 0.6958 |   0.8124 |   1,295 |

### 4.3. Nhận xét Classification Report

Kết quả cho thấy mô hình có hiệu năng không đồng đều giữa các lớp.

`Jogging` là lớp được nhận dạng tốt nhất với F1-Score đạt 0.9773. Cả Precision và Recall đều cao, lần lượt đạt 0.9755 và 0.9792.

`Downstairs` là lớp có F1-Score thấp nhất, chỉ đạt 0.6061. Mặc dù Recall đạt 0.8169, Precision chỉ đạt 0.4818. Điều này cho thấy mô hình có xu hướng dự đoán `Downstairs` cho nhiều mẫu thực tế thuộc các lớp khác.

`Walking` cũng thể hiện sự mất cân bằng tương tự: Precision rất cao (0.9821) nhưng Recall chỉ đạt 0.6996. Nói cách khác, khi mô hình dự đoán một mẫu là Walking thì phần lớn là đúng, nhưng mô hình bỏ sót một lượng đáng kể các mẫu Walking thực tế.

Đối với `Sitting`, Recall đạt rất cao (0.9831), cho thấy phần lớn các mẫu Sitting được nhận dạng chính xác. Tuy nhiên Precision chỉ đạt 0.7450, cho thấy một số mẫu thuộc lớp khác bị dự đoán nhầm thành Sitting.

---

## 5. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)

```text
Thực tế / Dự đoán    | Downstairs | Upstairs | Walking | Jogging | Sitting | Standing
---------------------------------------------------------------------------------------
Downstairs (486)     |    397     |    74    |    7    |    7    |    0    |    1
Upstairs (626)       |     68     |   542    |    9    |    1    |    6    |    0
Walking (1335)       |    353     |    42    |   934    |    5    |    0    |    1
Jogging (528)        |      5     |     5    |    1    |   517    |    0    |    0
Sitting (1183)       |      0     |     0    |    0    |    0    |  1163    |   20
Standing (1295)      |      1     |     1    |    0    |    0    |   392    |  901
```

Ma trận nhầm lẫn cho thấy một số cặp hoạt động là nguồn lỗi chính của mô hình.

Đáng chú ý nhất là sự nhầm lẫn giữa `Walking` và `Downstairs`. Trong tổng số 1.335 mẫu Walking, có 353 mẫu bị dự đoán thành Downstairs. Đồng thời, trong 486 mẫu Downstairs, có 74 mẫu bị dự đoán thành Upstairs.

Một hiện tượng đáng chú ý khác là sự nhầm lẫn giữa `Standing` và `Sitting`. Có 392 mẫu Standing bị dự đoán thành Sitting, trong khi chỉ có 20 mẫu Sitting bị dự đoán thành Standing. Điều này giải thích tại sao Recall của Sitting rất cao nhưng Precision thấp hơn đáng kể.

Ngược lại, `Jogging` gần như được phân biệt rõ ràng với các hoạt động còn lại, với 517/528 mẫu được dự đoán chính xác.

---

## 6. ĐÁNH GIÁ BASELINE

Mô hình 1D-CNN Baseline đạt **81.68% Accuracy** trên tập Test với chiến lược đánh giá Subject-Independent. Đây là một kết quả đủ để thiết lập mốc tham chiếu cho các thí nghiệm tiếp theo.

Điểm quan trọng của Baseline không chỉ nằm ở Accuracy mà còn ở việc toàn bộ pipeline đã được xây dựng theo hướng có thể tái sử dụng:

```text
Raw Sensor Data
       |
       v
Preprocessing
       |
       v
Sliding Window
       |
       v
1D-CNN Encoder
       |
       v
Feature Vector (128-D)
       |
       +----------------------+
       |                      |
       v                      v
Supervised Classifier      SSL Head
       |                      |
       v                      v
   Classification        Representation
                              Learning
```

Kiến trúc Encoder–Classifier được tách biệt giúp Feature Vector 128 chiều có thể trở thành đầu ra trung gian cho các giai đoạn nghiên cứu tiếp theo.

---

## 7. KẾT LUẬN VÀ ĐỊNH HƯỚNG GIAI ĐOẠN TIẾP THEO

Giai đoạn 1 đã hoàn thành việc xây dựng và đánh giá mô hình Baseline học có giám sát cho bài toán Human Activity Recognition trên dữ liệu cảm biến MotionSense.

Các thành phần chính đã được hoàn thiện gồm:

1. Tiền xử lý dữ liệu cảm biến 6 kênh.
2. Sliding Window với kích thước 128 mẫu và Overlap 50%.
3. Chiến lược chia dữ liệu Subject-Independent thành Train/Validation/Test.
4. Encoder 1D-CNN gồm 4 Convolutional Blocks.
5. Feature Vector 128 chiều.
6. Classifier Head cho 6 lớp hoạt động.
7. Quy trình lựa chọn checkpoint dựa trên Validation Set.
8. Đánh giá cuối cùng trên Test Set độc lập.

Kết quả cuối cùng của Baseline:

```text
Best Validation Accuracy : 87.79%
Test Accuracy             : 81.68%
Macro F1                  : 81.68%
Weighted F1               : 82.20%
```

Mô hình này sẽ được sử dụng làm mốc đối sánh cho các phương pháp Self-Supervised Learning và Cross-Domain Adaptation trong các giai đoạn tiếp theo.

Đặc biệt, Feature Vector 128 chiều từ Encoder sẽ là thành phần trung tâm để xây dựng pipeline SSL, trong đó Encoder có thể được Pre-train trên dữ liệu chưa gán nhãn trước khi được Fine-tune cho nhiệm vụ HAR.

---
