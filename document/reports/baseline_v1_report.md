# 📄 BÁO CÁO KỸ THUẬT: XÂY DỰNG MÔ HÌNH BASELINE HỌC CÓ GIÁM SÁT (VERSION 1.0)
**Đề tài:** Nhận dạng hoạt động người dùng (Human Activity Recognition - HAR) tiếp cận Học Tự Giám sát (SSL) & Thích nghi miền (Cross-Domain)  
**Tác giả thực hiện:** [Tên của bạn]  
**Ngày hoàn thành:** 15/08/2026  

---

## 1. TỔNG QUAN VÀ MỤC TIÊU GIAI ĐOẠN 1
Giai đoạn 1 tập trung xây dựng một đường ống học sâu hoàn chỉnh (End-to-End Deep Learning Pipeline) chuẩn phương pháp luận khoa học, thiết lập mốc so sánh (Baseline) học có giám sát (Supervised Learning) trên bộ dữ liệu **MotionSense**[cite: 4, 10].

Mô hình baseline này đóng vai trò làm thước đo đối sánh chuẩn mực cho các kịch bản Self-Supervised Learning (SSL) và Thích nghi miền (Cross-Domain Adaptation) trong các giai đoạn tiếp theo.

---

## 2. PHƯƠNG PHÁP LUẬN VÀ TIỀN XỬ LÝ DỮ LIỆU

### 2.1. Bộ dữ liệu MotionSense
* Thu thập dữ liệu từ cảm biến điện thoại thông minh (Gia tốc kế & Con quay hồi chuyển) đặt tại túi quần trước của 24 người tham gia.
* **6 hoạt động được gán nhãn:** Đi xuống cầu thang (*Downstairs*), Đi lên cầu thang (*Upstairs*), Đi bộ (*Walking*), Chạy bộ (*Jogging*), Ngồi (*Sitting*), Đứng (*Standing*).
* **6 kênh cảm biến sử dụng:** 3 trục gia tốc người dùng (`userAcceleration.x/y/z`) và 3 trục vận tốc góc (`rotationRate.x/y/z`)[cite: 4].

### 2.2. Chiến lược chia dữ liệu (3-Way Subject-Independent Split)
Để loại bỏ triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) và thiên vị lựa chọn mô hình (Model Selection Bias), dữ liệu 24 người dùng được chia độc lập thành 3 tập riêng biệt[cite: 4]:
* **Tập Huấn luyện (Train Set):** 14 người (`sub_1` $\rightarrow$ `sub_14`) $\rightarrow$ Thu được **12.375 cửa sổ** (193 Batches)[cite: 4, 6].
* **Tập Kiểm định (Validation Set):** 4 người (`sub_15` $\rightarrow$ `sub_18`) $\rightarrow$ Thu được **3.711 cửa sổ** (58 Batches), dùng theo dõi Loss và lưu Checkpoint tốt nhất[cite: 4, 6, 10].
* **Tập Đánh giá Độc lập (Test Set):** 6 người (`sub_19` $\rightarrow$ `sub_24`) $\rightarrow$ Thu được **5.453 cửa sổ** (86 Batches), giữ kín hoàn toàn và chỉ chấm điểm mô hình sau cùng[cite: 4, 6, 9].

### 2.3. Kỹ thuật Cắt cửa sổ trượt (Sliding Window)
* **Kích thước cửa sổ (`WINDOW_SIZE`):** 128 mẫu (tương đương 2.56 giây ở tần số lấy mẫu 50Hz)[cite: 4].
* **Độ chồng lấp (`Overlap`):** 50% (`STRIDE = 64` mẫu) nhằm tăng cường dữ liệu và giữ tính liên tục của hành vi[cite: 4].
* **Tensor đầu vào:** Đưa về chuẩn PyTorch `(Batch_Size, Channels, Time_Steps)` = `(64, 6, 128)`[cite: 6, 7].

---

## 3. KIẾN TRÚC MÔ HÌNH (1D-CNN BASELINE)

Mô hình được thiết kế theo cơ chế **Decoupled Architecture** (Tách rời Encoder và Classifier Head) để dễ dàng tái sử dụng Encoder cho Self-Supervised Learning sau này[cite: 3, 7, 8]:

```text
Input Tensor (B, 6, 128)
   │
   ▼
[Block 1]: Conv1D(6->32, k=7, p=3)   + BatchNorm + ReLU + MaxPool1d(2) ──> (B, 32, 64)
   │
   ▼
[Block 2]: Conv1D(32->64, k=5, p=2)  + BatchNorm + ReLU + MaxPool1d(2) ──> (B, 64, 32)
   │
   ▼
[Block 3]: Conv1D(64->128, k=5, p=2) + BatchNorm + ReLU + MaxPool1d(2) ──> (B, 128, 16)
   │
   ▼
[Block 4]: Conv1D(128->256, k=3, p=1)+ BatchNorm + ReLU + MaxPool1d(2) ──> (B, 256, 8)
   │
   ▼
[Pooling]: AdaptiveAvgPool1d(1) + Flatten                              ──> (B, 256)
   │
   ▼
[Projection]: Linear(256 -> 128)                                       ──> (B, 128) [Feature Vector]
   │
   ▼
[Classifier Head]: Dropout(p=0.2) + Linear(128 -> 6)                   ──> (B, 6)   [Logits Output]
```[cite: 7, 8]

* **Tổng số tham số (Parameters):** ~178.694 tham số (~0.68 MB)[cite: 7, 10].
* **Hàm mất mát (Loss):** `CrossEntropyLoss()`[cite: 10].
* **Thuật toán tối ưu:** `Adam(lr=0.001)`[cite: 4, 10].
* **Số lượt học:** 30 Epochs[cite: 4, 10].

---

## 4. KẾT QUẢ THỰC NGHIỆM CHI TIẾT

Mô hình đạt độ chính xác trên tập Validation cao nhất tại **Epoch 17 (Val Acc: 87.79%)** và được lưu checkpoint[cite: 10]. Khi tải checkpoint này đánh giá độc lập trên tập Test (Subjects 19–24), kết quả thu được như sau[cite: 9]:

### 4.1. Bảng số liệu tổng hợp (Classification Report)
* **Overall Accuracy:** **81.68%**
* **Macro F1-Score:** **81.68%**
* **Weighted F1-Score:** **82.20%**

| Hoạt động (Class) | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Downstairs** (Đi xuống) | 0.4818 | 0.8169 | 0.6061 | 486 |
| **Upstairs** (Đi lên) | 0.8163 | 0.8658 | 0.8403 | 626 |
| **Walking** (Đi bộ) | 0.9821 | 0.6996 | 0.8171 | 1.335 |
| **Jogging** (Chạy bộ) | 0.9755 | 0.9792 | **0.9773** | 528 |
| **Sitting** (Ngồi) | 0.7450 | 0.9831 | 0.8477 | 1.183 |
| **Standing** (Đứng) | 0.9762 | 0.6958 | 0.8124 | 1.295 |

### 4.2. Ma trận nhầm lẫn (Confusion Matrix)
```text
Thực tế / Dự đoán    | Downstairs | Upstairs | Walking | Jogging | Sitting | Standing
-----------------------------------------------------------------------------------
Downstairs (486)     |    397     |    74    |    7    |    7    |    0    |    1
Upstairs (626)       |     68     |   542    |    9    |    1    |    6    |    0
Walking (1335)       |    353     |    42    |   934   |    5    |    0    |    1
Jogging (528)        |      5     |     5    |    1    |   517   |    0    |    0
Sitting (1183)       |      0     |     0    |    0    |    0    |  1163   |   20
Standing (1295)      |      1     |     1    |    0    |    0    |   392   |  901