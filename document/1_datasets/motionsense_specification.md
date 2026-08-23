# PHÂN TÍCH BỘ DỮ LIỆU MOTIONSENSE & NGHỊCH LÝ ĐẶC TRƯNG CẢM BIẾN (12 VS. 9 KÊNH)

## 1. Tổng quan về Bộ dữ liệu MotionSense (Dataset Overview)

Bộ dữ liệu **MotionSense** được thu thập nhằm nghiên cứu mối quan hệ giữa hoạt động thể chất (Human Activity Recognition - HAR) và các đặc trưng nhân khẩu học/sinh trắc học của người dùng thông qua cảm biến điện thoại thông minh.

* **Thiết bị thu thập:** Apple iPhone 6s (đặt trong túi quần trước của người tham gia).
* **Đối tượng tham gia:** 24 người (14 nam, 10 nữ) với độ tuổi, chiều cao và cân nặng đa dạng.
* **Tần số lấy mẫu (Sampling Rate):** $50\text{ Hz}$ (50 mẫu/giây).
* **API thu thập:** Framework `CoreMotion` (lớp `CMDeviceMotion`) của hệ điều hành iOS.
* **6 Lớp hành động (Activity Classes):**
  1. `Walking` (Đi bộ - `wlk`)
  2. `Jogging` (Chạy bộ - `jog`)
  3. `Upstairs` (Đi lên cầu thang - `ups`)
  4. `Downstairs` (Đi xuống cầu thang - `dws`)
  5. `Sitting` (Ngồi yên - `sit`)
  6. `Standing` (Đứng yên - `std`)

---

## 2. Cấu trúc Tín hiệu & 12 Kênh Cảm biến Ban đầu

Framework `CMDeviceMotion` của iOS tự động tổng hợp và xử lý dữ liệu thô từ Accelerometer, Gyroscope và Magnetometer để xuất ra **12 kênh tín hiệu**:

| Nhóm đặc trưng | Số kênh | Tên các trục | Đơn vị | Bản chất vật lý |
| :--- | :---: | :--- | :---: | :--- |
| **`attitude`** | 3 | `roll`, `pitch`, `yaw` | Radian | Góc quay Euler thể hiện hướng xoay của máy so với hệ quy chiếu cố định của Trái Đất. |
| **`gravity`** | 3 | `x`, `y`, `z` | G ($1\text{G} \approx 9.81\text{ m/s}^2$) | Vector gia tốc trọng trường chiếu lên 3 trục của thiết bị. |
| **`rotationRate`** | 3 | `x`, `y`, `z` | Rad/giây | Vận tốc góc tức thời quanh 3 trục của thiết bị (từ Gyroscope). |
| **`userAcceleration`** | 3 | `x`, `y`, `z` | G ($1\text{G} \approx 9.81\text{ m/s}^2$) | Gia tốc tuyến tính do chuyển động của cơ thể (sau khi đã trừ đi vector trọng lực). |

> **Nguyên lý phân tách gia tốc của iOS:**
> $$\mathbf{a}_{\text{total}}(t) = \mathbf{a}_{\text{user}}(t) + \mathbf{g}(t)$$

---

## 3. Nghịch lý Thực nghiệm: Tại sao 9 Kênh lại vượt trội hơn 12 Kênh?

Trong thực nghiệm với giao thức phân chia người độc lập (**Subject-Independent Split**: 14 Train / 4 Val / 6 Test):

* **Mô hình 12 kênh (đủ `attitude`):** Đạt Test Accuracy $\approx 90.16\%$.
* **Mô hình 9 kênh (loại bỏ `attitude`):** Đạt Test Accuracy $\approx 93.89\%$ (**tăng $+3.73\%$**).
* **Số lỗi nhầm lẫn:** Ở bản 12 kênh, có tới **365 mẫu `Walking` bị đoán nhầm thành `Downstairs`** trên tập Test, trong khi bản 9 kênh triệt tiêu gần như toàn bộ lỗi nhầm lẫn này.

---

## 4. Phân tích Nguyên nhân Cốt lõi (Root Cause Analysis)

### 4.1. Hệ quy chiếu Cố định Trái Đất vs. Góc Bỏ Túi Người Dùng
* **Bản chất của `attitude` (Roll, Pitch, Yaw):** Là góc xoay tuyệt đối so với từ trường và mặt đất.
* **Sự biến thiên thực tế:**
  * Mỗi người có thói quen đút điện thoại vào túi quần khác nhau: người đút dọc màn hình quay vào trong, người đút chéo $15^\circ$, người đút túi bên trái/phải.
  * Hơn nữa, khi đi bộ, nếu người thứ nhất đi về hướng **Bắc** ($\text{Yaw} \approx 0^\circ$) và người thứ hai đi về hướng **Đông** ($\text{Yaw} \approx 90^\circ$), giá trị của `attitude` sẽ hoàn toàn lệch nhau dù cả hai cùng thực hiện hành động `Walking`.

### 4.2. Hiện tượng Tương quan Giả (Spurious Correlation & Shortcut Learning)
* Khi nạp cả 12 kênh vào mạng nơ-ron:
  $$\hat{y} = f_{\theta}(\text{attitude}, \text{gravity}, \text{rotationRate}, \text{userAcc})$$
* Mạng nơ-ron nhận thấy các giá trị `attitude` của 14 người trong tập Train rất ổn định cho từng lượt chạy. Nó vô tình học "đường tắt" (shortcut rule): 
  $$\text{"Nếu } \text{Roll} \in [1.2, 1.5] \implies \text{Người A đang đi bộ"}$$
* Khi sang **Tập Test (6 người mới hoàn toàn)**: 6 người này có góc đút túi và hướng di chuyển khác. Mạng bị "đánh lừa" bởi góc xoay lạ lẫm, dẫn đến phán đoán sai lệch nghiêm trọng.

### 4.3. Tại sao 9 Kênh còn lại là Đặc trưng Bất biến (Domain-Invariant Features)?
Khi loại bỏ 3 trục `attitude`, mô hình bị ép buộc phải học từ các quy luật động học và cơ sinh học (Biomechanics):

1. **`gravity` ($x, y, z$):** 
   * Không phụ thuộc vào hướng la bàn (hướng Đông/Tây/Nam/Bắc).
   * Chỉ phản ánh tư thế tương đối của cơ thể: Khi đứng/đi bộ, đùi thẳng đứng $\rightarrow$ vector trọng lực chiếu dọc thân máy. Khi ngồi, đùi nằm ngang $\rightarrow$ vector trọng lực đổi sang trục vuông góc.
2. **`userAcceleration` & `rotationRate` ($6\text{ kênh}$):**
   * Đại diện cho **nhịp điệu và động lực học bước chân** (Gait Dynamics):
     * *Đi bộ bằng phẳng (`Walking`):* Dạng sóng dao động hình sin điều hòa, chu kỳ bước đối xứng.
     * *Leo/xuống cầu thang (`Upstairs`/`Downstairs`):* Lực dậm chân bất đối xứng, gia tốc nâng/hạ trọng tâm cơ thể có biên độ đặc thù.

---

## 5. Ý nghĩa đối với Bài toán Chuyển giao Miền chéo (Cross-Domain)

| Cấu hình kênh | Áp dụng trên MotionSense | Khả năng Chuyển giao sang UCI-HAR / HHAR |
| :---: | :---: | :--- |
| **12 Kênh** | Kém tổng quát giữa các người dùng (Overfitting góc xoay). | **Không thể chuyển giao:** Các tập dữ liệu khác không cung cấp góc `attitude` tương đương. |
| **9 Kênh** | Rất tốt trên tập Test MotionSense ($93.89\%$). | **Hạn chế:** Các tập như UCI-HAR chỉ có gia tốc tổng thô và con quay hồi chuyển, chưa tách riêng vector trọng lực `gravity`. |
| **6 Kênh cốt lõi** (`userAcc` + `gyro`) | Đạt hiệu năng chuẩn mực. | **Tương thích hoàn toàn (Common Sensor Space):** Là chuẩn giao tiếp chung cho mọi phần cứng IMU trên thị trường. |

---

## 6. Tài liệu Tham khảo (Academic References)

1. **Malekzadeh, M., Clegg, R. G., Cavallaro, A., & Haddadi, H. (2019).** *Mobile sensor data anonymization.* In *Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI-19)*, pp. 49-58. (Paper giới thiệu bộ dữ liệu MotionSense).
2. **Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2021).** *Time-Series Representation Learning via Temporal and Contextual Contrasting.* In *IJCAI-21*, pp. 2352-2359.
3. **Wang, J., Chen, Y., Hao, S., Peng, X., & Hu, L. (2019).** *Deep learning for sensor-based human activity recognition: A survey.* *Pattern Recognition Letters*, 119, 3-11.
4. **Geirhos, R., Jacobsen, J. H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020).** *Shortcut learning in deep neural networks.* *Nature Machine Intelligence*, 2(11), 665-673.