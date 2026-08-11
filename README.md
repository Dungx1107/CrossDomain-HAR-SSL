# Cross-Domain HAR: Self-Supervised Learning and Enhanced Finetuning Approaches

Dự án nghiên cứu, xây dựng và đánh giá các phương pháp **Học Tự Giám Sát (Self-Supervised Learning - SSL)** kết hợp **Tinh chỉnh nâng cao (Enhanced Finetuning)** sử dụng Học Sâu cho bài toán **Nhận dạng hoạt động con người liên miền (Cross-Domain Human Activity Recognition - Cross-Domain HAR)**.

---

## 📌 Tổng Quan Đề Tài

* **Đề tài:** Nghiên cứu, xây dựng và đánh giá các phương pháp học tự giám sát sử dụng học sâu cho bài toán liên miền nhận dạng hoạt động con người (Cross-domain HAR).
* **Bài toán:** Nhận diện hoạt động con người dựa trên dữ liệu chuỗi thời gian (Time-Series) từ các cảm biến gia tốc (Accelerometer) và con quay hồi chuyển (Gyroscope).
* **Thách thức:** Sự lệch phân bố dữ liệu giữa các miền (Cross-Domain Shift) do sự khác biệt về người dùng, vị trí đeo thiết bị, loại thiết bị hoặc tần số lấy mẫu; đồng thời dữ liệu nhãn ở miền đích (Target Domain) bị hạn chế.
* **Giải pháp trọng tâm:**
  1. **Self-Supervised Learning (SSL):** Học biểu diễn đặc trưng tổng quát (Representation Learning) trên dữ liệu chuỗi thời gian không gắn nhãn.
  2. **Enhanced Finetuning:** Áp dụng các kĩ thuật tinh chỉnh nâng cao giúp mô hình thích ứng tốt và tổng quát hóa vượt trội trên các miền đích mới.

---

## 🛠️ Công Nghệ & Thư Viện

* **Ngôn ngữ:** Python 3.11+
* **Framework AI:** PyTorch, Torchvision
* **Xử lý & Phân tích Dữ liệu:** NumPy, Pandas, SciPy, Scikit-Learn
* **Trực quan hóa:** Matplotlib, Seaborn
* **Quản lý Môi trường & Package:** `uv` (Fast Python Package Installer)
* **Phần cứng:** NVIDIA GeForce RTX 2050 GPU (CUDA Enabled)

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ

### 1. Khai báo Môi trường với `uv`

Tạo môi trường ảo và cài đặt dependencies sử dụng cache cục bộ:

```bash
# Tạo virtualenv
uv venv

# Kích hoạt môi trường (Linux/Ubuntu)
source .venv/bin/activate

# Cài đặt bộ thư viện
uv pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn scipy tqdm