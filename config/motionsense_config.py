# FILE CẤU HÌNH THAM SỐ CHO BỘ DỮ LIỆU MOTIONSENSE
# 🧠 Bài giảng lý thuyết cấu hình:
#
#     1. Ánh xạ nhãn (Label Mapping): Tên thư mục bắt đầu bằng dws, ups, wlk, jog, sit, std.
#     Ta phải chuyển chúng về các số từ 0 đến 5 để máy tính làm toán loss function.
#
#     2. Chiến lược chia dữ liệu (Subject Split): Dùng 18 người đầu (sub_1 đến sub_18) làm tập Train,
#     và 6 người còn lại (sub_19 đến sub_24) làm tập Test để đánh giá độc lập (Subject-Independent).
#
#     3. Định dạng dữ liệu đầu vào: Cảm biến MotionSense có 12 kênh dữ liệu. Ở bài Baseline này,
#     ta có thể linh hoạt chọn 6 kênh cơ bản (userAcceleration + rotationRate) hoặc full 12 kênh.

import os

# -----------------------------------------------------------------------------
# TỰ ĐỘNG TÍNH ĐƯỜNG DẪN THƯ MỤC GỐC DỰ ÁN (PROJECT ROOT)
# File này nằm tại: CrossDomain-HAR-SSL/config/motionsense_config.py
# -> dirname 1 lần = thư mục 'config'
# -> dirname 2 lần = thư mục gốc 'CrossDomain-HAR-SSL'
# -----------------------------------------------------------------------------
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_DIR)


class MotionSenseConfig:
    # -------------------------------------------------------------------------
    # 1. ĐƯỜNG DẪN DỮ LIỆU (DATA PATHS)
    # -------------------------------------------------------------------------
    # Đường dẫn tới thư mục chứa dữ liệu thô MotionSense
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "motion_sense")

    # -------------------------------------------------------------------------
    # 2. ÁNH XẠ NHÃN HOẠT ĐỘNG (LABEL MAPPING)
    # -------------------------------------------------------------------------
    # Tác dụng: Chuyển mã ký tự thư mục gốc (dws, ups...) thành tên hoạt động
    # và chỉ số lớp (Class Index) dạng số nguyên từ 0 đến 5 để tính Loss trong PyTorch.
    LABEL_MAP = {
        'dws': 0,  # Downstairs (Đi xuống cầu thang)
        'ups': 1,  # Upstairs (Đi lên cầu thang)
        'wlk': 2,  # Walking (Đi bộ)
        'jog': 3,  # Jogging (Chạy bộ)
        'sit': 4,  # Sitting (Ngồi)
        'std': 5  # Standing (Đứng)
    }

    # Danh sách tên nhãn để hiển thị khi in kết quả / Confusion Matrix
    CLASS_NAMES = ['Downstairs', 'Upstairs', 'Walking', 'Jogging', 'Sitting', 'Standing']
    NUM_CLASSES = len(CLASS_NAMES)  # Tổng số lớp = 6

    # -------------------------------------------------------------------------
    # 3. PHÂN CHIA TẬP TRAIN / TEST THEO NGƯỜI DÙNG (SUBJECT-INDEPENDENT SPLIT)
    # -------------------------------------------------------------------------
    # Tác dụng: Chống Overfitting.
    # Ta dùng ID của 18 người dùng để Train, 6 người dùng hoàn toàn mới để Test.
    TRAIN_SUBJECTS = list(range(1, 19))  # ID từ 1 đến 18
    TEST_SUBJECTS = list(range(19, 25))  # ID từ 19 đến 24

    # -------------------------------------------------------------------------
    # 4. THAM SỐ CẮT CỬA SỔ TRƯỢT (SLIDING WINDOW PARAMETERS)
    # -------------------------------------------------------------------------
    # SAMPLING_RATE = 50Hz (MotionSense thu thập 50 mẫu/giây)
    # WINDOW_SIZE = 128 mẫu -> Tương đương 128 / 50 = 2.56 giây dữ liệu cho 1 cửa sổ
    WINDOW_SIZE = 128

    # OVERLAP = 0.5 (Độ chồng lấp 50%) -> Cửa sổ sau sẽ trượt lên cửa sổ trước 64 mẫu (1.28 giây)
    # Tác dụng: Tăng gấp đôi số lượng mẫu dữ liệu thu được, giữ tính liên tục của hành động.
    STRIDE = int(WINDOW_SIZE * (1 - 0.5))  # STRIDE = 64 mẫu

    # -------------------------------------------------------------------------
    # 5. LỰA CHỌN KÊNH CẢM BIẾN DỮ LIỆU (FEATURE COLUMNS)
    # -------------------------------------------------------------------------
    # Chọn 6 kênh cảm biến cơ bản nhất cho bài toán HAR:
    # 3 trục Gia tốc người dùng (userAcceleration) + 3 trục Vận tốc góc (rotationRate)
    FEATURE_COLS = [
        'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
        'rotationRate.x', 'rotationRate.y', 'rotationRate.z'
    ]
    IN_CHANNELS = len(FEATURE_COLS)  # Số kênh đầu vào = 6

    # -------------------------------------------------------------------------
    # 6. THAM SỐ HUẤN LUYỆN (TRAINING HYPERPARAMETERS)
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64  # Số lượng cửa sổ dữ liệu xử lý trong 1 lần nạp GPU
    LEARNING_RATE = 1e-3  # Tốc độ học (0.001) cho thuật toán Optimizer Adam
    EPOCHS = 30  # Số lượt duyệt qua toàn bộ tập dữ liệu Train
