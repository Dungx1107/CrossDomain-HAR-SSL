"""
===============================================================================
CẤU HÌNH CHO BỘ DỮ LIỆU UCI-HAR
===============================================================================
Mục tiêu:
    - Kế thừa BaseDatasetConfig để có sẵn path, window, channels.
    - Chỉ khai báo phần RIÊNG của UCI-HAR: classes, label mapping, subject split.
    - Đồng bộ LABEL MAP với MotionSense và HHAR để hỗ trợ Cross-Domain.

Thông tin bộ dữ liệu:
    - 30 subjects, 6 hoạt động: Walking, Upstairs, Downstairs, Sitting,
      Standing, Laying.
    - Cảm biến: Samsung Galaxy S2 gắn ở eo, tần số 50Hz.
    - Trong dự án này chỉ dùng 5 lớp đầu (bỏ Laying) cho cross-domain,
      nhưng vẫn khai báo đủ 6 lớp trong CLASS_NAMES để giữ tính tổng quát.
===============================================================================
"""

from config.datasets.base_dataset_config import BaseDatasetConfig


class UCIHARConfig(BaseDatasetConfig):
    """
    Cấu hình cho UCI-HAR.

    Kế thừa từ BaseDatasetConfig:
        - WINDOW_SIZE=128, STRIDE=64, IN_CHANNELS=6
        - Path tự động: PROCESSED_DATA_DIR = data/processed/uci_har
        - NUM_CLASSES tự động = len(CLASS_NAMES) = 6
    """

    # Tên thư mục con -> dùng để xây path tự động trong base.
    # data/raw/uci_har/  và  data/processed/uci_har/
    DATASET_NAME = "uci_har"

    # -------------------------------------------------------------------------
    # 1. ÁNH XẠ NHÃN (LABEL MAP)
    # -------------------------------------------------------------------------
    # Nhãn gốc UCI-HAR: 1=Walking, 2=Upstairs, 3=Downstairs, 4=Sitting,
    #                   5=Standing, 6=Laying.
    # Nhãn chuẩn dự án: 0=Walking, 1=Upstairs, 2=Downstairs, 3=Sitting,
    #                   4=Standing, 5=Laying.
    # -> Ánh xạ chỉ là trừ 1, nhưng viết tường minh để dễ kiểm tra.
    #
    # Lưu ý: Trong cross-domain, chỉ dùng 5 lớp đầu (0..4) để khớp với
    # MotionSense và HHAR. Lớp Laying (5) bị lọc bỏ ở downstream.
    LABEL_MAPPING = {
        1: 0,  # WALKING          -> Walking      (giống MotionSense wlk=0)
        2: 1,  # WALKING_UPSTAIRS -> Upstairs     (giống MotionSense ups=1)
        3: 2,  # WALKING_DOWNSTAIRS -> Downstairs (giống MotionSense dws=2)
        4: 3,  # SITTING          -> Sitting      (giống MotionSense sit=3)
        5: 4,  # STANDING         -> Standing     (giống MotionSense std=4)

        6: 5,  # LAYING           -> Laying       (chỉ có ở UCI-HAR)
    }

    # Tên lớp theo thứ tự index chuẩn (0..5).
    # Dùng cho hiển thị, confusion matrix, classification_report.
    CLASS_NAMES = [
        'Walking',
        'Upstairs',
        'Downstairs',
        'Sitting',
        'Standing',
        'Laying',
    ]

    # -------------------------------------------------------------------------
    # 2. CHIA DỮ LIỆU THEO SUBJECT (CHỐNG RÒ RỈ DỮ LIỆU)
    # -------------------------------------------------------------------------
    # UCI-HAR gốc đã có train/test split theo subject.
    # Trong dự án này, tách thêm 4 subject từ train gốc làm validation
    # để có thể chọn best model mà không đụng test set.
    #
    # Lưu ý: KHÔNG khai báo TRAIN_SUBJECTS / TEST_SUBJECTS vì đã có sẵn
    # trong file train.pt / test.pt do script preprocess tạo ra.
    # Chỉ khai báo VAL_SUBJECTS để script preprocess biết tách ai ra.
    VAL_SUBJECTS = [27, 28, 29, 30]

    # -------------------------------------------------------------------------
    # 3. ĐẶC TẢ TÍN HIỆU
    # -------------------------------------------------------------------------
    # 6 kênh: 3 gia tốc cơ thể (total_acc) + 3 con quay (body_gyro).
    # Tên này dùng để debug / hiển thị, không ảnh hưởng tensor shape.
    SIGNAL_NAMES = [
        "total_acc_x", "total_acc_y", "total_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
    ]