"""
===============================================================================
CẤU HÌNH CHO BỘ DỮ LIỆU HHAR (CẬP NHẬT CHUẨN ĐÓNG GÓI)
===============================================================================
Mục tiêu:
    - Kế thừa BaseDatasetConfig để có sẵn path gốc, window, channels.
    - Khai báo thêm path phụ cho phone và watch (HHAR có 2 loại thiết bị).
    - Khai báo tham số DSP riêng của HHAR (tần số, lọc thông thấp, ...).

Điểm đặc biệt so với UCI-HAR và MotionSense:
    - HHAR được tách thành 2 domain: hhar_phone và hhar_watch.
      Mỗi domain có file train/val/test riêng.
    - Có thêm file "combined" (gộp phone + watch) nếu cần.
    - Dữ liệu thô có timestamp không đều -> cần resample về 50Hz
      và lọc thông thấp 10Hz (xem các tham số DSP bên dưới).

Thông tin bộ dữ liệu:
    - 9 subjects (a..i), 5 hoạt động: walk, stairsup, stairsdown, sit, stand.
    - 2 loại thiết bị: phone (Samsung Galaxy S3) và watch (LG Watch).
===============================================================================
"""

from config.datasets.base_dataset_config import BaseDatasetConfig


class HHARConfig(BaseDatasetConfig):
    """
    Cấu hình cho HHAR.

    Kế thừa từ BaseDatasetConfig:
        - WINDOW_SIZE=128, STRIDE=64, IN_CHANNELS=6
        - Path gốc: PROCESSED_DATA_DIR = data/processed/hhar
        - NUM_CLASSES tự động = len(CLASS_NAMES) = 5

    Override _build_paths để thêm path phone/watch/combined.
    """

    # Tên thư mục con -> dùng để xây path gốc trong base.
    # data/raw/hhar/  và  data/processed/hhar/
    DATASET_NAME = "hhar"

    # -------------------------------------------------------------------------
    # 1. TÊN LỚP VÀ LABEL MAP
    # -------------------------------------------------------------------------
    # HHAR gốc có 5 lớp (không có Laying/Jogging).
    # Nhãn gốc là string -> cần mapping sang int 0..4.
    LABEL_MAPPING = {
        'walk': 0,  # Walking
        'stairsup': 1,  # Upstairs
        'stairsdown': 2,  # Downstairs
        'sit': 3,  # Sitting
        'stand': 4,  # Standing
    }

    CLASS_NAMES = [
        'Walking',
        'Upstairs',
        'Downstairs',
        'Sitting',
        'Standing',
    ]

    # -------------------------------------------------------------------------
    # 2. CHIA DỮ LIỆU THEO SUBJECT (CHỐNG RÒ RỈ DỮ LIỆU)
    # -------------------------------------------------------------------------
    # HHAR có 9 subjects được mã hóa 'a'..'i', ta map thành 0..8.
    # Chia 5/2/2:
    #   - Train: a, b, d, e, h  -> [0, 1, 3, 4, 7]
    #   - Val  : f, g           -> [5, 6]
    #   - Test : c, i           -> [2, 8]
    TRAIN_SUBJECTS = [0, 1, 3, 4, 7]
    VAL_SUBJECTS = [5, 6]
    TEST_SUBJECTS = [2, 8]

    # -------------------------------------------------------------------------
    # 3. THAM SỐ DSP (XỬ LÝ TÍN HIỆU SỐ)
    # -------------------------------------------------------------------------
    # HHAR gốc có timestamp không đều -> cần resample và lọc.
    # Các tham số này được dùng trong script preprocess_hhar.py.
    #
    # TARGET_FS: tần số lấy mẫu đích sau resample (50Hz, giống UCI/MotionSense).
    # GAP_THRESHOLD_S: nếu khoảng cách giữa 2 sample > 1.0s thì cắt đoạn
    #                  (tránh nối các đoạn rời rạc).
    # LOWPASS_CUTOFF: tần số cắt của bộ lọc thông thấp Butterworth (10Hz).
    # FILTER_ORDER: bậc của bộ lọc Butterworth.
    # MIN_SAMPLES: đoạn tối thiểu 256 mẫu (~5.12s) để tạo được ít nhất
    #              3 cửa sổ 128 với stride 64.
    TARGET_FS = 50.0
    GAP_THRESHOLD_S = 1.0
    LOWPASS_CUTOFF = 10.0
    FILTER_ORDER = 4
    MIN_SAMPLES = 256

    # -------------------------------------------------------------------------
    # 4. OVERRIDE _build_paths ĐỂ THÊM PATH PHONE/WATCH/COMBINED
    # -------------------------------------------------------------------------
    @classmethod
    def _build_paths(cls):
        """
        Mở rộng _build_paths của base để thêm path riêng của HHAR.

        Thứ tự:
            1. Gọi super()._build_paths() -> có PROCESSED_DATA_DIR
               = data/processed/hhar, cùng DATA_ALL_PATH, TRAIN/VAL/TEST_PATH.
            2. Từ PROCESSED_DATA_DIR, xây thêm 3 thư mục con:
                   hhar_combined/, hhar_phone/, hhar_watch/
            3. Từ mỗi thư mục con, xây path train/val/test tương ứng.
        """
        # Bước 1: base xây path gốc.
        super()._build_paths()

        # Bước 2: thêm 3 thư mục con.
        cls.PROCESSED_DIR_PHONE = cls.PROCESSED_DATA_DIR / "hhar_phone"
        cls.PROCESSED_DIR_WATCH = cls.PROCESSED_DATA_DIR / "hhar_watch"

        # Bước 3: path train/val/test cho phone.
        cls.DATA_ALL_PATH_PHONE = cls.PROCESSED_DIR_PHONE / "dataset_all.pt"
        cls.PROCESSED_TRAIN_PATH_PHONE = cls.PROCESSED_DIR_PHONE / "train.pt"
        cls.PROCESSED_VAL_PATH_PHONE = cls.PROCESSED_DIR_PHONE / "val.pt"
        cls.PROCESSED_TEST_PATH_PHONE = cls.PROCESSED_DIR_PHONE / "test.pt"

        # Bước 3: path train/val/test cho watch.
        cls.DATA_ALL_PATH_WATCH = cls.PROCESSED_DIR_WATCH / "dataset_all.pt"
        cls.PROCESSED_TRAIN_PATH_WATCH = cls.PROCESSED_DIR_WATCH / "train.pt"
        cls.PROCESSED_VAL_PATH_WATCH = cls.PROCESSED_DIR_WATCH / "val.pt"
        cls.PROCESSED_TEST_PATH_WATCH = cls.PROCESSED_DIR_WATCH / "test.pt"
