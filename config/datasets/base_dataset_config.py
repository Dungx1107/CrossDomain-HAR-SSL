"""
===============================================================================
MODULE: BASE DATASET CONFIG - LỚP ABSTRACT CHO MỌI BỘ DỮ LIỆU HAR
===============================================================================
Mục đích:
    - Gom toàn bộ phần CHUNG của các dataset vào một chỗ.
    - Mỗi dataset con chỉ cần khai báo phần RIÊNG (classes, subjects, DSP...).
    - Tự động xây dựng đường dẫn file (.pt) dựa trên DATASET_NAME.

Cơ chế hoạt động:
    1. Lớp con kế thừa BaseDatasetConfig.
    2. Lớp con khai báo DATASET_NAME (ví dụ "uci_har").
    3. Khi lớp con được định nghĩa, Python tự động gọi __init_subclass__.
    4. __init_subclass__ gọi _build_paths() để set các biến path:
           PROCESSED_DATA_DIR = data/processed/<DATASET_NAME>
           DATA_ALL_PATH      = .../dataset_all.pt
           PROCESSED_TRAIN_PATH = .../train.pt
           ...
    5. Nếu lớp con có CLASS_NAMES, tự động set NUM_CLASSES = len(CLASS_NAMES).

Tại sao dùng __init_subclass__?
    - Không cần khởi tạo instance (config dùng như class attribute).
    - Không cần gọi thủ công build_paths() sau khi định nghĩa class.
    - Tự động, ít lỗi quên.

Ví dụ sử dụng:
    class UCIHARConfig(BaseDatasetConfig):
        DATASET_NAME = "uci_har"
        CLASS_NAMES = ['Walking', ..., 'Laying']

    # Sau khi định nghĩa xong, tự động có:
    UCIHARConfig.PROCESSED_TRAIN_PATH  # = data/processed/uci_har/train.pt
    UCIHARConfig.NUM_CLASSES           # = 6
===============================================================================
"""

from abc import ABC

from config.data_paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


class BaseDatasetConfig(ABC):
    """
    Lớp abstract chứa cấu hình chung cho mọi bộ dữ liệu HAR.

    Lớp con BẮT BUỘC phải khai báo:
        - DATASET_NAME: tên thư mục con trong data/raw/ và data/processed/.
        - CLASS_NAMES : list tên lớp (theo thứ tự index 0..N-1).

    Lớp con CÓ THỂ override:
        - LABEL_MAPPING: dict ánh xạ nhãn gốc -> nhãn chuẩn (mặc định rỗng).
        - TRAIN_SUBJECTS / VAL_SUBJECTS / TEST_SUBJECTS: subject split.
        - WINDOW_SIZE / STRIDE / IN_CHANNELS: nếu khác mặc định.
    """

    # -------------------------------------------------------------------------
    # 1. THUỘC TÍNH BẮT BUỘC LỚP CON PHẢI OVERRIDE
    # -------------------------------------------------------------------------

    # Tên thư mục con (dùng để xây path và làm key trong registry).
    # Ví dụ: "uci_har", "motion_sense", "hhar".
    # Để None ở base -> lớp con nào quên khai báo sẽ bị lỗi rõ ràng.
    DATASET_NAME: str = None

    # Danh sách tên lớp theo thứ tự index (index 0 là lớp đầu tiên).
    # Ví dụ: ['Walking', 'Upstairs', ...].
    # Để None ở base -> lớp con nào quên khai báo sẽ bị lỗi rõ ràng.
    CLASS_NAMES: list = None

    # -------------------------------------------------------------------------
    # 2. THUỘC TÍNH CHUNG (CÓ THỂ OVERRIDE NẾU CẦN)
    # -------------------------------------------------------------------------

    # Kích thước cửa sổ trượt (số timestep mỗi mẫu).
    # 128 samples @ 50Hz = 2.56 giây. Dùng chung cho cả 3 dataset.
    WINDOW_SIZE: int = 128

    # Bước nhảy của cửa sổ trượt.
    # 64 = overlap 50% giữa 2 cửa sổ liên tiếp.
    STRIDE: int = 64

    # Số kênh cảm biến đầu vào (3 acc + 3 gyro = 6).
    IN_CHANNELS: int = 6

    # Dict ánh xạ nhãn gốc -> nhãn chuẩn (0..N-1).
    # Mặc định rỗng; dataset nào cần thì override.
    # Ví dụ UCI-HAR: {1: 0, 2: 1, ..., 6: 5}.
    LABEL_MAPPING: dict = {}

    # -------------------------------------------------------------------------
    # 3. CÁC BIẾN PATH (ĐƯỢC TỰ ĐỘNG SET BỞI _build_paths)
    # -------------------------------------------------------------------------
    # Khai báo ở đây với giá trị None để IDE biết các attribute này tồn tại
    # (giúp autocomplete và tránh warning "attribute not defined").
    # Giá trị thực sẽ được _build_paths() gán sau.

    RAW_DATA_DIR = None            # = data/raw/<DATASET_NAME>
    PROCESSED_DATA_DIR = None      # = data/processed/<DATASET_NAME>
    DATA_ALL_PATH = None           # = .../dataset_all.pt
    PROCESSED_TRAIN_PATH = None    # = .../train.pt
    PROCESSED_VAL_PATH = None      # = .../val.pt
    PROCESSED_TEST_PATH = None     # = .../test.pt
    NUM_CLASSES = None             # = len(CLASS_NAMES)

    # -------------------------------------------------------------------------
    # 4. HÀM XÂY DỰNG PATH
    # -------------------------------------------------------------------------

    @classmethod
    def _build_paths(cls):
        """
        Xây dựng các biến path dựa trên DATASET_NAME.

        Được gọi tự động từ __init_subclass__ ngay sau khi lớp con
        được định nghĩa (xem giải thích ở docstring đầu file).

        Lớp con nào có cấu trúc path PHỨC TẠP HƠN (ví dụ HHAR có thêm
        phone/watch) thì override hàm này, gọi super()._build_paths()
        trước rồi mới thêm path phụ.
        """
        # Đường dẫn thư mục gốc của dataset (raw + processed).
        # Ví dụ: data/raw/uci_har, data/processed/uci_har.
        cls.RAW_DATA_DIR = RAW_DATA_DIR / cls.DATASET_NAME
        cls.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / cls.DATASET_NAME

        # File gộp toàn bộ dữ liệu (dùng cho SSL pretraining).
        # SSL không cần nhãn và cần nhiều dữ liệu nhất có thể -> dùng file này.
        cls.DATA_ALL_PATH = cls.PROCESSED_DATA_DIR / "dataset_all.pt"

        # 3 file train/val/test (dùng cho downstream: LP / FT / cross-domain).
        cls.PROCESSED_TRAIN_PATH = cls.PROCESSED_DATA_DIR / "train.pt"
        cls.PROCESSED_VAL_PATH = cls.PROCESSED_DATA_DIR / "val.pt"
        cls.PROCESSED_TEST_PATH = cls.PROCESSED_DATA_DIR / "test.pt"

    # -------------------------------------------------------------------------
    # 5. HOOK TỰ ĐỘNG CHẠY KHI LỚP CON ĐƯỢC ĐỊNH NGHĨA
    # -------------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        """
        Hook đặc biệt của Python: tự động chạy mỗi khi một lớp con
        được định nghĩa (không cần khởi tạo instance).

        Cơ chế:
            - Khi Python gặp `class UCIHARConfig(BaseDatasetConfig):`,
              sau khi thực thi xong thân class, nó gọi __init_subclass__
              với cls = UCIHARConfig.
            - Ta dùng hook này để:
                1. Xây path dựa trên DATASET_NAME.
                2. Tự set NUM_CLASSES từ CLASS_NAMES.

        Lưu ý thứ tự:
            - Thân class con CHẠY XONG trước (các biến như DATASET_NAME,
              CLASS_NAMES đã có giá trị).
            - Sau đó __init_subclass__ mới chạy -> an toàn để đọc các biến đó.
        """
        super().__init_subclass__(**kwargs)

        # Chỉ xây path nếu lớp con đã khai báo DATASET_NAME.
        # (Tránh lỗi nếu ai đó tạo lớp trung gian chưa có DATASET_NAME.)
        if cls.DATASET_NAME is not None:
            cls._build_paths()

        # Tự động tính NUM_CLASSES từ CLASS_NAMES.
        # Giữ NUM_CLASSES là class attribute (không phải method) để
        # tương thích với code cũ đang dùng UCIHARConfig.NUM_CLASSES.
        if cls.CLASS_NAMES is not None:
            cls.NUM_CLASSES = len(cls.CLASS_NAMES)

    # -------------------------------------------------------------------------
    # 6. PHƯƠNG THỨC TIỆN ÍCH
    # -------------------------------------------------------------------------

    @classmethod
    def num_classes(cls) -> int:
        """
        Trả về số lớp. Có thể gọi UCIHARConfig.num_classes() thay vì
        UCIHARConfig.NUM_CLASSES nếu muốn dùng method.

        Giữ cả hai cách để tương thích ngược.
        """
        return len(cls.CLASS_NAMES)

    @classmethod
    def describe(cls) -> str:
        """
        In ra tóm tắt cấu hình dataset (tiện cho debug).
        """
        return (
            f"[{cls.__name__}]\n"
            f"  DATASET_NAME       = {cls.DATASET_NAME}\n"
            f"  NUM_CLASSES        = {cls.NUM_CLASSES}\n"
            f"  CLASS_NAMES        = {cls.CLASS_NAMES}\n"
            f"  IN_CHANNELS        = {cls.IN_CHANNELS}\n"
            f"  WINDOW_SIZE        = {cls.WINDOW_SIZE}\n"
            f"  STRIDE             = {cls.STRIDE}\n"
            f"  PROCESSED_DATA_DIR = {cls.PROCESSED_DATA_DIR}\n"
            f"  DATA_ALL_PATH      = {cls.DATA_ALL_PATH}\n"
            f"  PROCESSED_TRAIN    = {cls.PROCESSED_TRAIN_PATH}\n"
            f"  PROCESSED_VAL      = {cls.PROCESSED_VAL_PATH}\n"
            f"  PROCESSED_TEST     = {cls.PROCESSED_TEST_PATH}\n"
        )