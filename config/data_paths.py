"""
===============================================================================
MODULE: DATA PATHS - ĐƯỜNG DẪN GỐC CỦA DỰ ÁN
===============================================================================
Mục đích:
    - Xác định PROJECT_ROOT (thư mục gốc dự án) một cách tự động.
    - Phát hiện môi trường chạy (local vs Kaggle) để chọn DATA_ROOT phù hợp.
    - Cung cấp RAW_DATA_DIR và PROCESSED_DATA_DIR làm gốc cho mọi config dataset.

Tại sao cần file này?
    - Tránh hardcode đường dẫn tuyệt đối trong từng script.
    - Khi chuyển từ local sang Kaggle, chỉ cần đổi DATA_ROOT ở đây.
    - Mọi dataset config đều import từ file này -> single source of truth.
===============================================================================
"""

import os
from pathlib import Path

# =============================================================================
# 1. XÁC ĐỊNH GỐC DỰ ÁN & MÔI TRƯỜNG CHẠY
# =============================================================================

# __file__: Biến đặc biệt của Python, trả về đường dẫn đến file .py hiện tại,
#           bao gồm cả tên file và phần mở rộng .py.
#           Ví dụ: /home/dungx/CrossDomain-HAR-SSL/config/data_paths.py
#
# resolve(): Chuyển đường dẫn thành đường dẫn tuyệt đối và chuẩn hóa
#            (xử lý các ký hiệu ../, ./ và symlink).
#
# parent: Lấy thư mục cha trực tiếp của đường dẫn hiện tại (config/).
# parent.parent: Lùi lên thêm một cấp nữa, đến thư mục gốc dự án
#                (CrossDomain-HAR-SSL/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Kiểm tra biến môi trường KAGGLE_KERNEL_RUN_TYPE.
# - Trên Kaggle, biến này luôn tồn tại (do Kaggle tự set).
# - Trên local, biến này không tồn tại -> IS_KAGGLE = False.
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    # Trên Kaggle: dữ liệu nằm trong /kaggle/input/... (read-only).
    # Đường dẫn này cố định theo dataset bạn upload lên Kaggle.
    DATA_ROOT = Path("/kaggle/input/datasets/nguyendung009/har-data")
else:
    # Trên local: dữ liệu nằm trong thư mục data/ của dự án.
    DATA_ROOT = PROJECT_ROOT / "data"

# =============================================================================
# 2. ĐƯỜNG DẪN CON CHO RAW VÀ PROCESSED
# =============================================================================

# Dữ liệu thô (CSV/txt gốc tải về) đặt trong data/raw/.
RAW_DATA_DIR = DATA_ROOT / "raw"

# Dữ liệu đã tiền xử lý (file .pt sau khi chạy scripts/00_preprocess/) đặt
# trong data/processed/.
PROCESSED_DATA_DIR = DATA_ROOT / "processed"