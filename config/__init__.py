"""
===============================================================================
PACKAGE: CONFIG
===============================================================================
Package gốc chứa toàn bộ cấu hình của dự án CrossDomain-HAR-SSL.

Cấu trúc:
    config/
    ├── data_paths.py              <- Đường dẫn gốc (PROJECT_ROOT, DATA_ROOT, ...)
    ├── datasets/                  <- Cấu hình từng bộ dữ liệu HAR
    │   ├── base_dataset_config.py <- Lớp abstract chứa phần chung
    │   ├── uci_har_config.py      <- Cấu hình UCI-HAR
    │   ├── motionsense_config.py  <- Cấu hình MotionSense
    │   ├── hhar_config.py         <- Cấu hình HHAR (phone + watch)
    │   └── registry.py            <- Dict DATASETS gom tất cả
    └── methods/                   <- Cấu hình từng phương pháp SSL
        └── tstcc_config.py        <- Hyperparameter TS-TCC

Quy ước import:
    from config.data_paths import PROJECT_ROOT, DATA_ROOT
    from config.datasets.uci_har_config import UCIHARConfig
    from config.datasets.registry import DATASETS
    from config.methods.tstcc_config import TSTCCConfig

File __init__.py này để trống (chỉ có docstring) nhằm biến `config/`
thành một package chuẩn của Python, tránh lỗi ModuleNotFoundError
khi chạy từ thư mục khác hoặc dùng pytest.
===============================================================================
"""