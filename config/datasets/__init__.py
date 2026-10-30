"""
===============================================================================
PACKAGE: CONFIG.DATASETS
===============================================================================
Package con chứa cấu hình của từng bộ dữ liệu HAR.

Mỗi bộ dữ liệu là một class kế thừa `BaseDatasetConfig`, chỉ khai báo
phần riêng của nó (tên lớp, label mapping, subject split, ...).
Phần chung (path, window size, số kênh) được kế thừa từ base.

File __init__.py này để trống (chỉ docstring) nhằm biến `config/datasets/`
thành một package chuẩn, tránh lỗi ModuleNotFoundError khi import.
===============================================================================
"""