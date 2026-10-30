"""
===============================================================================
MODULE: DATASET REGISTRY - SINGLE SOURCE OF TRUTH CHO MỌI DATASET
===============================================================================
Mục đích:
    - Gom tất cả dataset vào một dict duy nhất.
    - Script downstream (run_few_shot.py, run_pretrain_tstcc.py, ...)
      chỉ cần `from config.datasets.registry import DATASETS` là có đủ.
    - Tránh copy-paste dict ở nhiều script -> tránh lệch cấu hình.

Cấu trúc mỗi entry:
    {
        "train": Path,   # file train.pt
        "val":   Path,   # file val.pt
        "test":  Path,   # file test.pt
        "all":   Path,   # file dataset_all.pt (dùng cho SSL pretraining)
    }
===============================================================================
"""

from pathlib import Path

from config.datasets.uci_har_config import UCIHARConfig
from config.datasets.motionsense_config import MotionSenseConfig
from config.datasets.hhar_config import HHARConfig

# Dict gom tất cả dataset. Mỗi key là tên dùng trong CLI (--datasets xxx).
DATASETS = {
    # -------------------------------------------------------------------------
    # UCI-HAR
    # -------------------------------------------------------------------------
    "uci_har": {
        "train": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
        "val": Path(UCIHARConfig.PROCESSED_VAL_PATH),
        "test": Path(UCIHARConfig.PROCESSED_TEST_PATH),
        "all": Path(UCIHARConfig.DATA_ALL_PATH),
    },

    # -------------------------------------------------------------------------
    # MotionSense
    # -------------------------------------------------------------------------
    "motion_sense": {
        "train": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
        "val": Path(MotionSenseConfig.PROCESSED_VAL_PATH),
        "test": Path(MotionSenseConfig.PROCESSED_TEST_PATH),
        "all": Path(MotionSenseConfig.DATA_ALL_PATH),
    },

    # -------------------------------------------------------------------------
    # HHAR Phone (target domain riêng)
    # -------------------------------------------------------------------------
    "hhar_phone": {
        "train": Path(HHARConfig.PROCESSED_TRAIN_PATH_PHONE),
        "val": Path(HHARConfig.PROCESSED_VAL_PATH_PHONE),
        "test": Path(HHARConfig.PROCESSED_TEST_PATH_PHONE),
        "all": Path(HHARConfig.DATA_ALL_PATH_PHONE),
    },

    # -------------------------------------------------------------------------
    # HHAR Watch (target domain riêng)
    # -------------------------------------------------------------------------
    "hhar_watch": {
        "train": Path(HHARConfig.PROCESSED_TRAIN_PATH_WATCH),
        "val": Path(HHARConfig.PROCESSED_VAL_PATH_WATCH),
        "test": Path(HHARConfig.PROCESSED_TEST_PATH_WATCH),
        "all": Path(HHARConfig.DATA_ALL_PATH_WATCH),
    },
}
