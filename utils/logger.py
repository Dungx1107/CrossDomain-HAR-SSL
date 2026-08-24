"""
===============================================================================
    Ghi log vào đúng cấu trúc thư mục gốc của Project:
      - metrics.csv, config.json, console_output.txt -> experiments/logs/
      - Đường dẫn tuyệt đối chuẩn xác PROJECT_ROOT/experiments
===============================================================================
"""

import os
import sys
import csv
import json
from datetime import datetime

# Xác định chính xác thư mục gốc CrossDomain-HAR-SSL
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))


class ExperimentTracker:
    def __init__(self, exp_name="baseline_cnn1d", config_obj=None, notes=""):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{exp_name}_{self.timestamp}"

        # Đường dẫn tuyệt đối tới experiments/logs và experiments/plots
        self.log_dir = os.path.join(PROJECT_ROOT, "experiments", "logs", self.run_id)
        self.plot_dir = os.path.join(PROJECT_ROOT, "experiments", "plots", self.run_id)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # 1. Ghi log Console ra console_output.txt
        self.txt_path = os.path.join(self.log_dir, "console_output.txt")
        self.terminal = sys.stdout
        self.txt_file = open(self.txt_path, "a", encoding="utf-8")
        sys.stdout = self

        # 2. Lưu config.json vào logs/
        if config_obj is not None:
            self._save_config(config_obj, notes)

        # 3. Khởi tạo metrics.csv vào logs/
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.csv_fields = [
            "epoch", "train_loss", "val_loss",
            "train_acc", "val_acc", "train_macro_f1", "val_macro_f1", "lr"
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writeheader()

        print(f"🚀 KHỞI TẠO THÍ NGHIỆM : {self.run_id}")
        print(f"📁 Thư mục Logs        : {self.log_dir}")
        print(f"🖼️ Thư mục Plots       : {self.plot_dir}")

    def _save_config(self, config_obj, notes):
        config_dict = {
            "run_id": self.run_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": notes
        }
        for key in dir(config_obj):
            if key.isupper():
                config_dict[key] = getattr(config_obj, key)

        json_path = os.path.join(self.log_dir, "config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

    def log_epoch(self, metrics: dict):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow(metrics)

    def write(self, message):
        self.terminal.write(message)
        self.txt_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.txt_file.flush()

    def close(self):
        print(f"\n✅ Hoàn tất thí nghiệm.")
        print(f"   - Logs & Data: {self.log_dir}")
        print(f"   - Plots: {self.plot_dir}")
        if hasattr(self, 'txt_file') and not self.txt_file.closed:
            self.txt_file.close()
        sys.stdout = self.terminal

    def log_complexity(self, complexity_dict: dict):
        """Cập nhật thông tin độ phức tạp vào file config.json hiện tại."""
        json_path = os.path.join(self.log_dir, "config.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                current_config = json.load(f)
        else:
            current_config = {}

        current_config["model_complexity"] = complexity_dict

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(current_config, f, indent=4)


# Class chuyển hướng ghi log song song ra Terminal và 1 File TXT duy nhất
class SimpleLogger:
    def __init__(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal