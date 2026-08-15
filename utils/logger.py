"""
===============================================================================
MODULE LOGGER - LƯU KẾT QUẢ THỰC NGHIỆM DẠNG CSV
===============================================================================
Quy tắc:
    1. Mỗi loại mô hình có 1 file CSV riêng (ví dụ: baseline_cnn.csv, resnet.csv)
    2. Nếu mô hình giống nhau -> append thêm 1 dòng mới (để so sánh tham số)
    3. Nếu mô hình khác nhau -> tự động tạo file mới
===============================================================================
"""

import os
import csv
from datetime import datetime
from typing import Dict, Any


def save_experiment_csv(model_name: str,
                        config: Dict[str, Any],
                        results: Dict[str, Any],
                        base_dir: str = "experiments"):
    """
    Lưu kết quả thực nghiệm vào file CSV (append mode)
    Mỗi model có 1 file riêng: {model_name}_results.csv

    Args:
        model_name (str): Tên mô hình (ví dụ: 'baseline_cnn', 'resnet', 'transformer')
        config (dict): Thông tin cấu hình
        results (dict): Kết quả
        base_dir (str): Thư mục lưu file (mặc định: 'experiments')
    """
    # Tạo thư mục nếu chưa có
    os.makedirs(base_dir, exist_ok=True)

    # Tên file: {model_name}_results.csv
    filename = f"{model_name}_results.csv"
    filepath = os.path.join(base_dir, filename)

    # Gộp config và results thành 1 row
    row = {
        # === THÔNG TIN LẦN CHẠY ===
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiment_id': datetime.now().strftime("%Y%m%d_%H%M%S"),

        # === CẤU HÌNH MÔ HÌNH ===
        'model_name': config.get('model_name', model_name),
        'encoder_type': config.get('encoder_type', ''),
        'in_channels': config.get('in_channels', 6),
        'feature_dim': config.get('feature_dim', 128),
        'num_classes': config.get('num_classes', 6),
        'window_size': config.get('window_size', 128),
        'kernel_size': config.get('kernel_size', 7),
        'dropout_rate': config.get('dropout_rate', 0.2),
        'num_blocks': config.get('num_blocks', 4),

        # === CẤU HÌNH HUẤN LUYỆN ===
        'epochs': config.get('epochs', 50),
        'learning_rate': config.get('learning_rate', 0.001),
        'batch_size': config.get('batch_size', 64),
        'optimizer': config.get('optimizer', 'Adam'),
        'loss_function': config.get('loss_function', 'CrossEntropyLoss'),
        'seed': config.get('seed', 42),

        # === KẾT QUẢ ===
        'best_test_accuracy': results.get('best_test_accuracy', 0.0),
        'best_test_loss': results.get('best_test_loss', 0.0),
        'best_train_accuracy': results.get('best_train_accuracy', 0.0),
        'best_epoch': results.get('best_epoch', 0),
        'total_parameters': results.get('total_parameters', 0),
        'training_time_seconds': results.get('training_time', 0.0),
        'device': results.get('device', 'cpu'),
        'model_checkpoint_path': results.get('model_checkpoint_path', '')
    }

    # Kiểm tra file đã tồn tại chưa
    file_exists = os.path.isfile(filepath)

    # Ghi vào CSV (append mode)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Ghi header nếu file chưa tồn tại
        if not file_exists:
            writer.writeheader()

        # Ghi dữ liệu
        writer.writerow(row)

    print(f"✅ Results saved to: {filepath}")
    print(f"   📊 {model_name} - Best Test Acc: {results.get('best_test_accuracy', 0):.2f}%")


def load_experiments_csv(model_name: str, base_dir: str = "experiments"):
    """
    Load tất cả kết quả của 1 model từ file CSV để phân tích

    Returns:
        pandas.DataFrame hoặc None nếu file không tồn tại
    """
    import pandas as pd

    filepath = os.path.join(base_dir, f"{model_name}_results.csv")

    if not os.path.isfile(filepath):
        print(f"⚠️ File {filepath} not found!")
        return None

    df = pd.read_csv(filepath)
    return df


def print_experiment_summary(model_name: str, base_dir: str = "experiments"):
    """
    In tóm tắt tất cả các lần chạy của 1 model
    """
    df = load_experiments_csv(model_name, base_dir)
    if df is None:
        return

    print(f"\n{'='*60}")
    print(f"📊 EXPERIMENT SUMMARY - {model_name}")
    print(f"{'='*60}")
    print(f"Total runs: {len(df)}")
    print(f"\nBest run:")
    best_idx = df['best_test_accuracy'].idxmax()
    best_row = df.loc[best_idx]
    print(f"  - Timestamp: {best_row['timestamp']}")
    print(f"  - Test Acc: {best_row['best_test_accuracy']:.2f}%")
    print(f"  - Epoch: {best_row['best_epoch']}")
    print(f"  - Learning Rate: {best_row['learning_rate']}")
    print(f"  - Dropout: {best_row['dropout_rate']}")
    print(f"\nAll runs:")
    print(df[['timestamp', 'best_test_accuracy', 'learning_rate', 'dropout_rate', 'best_epoch']].to_string(index=False))