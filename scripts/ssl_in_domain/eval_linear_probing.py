"""
MỤC TIÊU & NGUYÊN LÝ HOẠT ĐỘNG:
    - Đánh giá chất lượng biểu diễn tự thân (Feature Representation) của Encoder SSL.
    - Cơ chế (Linear Probing Protocol):
        1. Nạp trọng số SSL đã tiền huấn luyện vào Encoder Backbone.
        2. ĐÓNG BĂNG HOÀN TOÀN ENCODER (requires_grad = False).
        3. Khởi tạo một ClassifierHead mới và CHỈ HUẤN LUYỆN ĐẦU PHÂN LOẠI NÀY.
    - Quét qua toàn bộ dải tỷ lệ nhãn: [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
      (tương ứng 0.1%, 0.5%, 1%, 5%, 10%, 100% nhãn).
    - Chạy lặp qua 5 Seeds ngẫu nhiên để tính toán Mean ± Std chuẩn bài báo.
    - Tự động ghi lại toàn bộ log kết quả vào thư mục document/0_reports/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# Thiết lập đường dẫn thư mục gốc dự án
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.dataset import MotionSenseDataset
from models.har_classifier import HARClassifier
from utils.logger import SimpleLogger

# -----------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN VÀ THIẾT BỊ
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_SSL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "tstcc_encoder_pretrained_motionsense.pt")
REPORT_DIR = os.path.join(PROJECT_ROOT, "document", "0_reports", "4_ssl_linear_probing")
LOG_FILE_PATH = os.path.join(REPORT_DIR, "linear_probing_eval_logs.txt")

SEEDS = [42, 123, 456, 789, 2024]
LABEL_FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]


def extract_dataset_tensors(dataset):
    """Trích xuất toàn bộ Dataset thành 2 Tensors lớn (X, Y) để dễ dàng cắt lát theo tỷ lệ."""
    samples, labels = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        samples.append(x.numpy() if isinstance(x, torch.Tensor) else x)
        labels.append(int(y))
    return np.array(samples), np.array(labels)


def sample_stratified_subset(x_all, y_all, fraction, seed):
    """Cắt trích dữ liệu theo tỷ lệ nhãn nhưng vẫn đảm bảo cân bằng các lớp (Stratified Sampling)."""
    if fraction >= 1.0:
        return torch.tensor(x_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.long)

    # Đảm bảo mỗi lớp có ít nhất 1 mẫu huấn luyện
    min_samples = len(np.unique(y_all))
    n_samples = max(min_samples, int(len(x_all) * fraction))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(x_all, y_all))

    return torch.tensor(x_all[train_idx], dtype=torch.float32), torch.tensor(y_all[train_idx], dtype=torch.long)


def train_and_eval_single_run(x_train_sub, y_train_sub, x_val, y_val, x_test, y_test, num_classes, in_channels):
    """Huấn luyện Linear Classifier trên tập con có nhãn và đánh giá trên Test Set."""
    train_loader = DataLoader(TensorDataset(x_train_sub, y_train_sub), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    # 1. Khởi tạo mô hình và nạp trọng số Backbone SSL
    model = HARClassifier(in_channels=in_channels, num_classes=num_classes).to(DEVICE)

    if not os.path.exists(CHECKPOINT_SSL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số SSL tại: {CHECKPOINT_SSL_PATH}")

    checkpoint = torch.load(CHECKPOINT_SSL_PATH, map_location=DEVICE, weights_only=True)
    encoder_dict = checkpoint["encoder"] if "encoder" in checkpoint else checkpoint
    model.encoder.load_state_dict(encoder_dict)

    # 2. ĐÓNG BĂNG TRỌNG SỐ ENCODER (CHỈ TRAIN CLASSIFIER HEAD)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 3. Chỉ đưa các tham số của Classifier Head vào Optimizer
    optimizer = Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_full_model_state = None

    # 4. Vòng lặp huấn luyện ngắn (Linear Probe hội tụ rất nhanh)
    for epoch in range(1, 41):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # Đánh giá trên tập Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(DEVICE)
                preds = torch.argmax(model(x_b), dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_b.numpy())

        val_f1 = f1_score(val_targets, val_preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # LƯU ĐÚNG BIẾN best_full_model_state:
            best_full_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 5. Nạp lại mô hình tốt nhất (toàn bộ model gồm cả Backbone và Head đã nạp)
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_full_model_state.items()})
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(DEVICE)
            preds = torch.argmax(model(x_b), dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y_b.numpy())

    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds, average="macro") * 100
    return test_acc, test_f1, best_val_f1, best_full_model_state


def main():
    logger = SimpleLogger(LOG_FILE_PATH)
    sys.stdout = logger

    print("=" * 85)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ LINEAR PROBING TRÊN MOTIONSENSE (FROZEN BACKBONE)")
    print(f"🖥️ Thiết bị sử dụng : {DEVICE}")
    print(f"💾 File lưu logs     : {LOG_FILE_PATH}")
    print("=" * 85)

    try:
        # Nạp dữ liệu MotionSense gốc
        raw_train = MotionSenseDataset(MotionSenseConfig.RAW_DATA_DIR, MotionSenseConfig.TRAIN_SUBJECTS,
                                       MotionSenseConfig)
        raw_val = MotionSenseDataset(MotionSenseConfig.RAW_DATA_DIR, MotionSenseConfig.VAL_SUBJECTS, MotionSenseConfig)
        raw_test = MotionSenseDataset(MotionSenseConfig.RAW_DATA_DIR, MotionSenseConfig.TEST_SUBJECTS,
                                      MotionSenseConfig)

        x_train_all, y_train_all = extract_dataset_tensors(raw_train)
        x_val_all, y_val_all = extract_dataset_tensors(raw_val)
        x_test_all, y_test_all = extract_dataset_tensors(raw_test)

        x_val_t = torch.tensor(x_val_all, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_all, dtype=torch.long)
        x_test_t = torch.tensor(x_test_all, dtype=torch.float32)
        y_test_t = torch.tensor(y_test_all, dtype=torch.long)

        print(f"📦 Tổng tập Train: {len(x_train_all)} | Val: {len(x_val_all)} | Test: {len(x_test_all)}")
        print("-" * 85)

        results_summary = {}

        best_overall_100pct_val_f1 = -1.0
        best_source_checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "motionsense_linear_probed_best.pt")

        for frac in LABEL_FRACTIONS:
            f1_runs = []
            acc_runs = []
            print(
                f"\n▶️ ĐANG THỬ NGHIỆM TỶ LỆ NHÃN: {frac * 100:.1f}% (Số mẫu ước tính: {int(len(x_train_all) * frac)})")

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                x_tr_sub, y_tr_sub = sample_stratified_subset(x_train_all, y_train_all, frac, seed)
                acc, f1, val_f1, best_state = train_and_eval_single_run(
                    x_tr_sub, y_tr_sub, x_val_t, y_val_t, x_test_t, y_test_t,
                    num_classes=MotionSenseConfig.NUM_CLASSES,
                    in_channels=MotionSenseConfig.IN_CHANNELS
                )
                acc_runs.append(acc)
                f1_runs.append(f1)
                print(
                    f"   - Seed {seed:4d} | Samples: {len(x_tr_sub):4d} | Test Acc: {acc:.2f}% | Test Macro F1: {f1:.2f}%")

                # Tự động bắt và lưu mô hình 100% nhãn đạt kỷ lục Val F1 cao nhất
                if frac == 1.0 and val_f1 > best_overall_100pct_val_f1:
                    best_overall_100pct_val_f1 = val_f1
                    os.makedirs(os.path.dirname(best_source_checkpoint_path), exist_ok=True)
                    torch.save(best_state, best_source_checkpoint_path)
                    print(
                        f"     💾 [Auto-Save] Đã lưu Source Model 100% Linear Probing tốt nhất (Val F1: {val_f1 * 100:.2f}%) tại: {best_source_checkpoint_path}")
            mean_f1, std_f1 = np.mean(f1_runs), np.std(f1_runs)
            mean_acc, std_acc = np.mean(acc_runs), np.std(acc_runs)
            results_summary[frac] = (mean_f1, std_f1, mean_acc, std_acc)
            print(
                f"👉 KẾT QUẢ TỶ LỆ {frac * 100:.1f}%: Macro F1 = {mean_f1:.2f} ± {std_f1:.2f}% | Acc = {mean_acc:.2f} ± {std_acc:.2f}%")

        # In bảng tổng hợp cuối cùng
        print("\n" + "=" * 85)
        print("🏆 BẢNG TỔNG HỢP HIỆU NĂNG LINEAR PROBING (MEAN ± STD QUA 5 SEEDS)")
        print("=" * 85)
        print(f"{'Label %':<12} | {'Mẫu (Samples)':<15} | {'Macro F1 (%)':<22} | {'Accuracy (%)':<20}")
        print("-" * 85)
        for frac, (m_f1, s_f1, m_acc, s_acc) in results_summary.items():
            n_samples = int(len(x_train_all) * frac) if frac < 1.0 else len(x_train_all)
            print(
                f"{frac * 100:<10.1f}% | {n_samples:<15d} | {m_f1:6.2f} ± {s_f1:5.2f}%       | {m_acc:6.2f} ± {s_acc:5.2f}%")
        print("=" * 85)

    finally:
        logger.close()


if __name__ == "__main__":
    main()
