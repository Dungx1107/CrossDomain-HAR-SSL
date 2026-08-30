"""
MỤC TIÊU & NGUYÊN LÝ HOẠT ĐỘNG:
    - Tinh chỉnh toàn bộ mạng (Full Fine-Tuning) trên dữ liệu có nhãn.
    - Cơ chế:
        1. Nạp trọng số SSL vào Encoder Backbone.
        2. MỞ KHÓA TOÀN BỘ MẠNG (requires_grad = True).
        3. Áp dụng cơ chế Learning Rate phân tầng (Layer-wise LR):
           - Encoder: Học với tốc độ chậm (lr = 1e-4) để giữ lại tri thức biểu diễn sóng.
           - ClassifierHead: Học với tốc độ nhanh hơn (lr = 1e-3) để thích nghi nhanh với nhãn.
    - Hỗ trợ chạy toàn bộ dải tỷ lệ nhãn từ 0.1% -> 100% qua 5 Seeds.
    - Lưu toàn bộ kết quả vào document/0_reports/5_ssl_fine_tuning/
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
REPORT_DIR = os.path.join(PROJECT_ROOT, "document", "0_reports", "5_ssl_fine_tuning")
LOG_FILE_PATH = os.path.join(REPORT_DIR, "fine_tuning_eval_logs.txt")

SEEDS = [42, 123, 456, 789, 2024]
LABEL_FRACTIONS = [0.001, 0.005, 0.01, 0.05, 0.1, 1.0]


def extract_dataset_tensors(dataset):
    """Trích xuất dữ liệu thành Tensors phẳng."""
    samples, labels = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        samples.append(x.numpy() if isinstance(x, torch.Tensor) else x)
        labels.append(int(y))
    return np.array(samples), np.array(labels)


def sample_stratified_subset(x_all, y_all, fraction, seed):
    """Trích xuất tập con cân bằng các lớp (Stratified Sampling)."""
    if fraction >= 1.0:
        return torch.tensor(x_all, dtype=torch.float32), torch.tensor(y_all, dtype=torch.long)

    min_samples = len(np.unique(y_all))
    n_samples = max(min_samples, int(len(x_all) * fraction))

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    train_idx, _ = next(sss.split(x_all, y_all))

    return torch.tensor(x_all[train_idx], dtype=torch.float32), torch.tensor(y_all[train_idx], dtype=torch.long)


def train_and_eval_single_run(x_train_sub, y_train_sub, x_val, y_val, x_test, y_test, num_classes, in_channels):
    """Thực hiện 1 lượt huấn luyện Full Fine-Tuning với Layer-wise Learning Rate."""
    batch_size = 32 if len(x_train_sub) < 100 else 64
    train_loader = DataLoader(TensorDataset(x_train_sub, y_train_sub), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    # 1. Khởi tạo mô hình và nạp trọng số SSL vào Backbone
    model = HARClassifier(in_channels=in_channels, num_classes=num_classes).to(DEVICE)

    if not os.path.exists(CHECKPOINT_SSL_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy file trọng số SSL tại: {CHECKPOINT_SSL_PATH}")

    checkpoint = torch.load(CHECKPOINT_SSL_PATH, map_location=DEVICE, weights_only=True)
    encoder_dict = checkpoint["encoder"] if "encoder" in checkpoint else checkpoint
    model.encoder.load_state_dict(encoder_dict)

    # 2. MỞ KHÓA TOÀN BỘ CÁC TẦNG ĐỂ FINE-TUNE
    for param in model.parameters():
        param.requires_grad = True

    # 3. TỐI ƯU HÓA PHÂN TẦNG (LAYER-WISE LEARNING RATE)
    optimizer = Adam([
        {'params': model.encoder.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},  # Backbone học chậm
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4}  # Head học nhanh hơn
    ])
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_model_state = None

    # 4. Vòng lặp huấn luyện Fine-Tuning
    epochs = 50 if len(x_train_sub) > 200 else 70
    for epoch in range(1, epochs + 1):
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
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 5. Nạp lại trọng số tốt nhất và đánh giá trên tập Test
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
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
    return test_acc, test_f1, best_val_f1, best_model_state


def main():
    logger = SimpleLogger(LOG_FILE_PATH)
    sys.stdout = logger

    print("=" * 85)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ FULL FINE-TUNING TRÊN MOTIONSENSE (LAYER-WISE LR)")
    print(f"🖥️ Thiết bị sử dụng : {DEVICE}")
    print(f"💾 File lưu logs     : {LOG_FILE_PATH}")
    print("=" * 85)

    try:
        # Nạp dữ liệu MotionSense
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

        best_overall_100pct_f1 = -1.0
        best_source_model_path = os.path.join(PROJECT_ROOT, "checkpoints", "motionsense_full_finetuned_best.pt")

        for frac in LABEL_FRACTIONS:
            f1_runs = []
            acc_runs = []
            print(f"\n▶️ ĐANG THỬ NGHIỆM FINE-TUNE TỶ LỆ: {frac * 100:.1f}%")

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

            if frac == 1.0 and val_f1 > best_overall_100pct_f1:
                best_overall_100pct_f1 = val_f1
                os.makedirs(os.path.dirname(best_source_model_path), exist_ok=True)
                torch.save(best_state, best_source_model_path)
                print(
                    f"     💾 [Auto-Save] Đã lưu Source Model 100% tốt nhất (Val F1: {val_f1 * 100:.2f}%) tại: {best_source_model_path}")
            mean_f1, std_f1 = np.mean(f1_runs), np.std(f1_runs)
            mean_acc, std_acc = np.mean(acc_runs), np.std(acc_runs)
            results_summary[frac] = (mean_f1, std_f1, mean_acc, std_acc)
            print(
                f"👉 KẾT QUẢ TỶ LỆ {frac * 100:.1f}%: Macro F1 = {mean_f1:.2f} ± {std_f1:.2f}% | Acc = {mean_acc:.2f} ± {std_acc:.2f}%")

        # In bảng tổng hợp
        print("\n" + "=" * 85)
        print("🏆 BẢNG TỔNG HỢP HIỆU NĂNG FULL FINE-TUNING (MEAN ± STD QUA 5 SEEDS)")
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
