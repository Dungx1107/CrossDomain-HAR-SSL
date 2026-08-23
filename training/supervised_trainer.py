"""
===============================================================================
MỤC ĐÍCH:
    Cung cấp Engine điều phối toàn bộ quá trình huấn luyện và đánh giá mô hình
    có giám sát (Supervised Learning), phục vụ cho:
      1. Huấn luyện Baseline 1D-CNN trên 100% nhãn.
      2. Huấn luyện đối chứng ít nhãn (Few-Label Supervised: 1%, 5%, 10%).
      3. Fine-tuning Encoder sau khi hoàn tất Self-Supervised Pre-training.

CÁC CHỨC NĂNG CHÍNH:
    - Quản lý vòng lặp huấn luyện từng Epoch (Forward pass, Backward pass, Optimizer step).
    - Đánh giá trên tập Validation không tính Gradient (torch.no_grad).
    - Tính toán độ chính xác (Accuracy) và Macro F1-Score (Primary Metric).
    - Tự động điều chỉnh tốc độ học (Learning Rate Scheduler).
    - Cơ chế Model Checkpointing: Tự động lưu trọng số tốt nhất dựa trên Val Macro F1.
    - Tích hợp trực tiếp với ExperimentTracker để ghi log CSV/TXT và tự động vẽ biểu đồ.

NGUỒN THAM KHẢO HỌC THUẬT:
    - PyTorch Training Loop Best Practices: https://pytorch.org/tutorials/
    - Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning
      Library", NeurIPS 2019.
    - Grandini et al., "Metrics for Multi-Class Classification: an Overview", arXiv:2008.05756.
===============================================================================
"""

import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from utils.visualization import plot_learning_curves


class SupervisedTrainer:
    """
    Engine điều phối huấn luyện và kiểm thử mô hình có giám sát.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            tracker,
            checkpoint_path: str,
            scheduler=None,
    ):
        """
        Khởi tạo Supervised Trainer Engine.

        Args:
            model (nn.Module): Kiến trúc mạng nơ-ron (ví dụ: Supervised1DCNN).
            optimizer (Optimizer): Thuật toán tối ưu hóa Gradient (Adam, AdamW, SGD).
            criterion (nn.Module): Hàm mất mát (thường là nn.CrossEntropyLoss).
            device (torch.device): Thiết bị phần cứng tính toán (CPU hoặc CUDA GPU).
            tracker (ExperimentTracker): Đối tượng quản lý ghi nhận log và thông số.
            checkpoint_path (str): Đường dẫn file .pt để lưu trọng số mô hình tốt nhất.
            scheduler (optional): Bộ điều chỉnh Learning Rate theo tiến độ (ReduceLROnPlateau).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tracker = tracker
        self.checkpoint_path = checkpoint_path
        self.scheduler = scheduler
        self.best_val_f1 = 0.0  # Biến theo dõi kỷ lục Validation Macro F1 để lưu Checkpoint tốt nhất

    def train_one_epoch(self, dataloader):
        """
        Thực thi 1 vòng lặp huấn luyện (Training Loop) qua toàn bộ tập dữ liệu.
        Args:
            dataloader (DataLoader): Bộ nạp dữ liệu huấn luyện (Train DataLoader).
        Returns:
            tuple: (epoch_loss, epoch_acc, epoch_f1)
        """
        # Bật chế độ huấn luyện: Kích hoạt Dropout và cập nhật Running Mean/Var của BatchNorm
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        for x_batch, y_batch in dataloader:
            # Chuyển Tensor dữ liệu lên thiết bị tính toán (GPU/CPU)
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # 1. Xóa sạch Gradient tích lũy từ bước lặp trước
            self.optimizer.zero_grad()

            # 2. Lan truyền tiến (Forward Pass): Tính toán Logits đầu ra
            logits = self.model(x_batch)

            # 3. Tính toán hàm mất mát Cross-Entropy
            loss = self.criterion(logits, y_batch)

            # 4. Lan truyền ngược (Backward Pass): Tính toán Gradient của các tham số
            loss.backward()

            # 5. Cập nhật trọng số mạng nơ-ron theo Gradient
            self.optimizer.step()

            # Tích lũy giá trị Loss (nhân với số lượng mẫu trong batch để tính trung bình chuẩn)
            total_loss += loss.item() * len(y_batch)

            # Lấy nhãn dự đoán có xác suất cao nhất (argmax theo trục lớp phân loại)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        # Tính toán các chỉ số trung bình của toàn bộ Epoch
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average="macro")

        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, dataloader):
        """
        Đánh giá hiệu năng mô hình trên tập dữ liệu (Validation hoặc Test).
        Args:
            dataloader (DataLoader): Bộ nạp dữ liệu kiểm tra.
        Returns:
            tuple: (loss, acc, macro_f1)
        """
        # Chuyển mô hình sang chế độ suy luận: Tắt Dropout, cố định BatchNorm
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Tắt cơ chế tự động tính Gradient để tiết kiệm VRAM và tăng tốc độ tính toán
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        eval_loss = total_loss / len(dataloader.dataset)
        eval_acc = accuracy_score(all_targets, all_preds)
        eval_f1 = f1_score(all_targets, all_preds, average="macro")

        return eval_loss, eval_acc, eval_f1

    def fit(self, train_loader, val_loader, epochs: int):
        """
        Vòng lặp điều phối chính: Huấn luyện qua nhiều Epochs, lưu Checkpoint và ghi Log.

        Args:
            train_loader (DataLoader): Nạp tập Train.
            val_loader (DataLoader): Nạp tập Validation.
            epochs (int): Tổng số vòng lặp huấn luyện.
        """
        # Đảm bảo thư mục chứa Checkpoint đã tồn tại
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        print("\n" + "=" * 85)
        print(
            f"{'Epoch':^7} | {'Train Loss':^10} | {'Train F1':^10} | "
            f"{'Val Loss':^10} | {'Val F1':^10} | {'LR':^9} | {'Checkpoint':^10}"
        )
        print("=" * 85)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # 1. Huấn luyện 1 Epoch
            train_loss, train_acc, train_f1 = self.train_one_epoch(train_loader)

            # 2. Đánh giá trên tập Validation
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)

            # 3. Lấy Learning Rate hiện tại và cập nhật Scheduler nếu có
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                # Nếu là ReduceLROnPlateau, cập nhật dựa trên Val Macro F1
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            # 4. Kiểm tra điều kiện lưu mô hình tốt nhất (Best Checkpoint)
            checkpoint_status = ""
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.checkpoint_path)
                checkpoint_status = "⭐ BEST"

            # In thông số dòng hiện tại ra Terminal (Tracker tự động lưu vào console_output.txt)
            print(
                f"{epoch:^7d} | {train_loss:^10.4f} | {train_f1 * 100:^9.2f}% | "
                f"{val_loss:^10.4f} | {val_f1 * 100:^9.2f}% | {current_lr:^9.6f} | {checkpoint_status:^10}"
            )

            # 5. Ghi nhận chỉ số chi tiết của Epoch vào file CSV
            if self.tracker is not None:
                self.tracker.log_epoch({
                    "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "val_loss": round(val_loss, 5),
                    "train_acc": round(train_acc, 5),
                    "val_acc": round(val_acc, 5),
                    "train_macro_f1": round(train_f1, 5),
                    "val_macro_f1": round(val_f1, 5),
                    "lr": current_lr
                })

        print("=" * 85)
        print(f"🏆 Kỷ lục Validation Macro F1: {self.best_val_f1 * 100:.2f}%")
        print(f"💾 Checkpoint trọng số tối ưu: {self.checkpoint_path}")

        # 6. Tự động vẽ đồ thị đường cong học tập lưu vào experiments/plots/
        if self.tracker is not None and hasattr(self.tracker, 'csv_path'):
            plot_save_path = os.path.join(self.tracker.plot_dir, "learning_curves.png")
            plot_learning_curves(csv_path=self.tracker.csv_path, save_path=plot_save_path)