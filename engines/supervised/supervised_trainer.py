"""
===============================================================================
MODULE: SUPERVISED TRAINER ENGINE
===============================================================================
VAI TRÒ TRONG HỆ THỐNG:
    - Điều phối toàn bộ quy trình huấn luyện học có giám sát (Supervised Learning).
    - Phục vụ xây dựng mô hình cơ sở (Baseline) huấn luyện từ đầu (Train from Scratch)
      để đối chiếu hiệu năng trực tiếp với giải pháp Học tự giám sát (SSL).

NHIỆM VỤ CHÍNH:
    1. Quản lý vòng lặp epoch (Forward pass, Backward pass, cập nhật trọng số).
    2. Đánh giá tính tổng quát hóa trên tập Validation sau mỗi epoch.
    3. Triển khai cơ chế Model Checkpointing: Chỉ lưu lại trọng số có Macro F1 cao nhất,
       tránh hiện tượng Overfitting khi kết thúc huấn luyện.
    4. Điều chỉnh tốc độ học (Learning Rate Scheduler) dựa trên phản hồi của tập Val.

ĐẦU VÀO / ĐẦU RA:
    - Đầu vào: Mô hình HARClassifier, DataLoaders (Train, Val) shape (B, 6, 128).
    - Đầu ra: Mô hình tối ưu nhất nạp lại từ checkpoint tốt nhất, file trọng số .pt.
===============================================================================
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SupervisedTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            checkpoint_dir: str = "",
            scheduler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.best_val_f1 = -1.0
        self.best_model_state = None

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average="macro")
        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

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

    def fit(self,
            train_loader,
            val_loader,
            epochs: int,
            model_name: str = "best_model.pt"):
        print(f"🚀 Bắt đầu huấn luyện Supervised ({epochs} epochs)...")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_f1)

            # Lưu trọng số tốt nhất theo Macro F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if self.checkpoint_dir:
                    torch.save(self.best_model_state, os.path.join(self.checkpoint_dir, model_name))

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% F1: {val_f1 * 100:.2f}%")

        # Nạp lại trọng số tốt nhất cho model
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        return self.model

    '''
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# Tự động nhận diện thư mục gốc để đảm bảo import đúng module từ mọi vị trí thực thi
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SupervisedTrainer:
    """
    Bộ điều phối huấn luyện và kiểm thử mô hình có giám sát chuẩn mực cho bài toán HAR.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: torch.device,
            checkpoint_dir: str = "",
            scheduler=None,
    ):
        """
        Khởi tạo Engine huấn luyện.

        Tham số:
            model (nn.Module): Mạng nơ-ron cần huấn luyện (thường là HARClassifier).
            optimizer (torch.optim.Optimizer): Thuật toán tối ưu (Adam, AdamW...).
            criterion (nn.Module): Hàm mất mát (thường là nn.CrossEntropyLoss).
            device (torch.device): Phần cứng tính toán ('cuda' hoặc 'cpu').
            checkpoint_dir (str, optional): Thư mục lưu file trọng số .pt tốt nhất.
            scheduler (optional): Bộ điều chỉnh Learning Rate theo epoch (ví dụ ReduceLROnPlateau).
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler

        # Biến theo dõi kỷ lục Validation Macro F1 để lưu mô hình tốt nhất
        self.best_val_f1 = -1.0
        self.best_model_state = None

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_one_epoch(self, dataloader):
        """
        Thực thi 1 epoch huấn luyện (cập nhật Gradient trên từng Batch).

        Tham số:
            dataloader (DataLoader): DataLoader chứa dữ liệu tập Train.

        Trả về:
            tuple: (epoch_loss, epoch_accuracy, epoch_macro_f1)
        """
        # Chuyển mô hình sang chế độ huấn luyện (bật Dropout, cập nhật BatchNorm chạy)
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for x_batch, y_batch in dataloader:
            # Chuyển tensor sang GPU/CPU tương ứng: x_batch (B, 6, 128), y_batch (B,)
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Xóa gradient tích lũy từ bước trước
            self.optimizer.zero_grad()

            # Lan truyền tiến (Forward): Tính xác suất thô (Logits) shape (B, Num_Classes)
            logits = self.model(x_batch)

            # Tính hàm mất mát Cross Entropy
            loss = self.criterion(logits, y_batch)

            # Lan truyền ngược (Backward): Tính đạo hàm riêng cho từng tham số
            loss.backward()

            # Cập nhật trọng số mạng theo hướng giảm dốc sai số
            self.optimizer.step()

            # Tích lũy mất mát theo kích thước mẫu thực tế (tránh sai số do batch cuối bị lẻ)
            total_loss += loss.item() * len(y_batch)

            # Dự đoán lớp có xác suất cao nhất (argmax theo chiều nhãn)
            preds = torch.argmax(logits, dim=1)

            # Thu thập nhãn để tính chỉ số thống kê của toàn bộ epoch
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average="macro")
        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, dataloader):
        """
        Đánh giá hiệu năng mô hình trên một tập dữ liệu bất kỳ mà không cập nhật trọng số.

        Tham số:
            dataloader (DataLoader): Dữ liệu đánh giá (Val hoặc Test).

        Trả về:
            tuple: (eval_loss, eval_accuracy, eval_macro_f1)
        """
        # Chuyển mô hình sang chế độ suy luận (tắt Dropout, cố định mean/std của BatchNorm)
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        # Tắt bộ theo dõi gradient để tiết kiệm VRAM và tăng tốc độ tính toán
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

    def fit(self, train_loader, val_loader, epochs: int, model_name: str = "best_model.pt"):
        """
        Thực thi toàn bộ quy trình huấn luyện đa epoch kèm theo dõi và lưu checkpoint.

        Tham số:
            train_loader (DataLoader): Nạp dữ liệu tập Train.
            val_loader (DataLoader): Nạp dữ liệu tập Validation.
            epochs (int): Tổng số vòng lặp huấn luyện.
            model_name (str): Tên file lưu trọng số tối ưu.

        Trả về:
            nn.Module: Mô hình đã được nạp lại bộ trọng số tốt nhất trong lịch sử huấn luyện.
        """
        print(f"🚀 Bắt đầu huấn luyện Supervised ({epochs} epochs)...")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)

            # Cập nhật Learning Rate nếu dùng Scheduler
            if self.scheduler is not None:
                self.scheduler.step(val_f1)

            # Cơ chế chọn lọc tự nhiên (Best Model Selection):
            # So sánh Macro F1 của tập Val (thước đo khách quan khi tập dữ liệu mất cân bằng nhãn)
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                # Sao chép bản sao trọng số an toàn lên CPU
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if self.checkpoint_dir:
                    save_full_path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(self.best_model_state, save_full_path)

            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"Epoch [{epoch:03d}/{epochs:03d}] | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%"
                )

        # Sau khi train xong, nạp lại đúng thời điểm phong độ cao nhất cho model trước khi trả về
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
            print(f"🎯 Đã nạp lại trọng số tốt nhất đạt Val Macro F1: {self.best_val_f1*100:.2f}%")

        return self.model
    '''
