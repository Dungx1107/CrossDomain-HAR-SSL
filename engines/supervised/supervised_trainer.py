"""
===============================================================================
ENGINE: CONTRASTIVE PRETRAINING TRAINER CHO TS-TCC
===============================================================================
Tính năng chuẩn hóa:
    1. Full-pipeline Resume: Tự động hoặc chủ động nạp lại checkpoint (model,
       optimizer, scheduler, start_epoch, best_score), tiếp tục ghi log đúng cột.
    2. Online Linear Probing: Định kỳ đánh giá chất lượng biểu diễn đặc trưng
       trên downstream validation set, lấy Macro F1-score để lưu best_model.pt.
    3. Gradient Monitoring: Theo dõi chuẩn Gradient thực tế trước khi cắt (clip).
    4. Guard checks an toàn chống chia cho 0.
===============================================================================
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional
from sklearn.metrics import f1_score


class ContrastiveTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        probe_train_loader: Optional[DataLoader] = None,
        probe_val_loader: Optional[DataLoader] = None,
        num_classes: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        epochs: int = 100,
        checkpoint_dir: str = "./checkpoints/ssl_pretrain",
        log_interval: int = 10,
        probe_interval: int = 10,
        max_grad_norm: float = 2.0,
        resume_from: Optional[str] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.probe_train_loader = probe_train_loader
        self.probe_val_loader = probe_val_loader
        self.num_classes = num_classes
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.probe_interval = probe_interval
        self.max_grad_norm = max_grad_norm

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=3e-4
        )
        self.scheduler = scheduler

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_csv = os.path.join(self.checkpoint_dir, "loss_history.csv")

        # Trạng thái ban đầu phục vụ Resume
        self.start_epoch = 1
        self.best_probe_f1 = -1.0
        self.best_loss = float("inf")

        # Nạp trạng thái nếu có yêu cầu Resume
        if resume_from:
            self._load_checkpoint(resume_from)
        else:
            self._init_csv(overwrite=True)

    def _init_csv(self, overwrite: bool = True):
        mode = "w" if overwrite else "a"
        with open(self.history_csv, mode=mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["epoch", "train_loss_total", "train_loss_tc", "train_loss_cc", "grad_norm"]
            if self.val_loader is not None:
                header.extend(["val_loss_total", "val_loss_tc", "val_loss_cc"])
            if self.probe_val_loader is not None:
                header.extend(["probe_f1", "probe_acc"])
            header.append("lr")
            writer.writerow(header)

    def _load_checkpoint(self, checkpoint_path: str):
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"❌ Không tìm thấy checkpoint tại: {checkpoint_path}")

        print(f"🔄 Đang nạp checkpoint từ: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_loss = checkpoint.get("loss", float("inf"))
        self.best_probe_f1 = checkpoint.get("probe_f1", -1.0)

        print(f"   Khôi phục thành công! Tiếp tục từ Epoch {self.start_epoch} (Best F1: {self.best_probe_f1:.4f})")

        # Khởi tạo CSV dạng nối tiếp
        if not os.path.exists(self.history_csv):
            self._init_csv(overwrite=True)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, total_tc, total_cc, total_norm = 0.0, 0.0, 0.0, 0.0
        num_batches = max(1, len(self.train_loader))

        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.epochs}]", leave=False)
        for batch in pbar:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("Batch đầu vào phải gồm ít nhất 2 views: (x_weak, x_strong).")

            x_weak = batch[0].to(self.device, non_blocking=True)
            x_strong = batch[1].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            loss, loss_tc, loss_cc = self.model(x_weak, x_strong)
            loss.backward()

            # Giám sát và Clip Gradient Norm
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.max_grad_norm
            ).item() if self.max_grad_norm > 0 else 0.0

            self.optimizer.step()

            total_loss += loss.item()
            total_tc += loss_tc.item()
            total_cc += loss_cc.item()
            total_norm += grad_norm

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "TC": f"{loss_tc.item():.4f}",
                "CC": f"{loss_cc.item():.4f}",
                "|g|": f"{grad_norm:.2f}"
            })

        return {
            "train_loss_total": total_loss / num_batches,
            "train_loss_tc": total_tc / num_batches,
            "train_loss_cc": total_cc / num_batches,
            "grad_norm": total_norm / num_batches
        }

    @torch.no_grad()
    def evaluate_contrastive(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_tc, total_cc = 0.0, 0.0, 0.0
        num_batches = max(1, len(self.val_loader))

        for batch in self.val_loader:
            x_weak = batch[0].to(self.device, non_blocking=True)
            x_strong = batch[1].to(self.device, non_blocking=True)

            loss, loss_tc, loss_cc = self.model(x_weak, x_strong)

            total_loss += loss.item()
            total_tc += loss_tc.item()
            total_cc += loss_cc.item()

        return {
            "val_loss_total": total_loss / num_batches,
            "val_loss_tc": total_tc / num_batches,
            "val_loss_cc": total_cc / num_batches
        }

    def run_linear_probe(self, probe_epochs: int = 5) -> Dict[str, float]:
        """
        Đánh giá chất lượng biểu diễn thực tế:
        Đóng băng Encoder, train 1 lớp Linear duy nhất trong vài epoch ngắn trên tập có nhãn.
        """
        if not (self.probe_train_loader and self.probe_val_loader and self.num_classes):
            return {}

        self.model.eval()
        feature_dim = self.model.encoder.feature_dim if hasattr(self.model.encoder, "feature_dim") else 128
        classifier = nn.Linear(feature_dim, self.num_classes).to(self.device)
        probe_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Huấn luyện nhanh Classifier trên Frozen Encoder
        for _ in range(probe_epochs):
            classifier.train()
            for x, y in self.probe_train_loader:
                x, y = x.to(self.device), y.to(self.device)
                with torch.no_grad():
                    feats = self.model.encoder(x)
                    if feats.dim() == 3:
                        feats = feats.mean(dim=-1)

                probe_optimizer.zero_grad()
                out = classifier(feats)
                loss = criterion(out, y)
                loss.backward()
                probe_optimizer.step()

        # Đánh giá trên Validation Set
        classifier.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in self.probe_val_loader:
                x, y = x.to(self.device), y.to(self.device)
                feats = self.model.encoder(x)
                if feats.dim() == 3:
                    feats = feats.mean(dim=-1)
                preds = classifier(feats).argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()

        return {"probe_f1": f1, "probe_acc": acc}

    def train(self):
        print(f"\n🚀 BẮT ĐẦU TIỀN HUẤN LUYỆN TS-TCC")
        print(f"   - Epochs: {self.start_epoch} -> {self.epochs}")
        print(f"   - Thiết bị: {self.device}")
        print(f"   - Checkpoint Directory: {self.checkpoint_dir}\n" + "-" * 75)

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.evaluate_contrastive() if self.val_loader else {}

            # Chạy Linear Probe định kỳ để đo chất lượng biểu diễn thực tế
            probe_metrics = {}
            if (epoch % self.probe_interval == 0 or epoch == self.epochs) and self.probe_train_loader:
                probe_metrics = self.run_linear_probe()

            # Scheduler update
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Nếu có probe_f1 thì tối ưu theo probe_f1, nếu không theo loss
                    metric = probe_metrics.get("probe_f1", val_metrics.get("val_loss_total", train_metrics["train_loss_total"]))
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # Ghi nhật ký CSV an toàn
            row = [
                epoch,
                f"{train_metrics['train_loss_total']:.6f}",
                f"{train_metrics['train_loss_tc']:.6f}",
                f"{train_metrics['train_loss_cc']:.6f}",
                f"{train_metrics['grad_norm']:.4f}"
            ]
            if self.val_loader:
                row.extend([
                    f"{val_metrics['val_loss_total']:.6f}",
                    f"{val_metrics['val_loss_tc']:.6f}",
                    f"{val_metrics['val_loss_cc']:.6f}"
                ])
            if self.probe_val_loader:
                row.extend([
                    f"{probe_metrics.get('probe_f1', 0.0):.4f}",
                    f"{probe_metrics.get('probe_acc', 0.0):.4f}"
                ])
            row.append(f"{current_lr:.8f}")

            with open(self.history_csv, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            # In thông tin theo dõi
            if epoch % self.log_interval == 0 or epoch == self.epochs:
                msg = (
                    f"Epoch [{epoch:03d}/{self.epochs:03d}] | "
                    f"Train Loss: {train_metrics['train_loss_total']:.4f} "
                    f"(TC: {train_metrics['train_loss_tc']:.4f}, CC: {train_metrics['train_loss_cc']:.4f}, |g|: {train_metrics['grad_norm']:.2f})"
                )
                if probe_metrics:
                    msg += f" | Probe F1: {probe_metrics['probe_f1']:.4f}"
                msg += f" | LR: {current_lr:.6f}"
                print(msg)

            # Lựa chọn và lưu Checkpoint tốt nhất (Best Model)
            if probe_metrics:
                # Tiêu chuẩn vàng: Chọn best theo năng lực phân loại thực tế
                if probe_metrics["probe_f1"] > self.best_probe_f1:
                    self.best_probe_f1 = probe_metrics["probe_f1"]
                    self.save_checkpoint("best_model.pt", epoch, train_metrics["train_loss_total"], probe_metrics["probe_f1"], full_state=False)
            else:
                # Dự phòng nếu không có probe set: chọn theo contrastive loss
                target_loss = val_metrics.get("val_loss_total", train_metrics["train_loss_total"])
                if target_loss < self.best_loss:
                    self.best_loss = target_loss
                    self.save_checkpoint("best_model.pt", epoch, target_loss, full_state=False)

            # Luôn cập nhật last_checkpoint để sẵn sàng Resume bất cứ lúc nào
            self.save_checkpoint("last_checkpoint.pt", epoch, train_metrics["train_loss_total"], probe_metrics.get("probe_f1", 0.0), full_state=True)

        print("-" * 75)
        print(f"✅ HOÀN TẤT. Weights tốt nhất lưu tại: {self.checkpoint_dir}")

    def save_checkpoint(self, filename: str, epoch: int, loss: float, probe_f1: float = 0.0, full_state: bool = False):
        filepath = os.path.join(self.checkpoint_dir, filename)
        data = {
            "epoch": epoch,
            "loss": loss,
            "probe_f1": probe_f1,
            "model_state_dict": self.model.state_dict(),
            "encoder_state_dict": self.model.encoder.state_dict(),
        }
        if full_state:
            data["optimizer_state_dict"] = self.optimizer.state_dict()
            if self.scheduler:
                data["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(data, filepath)