"""
===============================================================================
ENGINE HUẤN LUYỆN PRETRAIN: MASKED SENSOR MODELING (MSM)
===============================================================================
"""

import os
import csv
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.logger import setup_logger


class MaskedSSLTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        save_dir: Union[str, Path] = "checkpoints/ssl_pretrain/masked",
        checkpoint_name: str = "masked_standard_encoder_pretrained_dataset.pt",
        max_grad_norm: float = 2.0,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.save_dir / checkpoint_name
        self.max_grad_norm = max_grad_norm
        self.logger = logger or setup_logger("MaskedSSLTrainer")

        self.best_loss = float("inf")
        self.history = []

    def _extract_x(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d} [Train Masked]")
        for batch in pbar:
            x = self._extract_x(batch).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs

            loss.backward()

            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    def evaluate(self, epoch: int) -> float:
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                x = self._extract_x(batch).to(self.device)
                outputs = self.model(x)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                total_loss += loss.item()

        return total_loss / max(num_batches, 1)

    def fit(self, epochs: int):
        if epochs < 1:
            raise ValueError("Số lượng epochs phải >= 1")

        self.logger.info(f"🚀 Bắt đầu Masked Pretrain trên {self.device} trong {epochs} epochs...")
        val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch) if self.val_loader else train_loss
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr
            })

            elapsed = time.time() - t0
            self.logger.info(
                f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s"
            )

            # LƯU CHECKPOINT CHUẨN ĐỒNG BỘ: Lưu key 'encoder_state_dict'
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                encoder_to_save = getattr(self.model, "encoder", self.model)
                torch.save({"encoder_state_dict": encoder_to_save.state_dict()}, self.checkpoint_path)
                self.logger.info(f"⭐ Lưu Best Encoder Checkpoint tại: {self.checkpoint_path} (Loss: {val_loss:.5f})")

        # Ghi log lịch sử CSV
        csv_path = self.save_dir / "loss_history.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
            for row in self.history:
                writer.writerow([row["epoch"], f"{row['train_loss']:.6f}", f"{row['val_loss']:.6f}", f"{row['lr']:.8f}"])

        # Vẽ biểu đồ loss_curve.png
        try:
            plt.figure(figsize=(8, 5))
            epochs_list = [h["epoch"] for h in self.history]
            plt.plot(epochs_list, [h["train_loss"] for h in self.history], label="Train Loss", color="royalblue")
            if self.val_loader:
                plt.plot(epochs_list, [h["val_loss"] for h in self.history], label="Val Loss", color="crimson")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("Pretrain Masked Reconstruction Loss")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.save_dir / "loss_curve.png", dpi=200)
            plt.close()
        except Exception:
            pass

        self.logger.info(f"✅ Hoàn tất pretrain! File trọng số sẵn sàng tại: {self.checkpoint_path}")
        return self.history