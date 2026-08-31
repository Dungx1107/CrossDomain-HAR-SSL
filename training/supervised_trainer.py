"""
===============================================================================
MỤC ĐÍCH:
    Cung cấp Engine điều phối toàn bộ quá trình huấn luyện và đánh giá mô hình
    có giám sát (Supervised Learning).
===============================================================================
"""

import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


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
            tracker=None,
            checkpoint_path: str = "",
            scheduler=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.tracker = tracker
        self.checkpoint_path = checkpoint_path
        self.scheduler = scheduler
        self.best_val_f1 = 0.0

    def train_one_epoch(self, dataloader):
        """
        Thực thi 1 vòng lặp huấn luyện qua toàn bộ dataloader.
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_targets = []

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
        """
        Đánh giá hiệu năng mô hình trên dataloader.
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

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