import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from models.har_classifier import HARClassifier


def train_and_eval_finetune(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        num_classes,
        in_channels=6,
        encoder_checkpoint_path=None,  # Truyền path .pt hoặc None nếu train Scratch
        device="cuda",
        epochs=60,
):
    batch_size = 32 if len(x_train) < 100 else 64
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=64,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=64,
        shuffle=False
    )

    # khoi tao mo hinh
    model = HARClassifier(in_channels=in_channels, num_classes=num_classes).to(device)

    if not os.path.exists(encoder_checkpoint_path):
        raise FileExistsError(f"khong tim thay file trong so ssl tai: {encoder_checkpoint_path}")

    # nap trong so ssl vao backbone
    checkpoint_encoder = torch.load(encoder_checkpoint_path, map_location=device, weights_only=True)
    encoder_dict = checkpoint_encoder["encoder"] if "encoder" in checkpoint_encoder else checkpoint_encoder
    model.encoder.load_state_dict(encoder_dict)

    # mo khoa toan bo trong so de fine tune
    for param in model.parameters():
        param.requires_grad = True

    # toi uu hoa learning rate phan tang
    optimizer = Adam([
        {"params": model.encoder.parameters(), "lr": 1e-4, "weight_decay": 1e-4},  # backbone hoc cham
        {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-4},  # head hoc nhanh
    ])

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    best_val_f1 = -1
    best_model_state = None

    # vong lap huan luyen fine tune
    epochs = 50 if len(x_train) > 200 else 70
    for epoch in range(epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    # danh gia tren tap Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            preds = torch.argmax(model(x_b), dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(y_b.cpu().numpy())

    val_f1 = f1_score(val_targets, val_preds, average="macro")
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }

    # nap lai trong so tot nhat danh gia tren tap test
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            preds = torch.argmax(model(x_b), dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(y_b.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds, average="macro") * 100
    return test_acc, test_f1, best_val_f1, best_model_state
