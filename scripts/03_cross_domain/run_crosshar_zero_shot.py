"""
===============================================================================
CLI SCRIPT: HUẤN LUYỆN CLASSIFIER NGUỒN VÀ ZERO-SHOT TEST CHUẨN CROSSHAR
===============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from config.hhar_config import HHARConfig

from datasets.crosshar_dataset import channel_instance_norm
from models.encoders.builder import build_encoder
from models.ssl.masked.crosshar_model import CrossHARClassifier
from engines.evaluation.evaluator import ModelEvaluator

COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_CLASSES = len(COMMON_CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATHS = {
    "motionsense": {
        "train": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
        "val": Path(MotionSenseConfig.PROCESSED_VAL_PATH),
        "test": Path(MotionSenseConfig.PROCESSED_TEST_PATH),
        "in_channels": int(MotionSenseConfig.IN_CHANNELS)
    },
    "uci_har": {
        "train": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
        "val": Path(UCIHARConfig.PROCESSED_VAL_PATH),
        "test": Path(UCIHARConfig.PROCESSED_TEST_PATH),
        "in_channels": int(UCIHARConfig.IN_CHANNELS)
    },
    "hhar_phone": {
        "train": HHARConfig.PROCESSED_DIR_PHONE / "train.pt",
        "val": HHARConfig.PROCESSED_DIR_PHONE / "val.pt",
        "test": HHARConfig.PROCESSED_DIR_PHONE / "test.pt",
        "in_channels": int(HHARConfig.IN_CHANNELS)
    },
    "hhar_watch": {
        "train": HHARConfig.PROCESSED_DIR_WATCH / "train.pt",
        "val": HHARConfig.PROCESSED_DIR_WATCH / "val.pt",
        "test": HHARConfig.PROCESSED_DIR_WATCH / "test.pt",
        "in_channels": int(HHARConfig.IN_CHANNELS)
    }
}

parser = argparse.ArgumentParser(description="CrossHAR Zero-Shot Transfer Runner")
parser.add_argument("--source", type=str, required=True,
                    choices=["motionsense", "uci_har", "hhar_phone", "hhar_watch"],
                    help="Miền nguồn (Source domain)")
parser.add_argument("--targets", nargs="+", default=["motionsense", "uci_har", "hhar_phone", "hhar_watch"],
                    help="Danh sách miền đích cần đánh giá zero-shot")
parser.add_argument("--backbone", type=str, default="standard",
                    choices=["standard", "cnn_transformer"],
                    help="Loại kiến trúc backbone")
parser.add_argument("--epochs", type=int, default=40, help="Số epoch fine-tune Head trên nguồn")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--freeze_backbone", action="store_true",
                    help="Đóng băng backbone trong quá trình train classifier trên nguồn")
args = parser.parse_args()


def load_normalized_tensor(pt_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    X = raw["samples"]
    y = raw["labels"].squeeze()

    mask = (y >= 0) & (y < 5)
    X, y = X[mask], y[mask]

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.float()

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)
    else:
        y = y.long()

    if X.ndim == 3 and X.shape[1] == 128 and X.shape[2] == 6:
        X = X.permute(0, 2, 1)

    X_norm = channel_instance_norm(X)
    return X_norm, y


def get_source_10_percent_split(train_path: Path, seed: int = 42):
    X, y = load_normalized_tensor(train_path)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    _, finetune_idx = next(sss.split(X, y.numpy()))
    return X[finetune_idx], y[finetune_idx]


def main():
    src = args.source
    targets = [t for t in args.targets if t != src]

    print("\n" + "=" * 90)
    print("🌐 THỰC THI ZERO-SHOT BENCHMARK CHUẨN CROSSHAR")
    print(f"🎯 NGUỒN: [{src.upper()}] ➔ ĐÍCH: {[t.upper() for t in targets]}")
    print(f"🔧 Backbone: {args.backbone.upper()} | Đóng băng Backbone: {args.freeze_backbone}")
    print("=" * 90)

    # 1. KIỂM TRA CHECKPOINT PRETRAIN CROSSHAR
    ssl_ckpt = (PROJECT_ROOT / "checkpoints" / "crosshar_pretrain" / src /
                args.backbone / f"crosshar_{args.backbone}_encoder_pretrained_{src}.pt")
    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy Checkpoint tại: {ssl_ckpt}\n"
                                f"👉 Hãy chạy 'scripts/02_pretrain_ssl/run_crosshar_pretrain.py --datasets {src}' trước!")

    # 2. CHUẨN BỊ DỮ LIỆU MIỀN NGUỒN
    src_cfg = DATASET_PATHS[src]
    X_src_10, y_src_10 = get_source_10_percent_split(src_cfg["train"], seed=args.seed)
    X_src_val, y_src_val = load_normalized_tensor(src_cfg["val"])
    X_src_test, y_src_test = load_normalized_tensor(src_cfg["test"])

    print(f"📊 Dữ liệu miền nguồn ({src.upper()}):")
    print(f"   - 10% Train có nhãn   : {X_src_10.shape[0]} mẫu")
    print(f"   - 100% Val nguồn      : {X_src_val.shape[0]} mẫu")
    print(f"   - 100% Test In-domain : {X_src_test.shape[0]} mẫu")

    train_loader = DataLoader(TensorDataset(X_src_10, y_src_10), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_src_val, y_src_val), batch_size=args.batch_size, shuffle=False)
    test_loader_src = DataLoader(TensorDataset(X_src_test, y_src_test), batch_size=args.batch_size, shuffle=False)

    # 3. KHỞI TẠO MÔ HÌNH PHÂN LOẠI CROSSHAR
    encoder = build_encoder(backbone_type=args.backbone, in_channels=src_cfg["in_channels"]).to(DEVICE)
    encoder_state = torch.load(ssl_ckpt, map_location=DEVICE, weights_only=True)
    encoder.load_state_dict(encoder_state, strict=True)
    print(f"📦 Đã nạp thành công Encoder CrossHAR Pretrained từ: {ssl_ckpt.name}")

    with torch.no_grad():
        dummy_in = torch.randn(2, src_cfg["in_channels"], 128).to(DEVICE)
        dummy_out = encoder(dummy_in)
        detected_feat_len = dummy_out.shape[2]
    model = CrossHARClassifier(
        encoder=encoder,
        feature_dim=128,
        feature_length=detected_feat_len,
        num_classes=NUM_CLASSES,
        dropout=0.3
    ).to(DEVICE)

    if args.freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer = Adam(list(model.context_module.parameters()) + list(model.classifier.parameters()), lr=args.lr)
    else:
        for p in model.encoder.parameters():
            p.requires_grad = True
        optimizer = Adam([
            {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
            {"params": model.context_module.parameters(), "lr": args.lr},
            {"params": model.classifier.parameters(), "lr": args.lr}
        ])

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # 4. HUẤN LUYỆN CLASSIFIER TRÊN 10% NGUỒN
    best_val_f1 = -1.0
    best_state = None

    print("\n⏳ Đang huấn luyện Transformer Context + FeatureMixer Head trên miền nguồn...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, b_cnt = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            b_cnt += 1

        avg_loss = train_loss / max(b_cnt, 1)

        model.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = xv.to(DEVICE)
                preds = torch.argmax(model(xv), dim=1)
                v_preds.extend(preds.cpu().numpy())
                v_targets.extend(yv.numpy())

        v_acc = accuracy_score(v_targets, v_preds) * 100
        v_f1 = f1_score(v_targets, v_preds, average="macro") * 100
        scheduler.step(v_f1)

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_loss:.4f} | "
                  f"Val Acc: {v_acc:5.2f}% | Val Macro F1: {v_f1:5.2f}% (Best: {best_val_f1:5.2f}%)")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    save_dir = PROJECT_ROOT / "checkpoints" / "crosshar_zero_shot_experiments" / f"{src}_{args.backbone}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_dir / "best_crosshar_full_model.pt")

    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))
    results_summary = {}

    # 5. IN-DOMAIN TEST
    print("\n" + "-" * 75)
    print(f"🏠 ĐÁNH GIÁ IN-DOMAIN: [{src.upper()} ➔ {src.upper()}] (Test Set)")
    print("-" * 75)
    in_cm_path = str(save_dir / f"cm_in_domain_{src}.png")
    in_res = evaluator.evaluate(model=model, test_loader=test_loader_src,
                                plot_save_path=in_cm_path, title_prefix=f"CrossHAR In-Domain: {src.upper()}")

    in_acc = in_res.get("accuracy", 0.0)
    in_f1 = in_res.get("macro_f1", 0.0)
    if in_acc <= 1.0: in_acc *= 100.0
    if in_f1 <= 1.0: in_f1 *= 100.0

    results_summary[f"{src} (In-Domain)"] = {"accuracy": in_acc, "macro_f1": in_f1}
    print(f"👉 In-Domain Macro F1: {in_f1:.2f}% | Acc: {in_acc:.2f}%")

    # 6. ZERO-SHOT TEST TRÊN CÁC MIỀN ĐÍCH
    for tgt in targets:
        print("\n" + "-" * 75)
        print(f"🚀 ĐÁNH GIÁ ZERO-SHOT: [{src.upper()} ➔ {tgt.upper()}] (0% Nhãn đích)")
        print("-" * 75)
        tgt_cfg = DATASET_PATHS[tgt]
        dataset_all_path = tgt_cfg["test"].parent / "dataset_all.pt"
        X_tgt_test, y_tgt_test = load_normalized_tensor(dataset_all_path)
        test_loader_tgt = DataLoader(TensorDataset(X_tgt_test, y_tgt_test), batch_size=args.batch_size, shuffle=False)

        tgt_cm_path = str(save_dir / f"cm_zero_shot_{src}_to_{tgt}.png")
        tgt_res = evaluator.evaluate(model=model, test_loader=test_loader_tgt,
                                     plot_save_path=tgt_cm_path, title_prefix=f"CrossHAR Zero-Shot: {src.upper()}->{tgt.upper()}")

        t_acc = tgt_res.get("accuracy", 0.0)
        t_f1 = tgt_res.get("macro_f1", 0.0)
        if t_acc <= 1.0: t_acc *= 100.0
        if t_f1 <= 1.0: t_f1 *= 100.0

        results_summary[f"{src} -> {tgt} (Zero-Shot)"] = {"accuracy": t_acc, "macro_f1": t_f1}
        print(f"👉 Zero-Shot Macro F1 ({tgt}): {t_f1:.2f}% | Acc: {t_acc:.2f}%")

    with open(save_dir / "zero_shot_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)

    # 7. IN BẢNG TỔNG KẾT
    print("\n" + "=" * 90)
    print(f"🏆 BẢNG TỔNG HỢP ZERO-SHOT TRANSFER (CHUẨN CROSSHAR - NGUỒN: {src.upper()})")
    print("=" * 90)
    print(f"{'Kịch bản chuyển giao':<38} | {'Accuracy (%)':<15} | {'Macro F1 (%)'}")
    print("-" * 90)
    for scen, met in results_summary.items():
        print(f"{scen:<38} | {met['accuracy']:<15.2f} | {met['macro_f1']:.2f}%")
    print("=" * 90)
    print(f"📁 Checkpoint và biểu đồ lưu tại: {save_dir}")


if __name__ == "__main__":
    main()
