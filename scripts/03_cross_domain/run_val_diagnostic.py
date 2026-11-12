"""
===============================================================================
SCRIPT: VAL DIAGNOSTIC CHO FEW-SHOT CROSS-DOMAIN
===============================================================================
Đặt file tại: scripts/03_cross_domain/run_val_diagnostic.py

Mục đích: trả lời "chọn epoch bằng val có cải thiện kết quả đáng kể so với
fixed-epoch không?" mà KHÔNG đụng vào pipeline hiện tại.

Thiết kế (cặp đôi trong cùng một lần chạy):
    - Quỹ đạo huấn luyện giống hệt giao thức fixed-epoch: scheduler theo Train
      Loss, cùng optimizer/LR/loss như train_and_eval_finetune.
    - Mỗi epoch ghi Val F1 (tập val con, <= val_per_class mẫu/lớp) và Test F1.
    - Phân tích sau (analyze_val_diagnostic.py) so, trên CÙNG một run:
        * F1 test ở epoch cố định (vd 40)
        * F1 test ở epoch có Val F1 cao nhất (chọn bằng val)
        * F1 test cao nhất mọi epoch (oracle, nhìn trộm test, chỉ để tham khảo)
    - Test KHÔNG tham gia chọn epoch; đường cong test chỉ phục vụ chẩn đoán.

Hỗ trợ chạy tiếp (resume): run nào đã có file JSON sẽ được bỏ qua.
===============================================================================
"""

import sys
import json
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

from utils.checkpoint_paths import ssl_checkpoint_path
from engines.transfer.finetune_trainer import _train_one_epoch, _build_scheduler
from engines.transfer.model_builder import (
    build_transfer_model, configure_optimizer, VALID_FREEZE_MODES,
)
from engines.evaluation.evaluator import predict, compute_metrics
from models.encoders.builder import build_encoder
from config.datasets.registry import DATASETS
from config.datasets.base_dataset_config import BaseDatasetConfig
from utils.data_io import load_data
from utils.sampling import sample_k_shot

NUM_COMMON_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Chẩn đoán ảnh hưởng của val lên few-shot")
    p.add_argument("--pairs", nargs="+", required=True, help="vd: motion_sense:uci_har uci_har:hhar_phone")
    p.add_argument("--method", type=str, default="tstcc")
    p.add_argument("--backbone", type=str, default="standard")
    p.add_argument("--pretrain_tag", type=str, default="")
    p.add_argument("--k_shots", nargs="+", type=int, default=[10, 30, 100])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024])
    p.add_argument("--freeze_modes", nargs="+", type=str, default=["none", "full"])
    p.add_argument("--epochs", type=int, default=60,
                   help="Chạy dài hơn giao thức (40) để thấy đường cong; epoch-40 của run 60 epoch "
                        "giống hệt run 40 epoch vì scheduler chỉ phụ thuộc quá khứ")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_per_class", type=int, default=500)
    p.add_argument("--val_seed", type=int, default=0, help="Cố định, dùng chung mọi phương pháp")
    p.add_argument("--out_name", type=str, default="val_diagnostic")
    return p.parse_args()


def parse_mode(mode_str):
    mode, _, blocks = mode_str.partition(":")
    if mode not in VALID_FREEZE_MODES:
        raise ValueError(f"freeze_mode '{mode}' không hợp lệ. Chọn trong {VALID_FREEZE_MODES}")
    n = int(blocks) if blocks else 1
    name = {"full": "linear_probing", "none": "full_finetuning"}.get(mode, f"partial_{n}blocks")
    return mode, n, name


def subsample_per_class(x, y, n_per_class, seed, num_classes):
    """Lấy tối đa n_per_class mẫu mỗi lớp, cố định theo seed. Trả (x, y, số mẫu thực tế mỗi lớp)."""
    assert y.dim() == 1, "Cần nhãn dạng chỉ số lớp 1D (không phải one-hot)"
    g = torch.Generator().manual_seed(seed)
    keep, counts = [], []
    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        if len(idx) > n_per_class:
            idx = idx[torch.randperm(len(idx), generator=g)[:n_per_class]]
        keep.append(idx)
        counts.append(int(len(idx)))
    keep = torch.cat(keep)
    return x[keep], y[keep], counts


def run_one(args, ckpt, X_tr, y_tr, val_loader, test_loader, val_counts, src, tgt, k, mode, seed, out_path):
    freeze_mode, num_blocks, display = mode

    # Thứ tự khởi tạo/seed giống run_few_shot.py
    torch.manual_seed(seed)
    x_sub, y_sub = sample_k_shot(X_tr, y_tr, k, seed, NUM_COMMON_CLASSES)
    tr_loader = DataLoader(TensorDataset(x_sub, y_sub),
                           batch_size=min(args.batch_size, len(x_sub)), shuffle=True)

    model = build_transfer_model(
        encoder_checkpoint_path=ckpt,
        encoder=build_encoder(args.backbone, BaseDatasetConfig.IN_CHANNELS),
        num_classes=NUM_COMMON_CLASSES,
        in_channels=BaseDatasetConfig.IN_CHANNELS,
        device=DEVICE,
    )
    optimizer = configure_optimizer(model, freeze_mode=freeze_mode, num_unfrozen_blocks=num_blocks)
    scheduler = _build_scheduler(optimizer, has_val=False)   # theo Train Loss như fixed-epoch
    criterion = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "val_f1": [], "val_acc": [], "test_f1": [], "test_acc": []}
    t0 = time.time()
    for _ in range(args.epochs):
        loss = _train_one_epoch(model, tr_loader, optimizer, criterion, DEVICE)
        scheduler.step(loss)
        v = compute_metrics(*predict(model, val_loader, DEVICE))
        t = compute_metrics(*predict(model, test_loader, DEVICE))
        hist["train_loss"].append(round(float(loss), 5))
        hist["val_f1"].append(round(float(v["macro_f1"]), 3))
        hist["val_acc"].append(round(float(v["accuracy"]), 3))
        hist["test_f1"].append(round(float(t["macro_f1"]), 3))
        hist["test_acc"].append(round(float(t["accuracy"]), 3))

    record = {
        "source": src, "target": tgt, "k": k, "mode": display, "seed": seed,
        "method": args.method, "backbone": args.backbone, "pretrain_tag": args.pretrain_tag,
        "epochs": args.epochs, "train_samples": int(len(y_sub)),
        "val_per_class_counts": val_counts, "checkpoint": str(ckpt),
        "run_time_sec": round(time.time() - t0, 1), "history": hist,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return hist


def main():
    args = parse_args()
    modes = [parse_mode(m) for m in args.freeze_modes]
    tag_suffix = f"_{args.pretrain_tag}" if args.pretrain_tag else ""
    root = OUTPUT_ROOT / "outputs_evaluation" / args.out_name / args.method / f"{args.backbone}{tag_suffix}"

    total = len(args.pairs) * len(args.k_shots) * len(modes) * len(args.seeds)
    done, t_start = 0, time.time()
    print(f"🔬 VAL DIAGNOSTIC | method={args.method} | backbone={args.backbone}{tag_suffix} | "
          f"{total} run x {args.epochs} epoch | val<= {args.val_per_class}/lớp | device={DEVICE}")

    for pair in args.pairs:
        src, tgt = [s.strip() for s in pair.split(":")]
        cfg = DATASETS[tgt]

        ckpt = ssl_checkpoint_path(OUTPUT_ROOT, method=args.method, dataset=src,
                                   backbone=args.backbone, tag=args.pretrain_tag)
        if not ckpt.exists():
            raise FileNotFoundError(f"❌ Thiếu checkpoint pretrain tại: {ckpt}")

        X_tr, y_tr = load_data(cfg["train"], NUM_COMMON_CLASSES)
        X_v, y_v = load_data(cfg["val"], NUM_COMMON_CLASSES)
        X_ts, y_ts = load_data(cfg["test"], NUM_COMMON_CLASSES)

        X_v, y_v, val_counts = subsample_per_class(X_v, y_v, args.val_per_class, args.val_seed, NUM_COMMON_CLASSES)
        print(f"\n▶ {src} ➔ {tgt} | val mỗi lớp thực tế: {val_counts} (tổng {len(y_v)}) | test: {len(y_ts)}")
        if min(val_counts) < args.val_per_class:
            print(f"   ⚠️ Có lớp ít hơn {args.val_per_class} mẫu val — val không cân bằng hoàn toàn.")

        val_loader = DataLoader(TensorDataset(X_v, y_v), batch_size=64, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_ts, y_ts), batch_size=64, shuffle=False)

        for k in args.k_shots:
            for mode in modes:
                for seed in args.seeds:
                    done += 1
                    out_path = root / f"{src}_to_{tgt}" / f"k{k}" / mode[2] / f"seed{seed}.json"
                    if out_path.exists():
                        print(f"[{done}/{total}] bỏ qua (đã có): {out_path.relative_to(root)}")
                        continue
                    hist = run_one(args, ckpt, X_tr, y_tr, val_loader, test_loader, val_counts,
                                   src, tgt, k, mode, seed, out_path)
                    best = max(range(args.epochs), key=lambda i: hist["val_f1"][i])
                    el = time.time() - t_start
                    print(f"[{done}/{total}] k={k:<3} {mode[2]:<18} seed={seed:<5} | "
                          f"F1@{min(40, args.epochs)}={hist['test_f1'][min(40, args.epochs) - 1]:5.2f} | "
                          f"val-best ep={best + 1:<3} -> test F1={hist['test_f1'][best]:5.2f} | "
                          f"đã chạy {el / 60:.1f} phút")

    print(f"\n✅ Xong. Kết quả JSON tại: {root}")


if __name__ == "__main__":
    main()

# =============================================================================
# CÁCH CHẠY
# =============================================================================
# Test nhanh (3 epoch, 1 run) — kiểm tra pipeline và thời gian mỗi run:
#   uv run python scripts/03_cross_domain/run_val_diagnostic.py \
#       --pairs motion_sense:uci_har --method tstcc --backbone standard --pretrain_tag bs64 \
#       --k_shots 10 --seeds 42 --freeze_modes none --epochs 3
#
# Chạy thật (ví dụ 3 cặp: dễ / trung bình / khó, cho từng phương pháp):
#   uv run python scripts/03_cross_domain/run_val_diagnostic.py \
#       --pairs motion_sense:uci_har uci_har:hhar_phone uci_har:hhar_watch \
#       --method tstcc --backbone standard --pretrain_tag bs64 \
#       --k_shots 10 30 100 --seeds 42 100 2024 --freeze_modes none full --epochs 60
# =============================================================================