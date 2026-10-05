"""
===============================================================================
SCRIPT: FEW-SHOT CROSS-DOMAIN ADAPTATION (CHUYÊN SÂU & TỐI ƯU LƯU TRỮ)
- Không lưu file trọng số finetune (tránh rác bộ nhớ).
- Tính toán đầy đủ Precision, Recall, F1, Cohen's Kappa, Per-class.
- Ghi nhận siêu dữ liệu (Metadata): số tham số, cấu hình epoch, batch size.
===============================================================================
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from config.hhar_config import HHARConfig
from engines.transfer.finetune_trainer import train_and_eval_finetune
from engines.evaluation.evaluator import ModelEvaluator
from models.encoders.builder import build_encoder

parser = argparse.ArgumentParser(description="Few-shot Cross-Domain HAR")
parser.add_argument("--method", type=str, default="crosshar")
parser.add_argument("--backbone", type=str, default="cnn_transformer")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 7, 99])
parser.add_argument("--k_shots", nargs="+", type=int, default=[1, 5, 10, 20])
parser.add_argument("--pairs", nargs="+", type=str, required=True)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)

DOMAIN_DATA_PATHS = {
    "motionsense": {"train": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
                    "val": Path(MotionSenseConfig.PROCESSED_VAL_PATH),
                    "test": Path(MotionSenseConfig.PROCESSED_TEST_PATH), "ch": 6},
    "uci_har": {"train": Path(UCIHARConfig.PROCESSED_TRAIN_PATH), "val": Path(UCIHARConfig.PROCESSED_VAL_PATH),
                "test": Path(UCIHARConfig.PROCESSED_TEST_PATH), "ch": 6},
    "hhar_phone": {"train": HHARConfig.PROCESSED_DIR_PHONE / "train.pt",
                   "val": HHARConfig.PROCESSED_DIR_PHONE / "val.pt", "test": HHARConfig.PROCESSED_DIR_PHONE / "test.pt",
                   "ch": 6},
    "hhar_watch": {"train": HHARConfig.PROCESSED_DIR_WATCH / "train.pt",
                   "val": HHARConfig.PROCESSED_DIR_WATCH / "val.pt", "test": HHARConfig.PROCESSED_DIR_WATCH / "test.pt",
                   "ch": 6},
}


def load_data(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    mask = (d["labels"].squeeze() >= 0) & (d["labels"].squeeze() < 5)
    s, l = d["samples"][mask].float(), d["labels"].squeeze()[mask].long()
    if s.ndim == 3 and s.shape[1] == 128 and s.shape[2] == 6: s = s.permute(0, 2, 1)
    return s, l


def sample_k_shot(X, y, k, seed):
    rng = np.random.default_rng(seed)
    idx_list = []
    for c in range(NUM_COMMON_CLASSES):
        c_idx = torch.where(y == c)[0].numpy()
        if len(c_idx) < k: raise ValueError(f"Không đủ mẫu cho lớp {c}. Yêu cầu {k}, chỉ có {len(c_idx)}")
        rng.shuffle(c_idx)
        idx_list.extend(c_idx[:k])
    return X[idx_list], y[idx_list]


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


def run_few_shot(src, tgt):
    print(f"\n{'=' * 70}\n🔄 FEW-SHOT: [{src.upper()}] ➔ [{tgt.upper()}]\n{'=' * 70}")
    cfg = DOMAIN_DATA_PATHS[tgt]

    X_tr, y_tr = load_data(cfg["train"])
    X_v, y_v = load_data(cfg["val"])
    X_ts, y_ts = load_data(cfg["test"])

    source_ckpt = PROJECT_ROOT / "checkpoints/ssl_pretrain/crosshar" / src / args.backbone / f"crosshar_{args.backbone}_encoder_pretrained_{src}.pt"
    if not source_ckpt.exists(): raise FileNotFoundError(f"❌ Thiếu checkpoint pretrain: {source_ckpt}")

    test_loader = DataLoader(TensorDataset(X_ts, y_ts), batch_size=64, shuffle=False)
    base_save_dir = PROJECT_ROOT / "checkpoints/cross_domain_fewshot/crosshar" / args.backbone / f"{src}_to_{tgt}"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))

    # Đếm số lượng tham số của mô hình Encoder
    dummy_encoder = build_encoder(args.backbone, cfg["ch"])
    encoder_params = sum(p.numel() for p in dummy_encoder.parameters() if p.requires_grad)

    # 1. Ghi Meta-data file (Cấu hình tổng quan của cặp chạy này)
    metadata = {
        "experiment_type": "few_shot_cross_domain",
        "source_domain": src,
        "target_domain": tgt,
        "pretrain_method": args.method,
        "backbone": args.backbone,
        "pretrain_checkpoint_loaded": str(source_ckpt),
        "encoder_trainable_parameters": encoder_params,
        "k_shots_tested": args.k_shots,
        "seeds": args.seeds,
        "finetune_epochs": args.epochs,
        "batch_size": args.batch_size,
        "target_classes": COMMON_CLASS_NAMES
    }
    with open(base_save_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    for k in args.k_shots:
        for proto in [{"name": "linear_probing", "frz": True}, {"name": "full_finetuning", "frz": False}]:
            print(f"\n▶ TIẾN TRÌNH: {k}-SHOT | {proto['name'].upper()}")

            k_save_dir = base_save_dir / f"{k}_shot" / proto["name"]
            k_save_dir.mkdir(parents=True, exist_ok=True)

            f1_list, acc_list = [], []
            best_f1, best_model = -1.0, None
            detailed_results = {}

            for seed in args.seeds:
                torch.manual_seed(seed)
                x_sub_tr, y_sub_tr = sample_k_shot(X_tr, y_tr, k, seed)
                x_sub_val, y_sub_val = sample_k_shot(X_v, y_v, k, seed)

                tr_loader = DataLoader(TensorDataset(x_sub_tr, y_sub_tr),
                                       batch_size=min(args.batch_size, len(x_sub_tr)), shuffle=True)
                val_loader = DataLoader(TensorDataset(x_sub_val, y_sub_val),
                                        batch_size=min(args.batch_size, len(x_sub_val)), shuffle=False)

                _, _, model, _ = train_and_eval_finetune(
                    train_loader=tr_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    encoder_checkpoint_path=source_ckpt,
                    encoder=build_encoder(args.backbone, cfg["ch"]),
                    num_classes=NUM_COMMON_CLASSES,
                    in_channels=cfg["ch"],
                    epochs=args.epochs,
                    freeze_backbone=proto["frz"],
                    device=DEVICE
                )

                # Suy diễn để lấy dự đoán thực tế tính metrics chuyên sâu
                y_true, y_pred = get_predictions(model, test_loader, DEVICE)

                report = classification_report(y_true, y_pred, target_names=COMMON_CLASS_NAMES, output_dict=True,
                                               zero_division=0)
                kappa = cohen_kappa_score(y_true, y_pred)

                acc = report["accuracy"] * 100
                f1 = report["macro avg"]["f1-score"] * 100

                f1_list.append(f1);
                acc_list.append(acc)

                detailed_results[f"seed_{seed}"] = {
                    "accuracy": round(acc, 2),
                    "macro_f1": round(f1, 2),
                    "macro_precision": round(report["macro avg"]["precision"] * 100, 2),
                    "macro_recall": round(report["macro avg"]["recall"] * 100, 2),
                    "cohen_kappa": round(kappa, 4),
                    "per_class_f1": {cls: round(report[cls]["f1-score"] * 100, 2) for cls in COMMON_CLASS_NAMES},
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
                }

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model

                print(f"   [Seed {seed:<4}] Acc: {acc:5.2f}% | F1: {f1:5.2f}%")

            # Chỉ xuất hình ảnh Confusion Matrix của lần chạy tốt nhất (KHÔNG lưu mô hình .pt)
            if best_model:
                evaluator.evaluate(best_model, test_loader,
                                   plot_save_path=str(k_save_dir / "best_confusion_matrix.png"))

            # Ghi file summary rút gọn
            summary_data = {
                "accuracy": f"{np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}",
                "macro_f1": f"{np.mean(f1_list):.2f} ± {np.std(f1_list):.2f}"
            }
            with open(k_save_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)

            # Ghi file chi tiết chứa Precision/Recall/Kappa
            with open(k_save_dir / "detailed_results.json", "w", encoding="utf-8") as f:
                json.dump(detailed_results, f, indent=4)

            print(f"✅ Đã lưu -> Tổng kết {k}-SHOT | {proto['name']}: F1 = {summary_data['macro_f1']}%")


if __name__ == "__main__":
    for p in args.pairs:
        src, tgt = p.split(":")
        run_few_shot(src.strip(), tgt.strip())