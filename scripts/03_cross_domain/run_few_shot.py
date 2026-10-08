"""
===============================================================================
SCRIPT: FEW-SHOT CROSS-DOMAIN ADAPTATION (CHUYÊN SÂU & SIÊU TỐI ƯU LƯU TRỮ)
- Đánh giá trên tập Val và Test đầy đủ (không lấy mẫu K-shot cho Val/Test).
- Không tạo thư mục con cho từng Seed.
- Gom toàn bộ chi tiết các seed vào 1 file `detailed_all_seeds.json`.
- Ghi 1 file `summary_...json` tính trung bình cộng để làm báo cáo.
- Vẽ và lưu 1 ảnh `aggregated_confusion_matrix.png` (trung bình qua các seeds).
===============================================================================
"""

import sys, json, argparse, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from config.hhar_config import HHARConfig
from engines.transfer.finetune_trainer import train_and_eval_finetune
from models.encoders.builder import build_encoder

parser = argparse.ArgumentParser(description="Few-shot Cross-Domain HAR")
parser.add_argument("--method", type=str, default="contrastive", choices=["contrastive", "prototype", "crosshar"])
parser.add_argument("--backbone", type=str, default="cnn_transformer", choices=["standard", "cnn_transformer", "vit_1d"])
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 7, 99])
parser.add_argument("--k_shots", nargs="+", type=int, default=[1, 5, 10, 20])
parser.add_argument("--pairs", nargs="+", type=str, required=True, help="e.g., motionsense:uci_har")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)

DOMAIN_DATA_PATHS = {
    "motionsense": {"train": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH), "val": Path(MotionSenseConfig.PROCESSED_VAL_PATH), "test": Path(MotionSenseConfig.PROCESSED_TEST_PATH), "ch": 6},
    "uci_har": {"train": Path(UCIHARConfig.PROCESSED_TRAIN_PATH), "val": Path(UCIHARConfig.PROCESSED_VAL_PATH), "test": Path(UCIHARConfig.PROCESSED_TEST_PATH), "ch": 6},
    "hhar_phone": {"train": HHARConfig.PROCESSED_DIR_PHONE / "train.pt", "val": HHARConfig.PROCESSED_DIR_PHONE / "val.pt", "test": HHARConfig.PROCESSED_DIR_PHONE / "test.pt", "ch": 6},
    "hhar_watch": {"train": HHARConfig.PROCESSED_DIR_WATCH / "train.pt", "val": HHARConfig.PROCESSED_DIR_WATCH / "val.pt", "test": HHARConfig.PROCESSED_DIR_WATCH / "test.pt", "ch": 6},
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
        if len(c_idx) < k: raise ValueError(f"Không đủ mẫu cho lớp {c}. Yêu cầu {k}, có {len(c_idx)}")
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
    print(f"\n{'=' * 70}\n🔄 FEW-SHOT BENCHMARK: [{src.upper()}] ➔ [{tgt.upper()}]\n{'=' * 70}")
    cfg = DOMAIN_DATA_PATHS[tgt]

    X_tr_full, y_tr_full = load_data(cfg["train"])
    X_v_full, y_v_full = load_data(cfg["val"])
    X_ts_full, y_ts_full = load_data(cfg["test"])

    prefix_map = {"contrastive": "tstcc", "prototype": "prototype", "crosshar": "crosshar"}
    prefix = prefix_map.get(args.method.lower(), args.method.lower())
    source_ckpt = (OUTPUT_ROOT / "checkpoints/ssl_pretrain" / args.method / src / args.backbone / f"{prefix}_{args.backbone}_encoder_pretrained_{src}.pt")
    if not source_ckpt.exists(): raise FileNotFoundError(f"❌ Thiếu checkpoint pretrain tại: {source_ckpt}")

    val_loader = DataLoader(TensorDataset(X_v_full, y_v_full), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_ts_full, y_ts_full), batch_size=64, shuffle=False)

    base_save_dir = OUTPUT_ROOT / "outputs_evaluation/cross_k_shot" / args.method / args.backbone / f"{src}_to_{tgt}"
    base_save_dir.mkdir(parents=True, exist_ok=True)

    dummy_encoder = build_encoder(args.backbone, cfg["ch"])
    encoder_params = sum(p.numel() for p in dummy_encoder.parameters() if p.requires_grad)

    metadata = {
        "experiment_type": "few_shot_cross_domain", "source_domain": src, "target_domain": tgt,
        "pretrain_method": args.method, "backbone": args.backbone, "checkpoint": str(source_ckpt),
        "k_shots": args.k_shots, "seeds": args.seeds, "epochs": args.epochs, "batch_size": args.batch_size
    }
    with open(base_save_dir / "experiment_metadata.json", "w", encoding="utf-8") as f: json.dump(metadata, f, indent=4)

    for k in args.k_shots:
        k_shot_name = f"{k}_shot"
        for proto in [{"name": "linear_probing", "frz": True}, {"name": "full_finetuning", "frz": False}]:
            print(f"\n▶ TIẾN TRÌNH: {k_shot_name.upper()} | {proto['name'].upper()}")
            save_dir = base_save_dir / k_shot_name / proto["name"]
            save_dir.mkdir(parents=True, exist_ok=True)

            acc_list, f1_list, prec_list, rec_list, kappa_list = [], [], [], [], []
            per_class_f1_lists = {cls: [] for cls in COMMON_CLASS_NAMES}
            seed_quick_lookup, all_seeds_details = {}, {}
            aggregated_cm = None

            for seed in args.seeds:
                torch.manual_seed(seed)
                x_sub_tr, y_sub_tr = sample_k_shot(X_tr_full, y_tr_full, k, seed)
                tr_loader = DataLoader(TensorDataset(x_sub_tr, y_sub_tr), batch_size=min(args.batch_size, len(x_sub_tr)), shuffle=True)

                _, _, model, _ = train_and_eval_finetune(
                    train_loader=tr_loader, val_loader=val_loader, test_loader=test_loader,
                    encoder_checkpoint_path=source_ckpt, encoder=build_encoder(args.backbone, cfg["ch"]),
                    num_classes=NUM_COMMON_CLASSES, in_channels=cfg["ch"], epochs=args.epochs,
                    freeze_backbone=proto["frz"], device=DEVICE
                )

                y_true, y_pred = get_predictions(model, test_loader, DEVICE)
                report = classification_report(y_true, y_pred, target_names=COMMON_CLASS_NAMES, output_dict=True, zero_division=0)

                acc = report["accuracy"] * 100
                f1 = report["macro avg"]["f1-score"] * 100
                prec = report["macro avg"]["precision"] * 100
                rec = report["macro avg"]["recall"] * 100
                kappa = cohen_kappa_score(y_true, y_pred)

                acc_list.append(acc); f1_list.append(f1); prec_list.append(prec); rec_list.append(rec); kappa_list.append(kappa)
                seed_quick_lookup[str(seed)] = {"accuracy": round(acc, 2), "macro_f1": round(f1, 2)}

                seed_per_class_f1 = {cls: round(report[cls]["f1-score"] * 100, 2) for cls in COMMON_CLASS_NAMES}
                for cls in COMMON_CLASS_NAMES: per_class_f1_lists[cls].append(seed_per_class_f1[cls])

                current_cm = confusion_matrix(y_true, y_pred)
                aggregated_cm = current_cm.copy() if aggregated_cm is None else aggregated_cm + current_cm

                all_seeds_details[f"seed_{seed}"] = {
                    "train_samples_used": len(y_sub_tr),
                    "metrics": {"accuracy": round(acc, 2), "macro_f1": round(f1, 2), "cohen_kappa": round(kappa, 4)},
                    "per_class_f1": seed_per_class_f1,
                    "confusion_matrix": current_cm.tolist()
                }

                print(f"   [Seed {seed:<4}] Mẫu Train: {len(y_sub_tr):<4} | Acc: {acc:5.2f}% | F1: {f1:5.2f}%")

            # 1. Ghi tệp toàn bộ chi tiết của tất cả các Seed
            with open(save_dir / "detailed_all_seeds.json", "w", encoding="utf-8") as f:
                json.dump(all_seeds_details, f, indent=4)

            # 2. Ghi tệp Summary trung bình
            summary_filename = f"summary_{proto['name']}_{k_shot_name}.json"
            aggregate_data = {
                "protocol": proto["name"], "k_shot_name": k_shot_name, "k_value": k, "num_seeds_tested": len(args.seeds),
                "overall_metrics": {
                    "accuracy_mean": round(np.mean(acc_list), 2), "accuracy_std": round(np.std(acc_list), 2),
                    "macro_f1_mean": round(np.mean(f1_list), 2), "macro_f1_std": round(np.std(f1_list), 2)
                },
                "per_class_f1_mean": {cls: round(np.mean(lst), 2) for cls, lst in per_class_f1_lists.items()},
                "seed_quick_lookup": seed_quick_lookup,
                "aggregated_confusion_matrix": aggregated_cm.tolist()
            }
            with open(save_dir / summary_filename, "w", encoding="utf-8") as f:
                json.dump(aggregate_data, f, indent=4)

            # 3. Tính toán và vẽ Aggregated Confusion Matrix
            avg_cm = aggregated_cm / len(args.seeds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', xticklabels=COMMON_CLASS_NAMES, yticklabels=COMMON_CLASS_NAMES)
            plt.title(f"Average Confusion Matrix ({k_shot_name} | {proto['name']})", fontsize=12, fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            cm_save_path = save_dir / "aggregated_confusion_matrix.png"
            plt.savefig(str(cm_save_path), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Đã lưu cấu trúc gọn -> {k_shot_name} | {proto['name']}: F1 Mean = {aggregate_data['overall_metrics']['macro_f1_mean']}%")

if __name__ == "__main__":
    for p in args.pairs:
        src, tgt = p.split(":")
        run_few_shot(src.strip(), tgt.strip())