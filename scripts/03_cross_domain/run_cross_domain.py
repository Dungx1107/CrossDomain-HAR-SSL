"""
===============================================================================
SCRIPT: FRACTION-BASED CROSS-DOMAIN ADAPTATION (CHUYÊN SÂU & TỐI ƯU LƯU TRỮ)
===============================================================================
- Dùng utils.sampling.sample_subset_by_ratio chuẩn của dự án để trích xuất Train.
- Dùng 100% Val đích để giám sát loss. Đánh giá trên 100% Test đích.
- Phân cấp thư mục kết quả: Fraction -> Protocol -> Seed
- Lấy kết quả trung bình, độ lệch chuẩn và CỘNG DỒN ma trận nhầm lẫn (Confusion Matrix).
- Xuất báo cáo dạng JSON chuyên sâu phục vụ viết báo khoa học (KHÔNG LƯU TRỌNG SỐ .PT).
===============================================================================
"""

import sys, json, argparse, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

# Cấu hình gốc dự án
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tự động chuyển đổi thư mục lưu trữ giữa Kaggle và Local
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
OUTPUT_ROOT = Path("/kaggle/working") if IS_KAGGLE else PROJECT_ROOT

from config.uci_har_config import UCIHARConfig
from config.motionsense_config import MotionSenseConfig
from config.hhar_config import HHARConfig
from utils.sampling import sample_subset_by_ratio
from engines.transfer.finetune_trainer import train_and_eval_finetune
from engines.evaluation.evaluator import ModelEvaluator
from models.encoders.builder import build_encoder

# ----------------- PARSER -----------------
parser = argparse.ArgumentParser(description="Fraction-based Cross-Domain HAR")
parser.add_argument("--method", type=str, default="contrastive", choices=["contrastive", "prototype", "crosshar"])
parser.add_argument("--backbone", type=str, default="cnn_transformer",
                    choices=["tstcc", "standard", "cnn_transformer", "vit_1d"])
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 7, 99])
parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10])
parser.add_argument("--pairs", nargs="+", type=str, required=True, help="e.g., motionsense:uci_har")
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


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds)


def run_fraction_experiment(src, tgt):
    print(f"\n{'=' * 70}\n🔄 FRACTION BENCHMARK: [{src.upper()}] ➔ [{tgt.upper()}]\n{'=' * 70}")
    cfg = DOMAIN_DATA_PATHS[tgt]

    X_tr_full, y_tr_full = load_data(cfg["train"])
    X_v_full, y_v_full = load_data(cfg["val"])
    X_ts_full, y_ts_full = load_data(cfg["test"])

    # Tự động ánh xạ phương pháp sang tiền tố của file checkpoint pretrain
    prefix_map = {"contrastive": "tstcc", "prototype": "prototype", "crosshar": "crosshar"}
    prefix = prefix_map.get(args.method.lower(), args.method.lower())

    source_ckpt = (
            OUTPUT_ROOT / "checkpoints" / "ssl_pretrain" / args.method / src / args.backbone /
            f"{prefix}_{args.backbone}_encoder_pretrained_{src}.pt"
    )

    if not source_ckpt.exists():
        raise FileNotFoundError(f"❌ Thiếu checkpoint pretrain tại: {source_ckpt}")

    test_loader = DataLoader(TensorDataset(X_ts_full, y_ts_full), batch_size=64, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_v_full, y_v_full), batch_size=64, shuffle=False)

    # Thư mục lưu kết quả evaluation được tách biệt khỏi pretrain
    base_save_dir = OUTPUT_ROOT / "outputs_evaluation" / "cross_fraction" / args.method / args.backbone / f"{src}_to_{tgt}"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))

    dummy_encoder = build_encoder(args.backbone, cfg["ch"])
    encoder_params = sum(p.numel() for p in dummy_encoder.parameters() if p.requires_grad)

    # Lưu Metadata thí nghiệm
    metadata = {
        "experiment_type": "fraction_based_cross_domain",
        "source_domain": src, "target_domain": tgt,
        "pretrain_method": args.method, "backbone": args.backbone,
        "pretrain_checkpoint_loaded": str(source_ckpt),
        "encoder_trainable_parameters": encoder_params,
        "fractions_tested": args.fractions, "seeds": args.seeds,
        "finetune_epochs": args.epochs, "batch_size": args.batch_size,
        "target_classes": COMMON_CLASS_NAMES
    }
    with open(base_save_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Vòng lặp các Fractions
    for frac in args.fractions:
        frac_name = f"{int(frac * 100)}_percent"

        # Vòng lặp các Protocols (Linear Probing / Full Finetuning)
        for proto in [{"name": "linear_probing", "frz": True}, {"name": "full_finetuning", "frz": False}]:
            print(f"\n▶ TIẾN TRÌNH: {frac_name.upper()} | {proto['name'].upper()}")
            save_dir = base_save_dir / frac_name / proto["name"]
            save_dir.mkdir(parents=True, exist_ok=True)

            # Khởi tạo các mảng để tính giá trị trung bình và cộng dồn ma trận
            acc_list, f1_list, prec_list, rec_list, kappa_list = [], [], [], [], []
            per_class_f1_lists = {cls: [] for cls in COMMON_CLASS_NAMES}
            seed_quick_lookup = {}
            aggregated_cm = None

            # Vòng lặp các Seeds
            for seed in args.seeds:
                torch.manual_seed(seed)
                seed_dir = save_dir / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                x_sub_tr, y_sub_tr = sample_subset_by_ratio(X_tr_full, y_tr_full, fraction=frac, seed=seed)

                tr_loader = DataLoader(TensorDataset(x_sub_tr, y_sub_tr),
                                       batch_size=min(args.batch_size, len(x_sub_tr)), shuffle=True)

                _, _, model, _ = train_and_eval_finetune(
                    train_loader=tr_loader, val_loader=val_loader, test_loader=test_loader,
                    encoder_checkpoint_path=source_ckpt, encoder=build_encoder(args.backbone, cfg["ch"]),
                    num_classes=NUM_COMMON_CLASSES, in_channels=cfg["ch"], epochs=args.epochs,
                    freeze_backbone=proto["frz"], device=DEVICE
                )

                y_true, y_pred = get_predictions(model, test_loader, DEVICE)
                report = classification_report(y_true, y_pred, target_names=COMMON_CLASS_NAMES, output_dict=True,
                                               zero_division=0)

                # Trích xuất các chỉ số
                acc = report["accuracy"] * 100
                f1 = report["macro avg"]["f1-score"] * 100
                prec = report["macro avg"]["precision"] * 100
                rec = report["macro avg"]["recall"] * 100
                kappa = cohen_kappa_score(y_true, y_pred)

                # Lưu vào danh sách để tính trung bình tổng
                acc_list.append(acc);
                f1_list.append(f1);
                prec_list.append(prec);
                rec_list.append(rec);
                kappa_list.append(kappa)
                seed_quick_lookup[str(seed)] = {"accuracy": round(acc, 2), "macro_f1": round(f1, 2)}

                # Trích xuất F1 cho từng class
                seed_per_class_f1 = {}
                for cls in COMMON_CLASS_NAMES:
                    cls_f1 = report[cls]["f1-score"] * 100
                    seed_per_class_f1[cls] = round(cls_f1, 2)
                    per_class_f1_lists[cls].append(cls_f1)

                # Tính và CỘNG DỒN ma trận nhầm lẫn
                current_cm = confusion_matrix(y_true, y_pred)
                if aggregated_cm is None:
                    aggregated_cm = current_cm.copy()
                else:
                    aggregated_cm += current_cm

                # Lưu số liệu của riêng seed này
                seed_metrics = {
                    "seed": seed,
                    "train_samples_used": len(y_sub_tr),
                    "metrics": {
                        "accuracy": round(acc, 2), "macro_f1": round(f1, 2),
                        "macro_precision": round(prec, 2), "macro_recall": round(rec, 2),
                        "cohen_kappa": round(kappa, 4)
                    },
                    "per_class_f1": seed_per_class_f1,
                    "confusion_matrix": current_cm.tolist(),
                    "finetune_data_subset": {
                        "samples": x_sub_tr.cpu().tolist(),
                        "labels": y_sub_tr.cpu().tolist()
                    }
                }
                with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(seed_metrics, f, indent=4)

                # Đánh giá & vẽ ma trận nhầm lẫn bằng hình ảnh cho seed này
                evaluator.evaluate(model, test_loader, plot_save_path=str(seed_dir / "confusion_matrix.png"))
                print(f"   [Seed {seed:<4}] Mẫu: {len(y_sub_tr):<4} | Acc: {acc:5.2f}% | F1: {f1:5.2f}%")

            # Xây dựng báo cáo tổng hợp (Aggregated Report) sau khi chạy xong các seeds
            summary_filename = f"summary_{proto['name']}_{frac_name}.json"
            aggregate_data = {
                "protocol": proto["name"],
                "fraction_name": frac_name,
                "fraction_value": frac,
                "num_seeds_tested": len(args.seeds),
                "overall_metrics": {
                    "accuracy_mean": round(np.mean(acc_list), 2), "accuracy_std": round(np.std(acc_list), 2),
                    "macro_f1_mean": round(np.mean(f1_list), 2), "macro_f1_std": round(np.std(f1_list), 2),
                    "macro_precision_mean": round(np.mean(prec_list), 2),
                    "macro_recall_mean": round(np.mean(rec_list), 2),
                    "cohen_kappa_mean": round(np.mean(kappa_list), 4)
                },
                "per_class_f1_mean": {cls: round(np.mean(lst), 2) for cls, lst in per_class_f1_lists.items()},
                "seed_quick_lookup": seed_quick_lookup,
                "aggregated_confusion_matrix": aggregated_cm.tolist()
            }
            with open(save_dir / summary_filename, "w", encoding="utf-8") as f:
                json.dump(aggregate_data, f, indent=4)

            print(
                f"✅ Đã lưu file tổng -> {summary_filename}: F1 Mean = {aggregate_data['overall_metrics']['macro_f1_mean']}%")


if __name__ == "__main__":
    for p in args.pairs:
        src, tgt = p.split(":")
        run_fraction_experiment(src.strip(), tgt.strip())
