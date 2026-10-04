"""
===============================================================================
SCRIPT: FEW-SHOT CROSS-DOMAIN ADAPTATION
- k <= 5: Linear Probing
- k >= 10: Full Fine-Tuning
- Support (Train) & Query (Val) được rút ngẫu nhiên k mẫu mỗi lớp từ train.pt
- Test Set giữ nguyên 100% từ test.pt
- Tự động tách file results.json riêng cho từng k-shot và merge an toàn vào file tổng.
===============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

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

# CẤU HÌNH CƠ BẢN
DEFAULT_TRANSFER_PAIRS = [
    # 1. Cặp nội bộ Phone ↔ Phone (Cùng vị trí đeo túi/thắt lưng)
    ("uci_har", "motionsense"),
    ("motionsense", "uci_har"),

    # 2. Cặp nội bộ HHAR Phone ↔ Watch (Chéo vị trí)
    ("hhar_phone", "hhar_watch"),
    ("hhar_watch", "hhar_phone"),

    # 3. Chuyển giao từ Phone sang Watch (Cross-position)
    ("motionsense", "hhar_watch"),
    ("uci_har", "hhar_watch"),

    # 4. Chuyển giao giữa các thiết bị Phone khác bộ dữ liệu (Same-position)
    ("motionsense", "hhar_phone"),
    ("uci_har", "hhar_phone"),
    ("hhar_phone", "motionsense"),
    ("hhar_phone", "uci_har"),

    # 5. Chuyển giao từ Watch sang Phone khác bộ dữ liệu (Cross-position)
    ("hhar_watch", "motionsense"),
    ("hhar_watch", "uci_har"),
]

parser = argparse.ArgumentParser(description="Few-shot Cross-Domain HAR Benchmark")
parser.add_argument("--method", type=str, default="crosshar", choices=["tstcc", "prototype", "crosshar"])
parser.add_argument("--backbone", type=str, default="standard")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2024, 7, 99])
parser.add_argument("--k_shots", nargs="+", type=int, default=[1, 5, 10, 20, 30],
                    help="Danh sách giá trị k (vd: 1 5 10 20 30)")
parser.add_argument("--pairs", nargs="+", type=str, default=None)
args = parser.parse_args()

SEEDS = args.seeds
K_SHOTS = args.k_shots
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

if args.pairs:
    SELECTED_PAIRS = [tuple(p.split(":")) for p in args.pairs]
else:
    SELECTED_PAIRS = DEFAULT_TRANSFER_PAIRS

METHOD_TO_FOLDER = {"tstcc": "contrastive", "prototype": "prototype", "crosshar": "crosshar"}
COMMON_CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
NUM_COMMON_CLASSES = len(COMMON_CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROTOCOLS = [
    {"name": "linear_probing", "freeze_backbone": True},
    {"name": "full_finetuning", "freeze_backbone": False}
]

DOMAIN_DATA_PATHS = {
    "motionsense": {"train_path": Path(MotionSenseConfig.PROCESSED_TRAIN_PATH),
                    "test_path": Path(MotionSenseConfig.PROCESSED_TEST_PATH),
                    "in_channels": int(MotionSenseConfig.IN_CHANNELS)},
    "uci_har": {"train_path": Path(UCIHARConfig.PROCESSED_TRAIN_PATH),
                "test_path": Path(UCIHARConfig.PROCESSED_TEST_PATH),
                "in_channels": int(UCIHARConfig.IN_CHANNELS)},
    "hhar_phone": {"train_path": HHARConfig.PROCESSED_DIR_PHONE / "train.pt",
                   "test_path": HHARConfig.PROCESSED_DIR_PHONE / "test.pt",
                   "in_channels": int(HHARConfig.IN_CHANNELS)},
    "hhar_watch": {"train_path": HHARConfig.PROCESSED_DIR_WATCH / "train.pt",
                   "test_path": HHARConfig.PROCESSED_DIR_WATCH / "test.pt",
                   "in_channels": int(HHARConfig.IN_CHANNELS)},
}


def load_and_prepare_target_data(domain_name: str):
    cfg = DOMAIN_DATA_PATHS[domain_name]
    raw_train = torch.load(cfg["train_path"], map_location="cpu", weights_only=True)
    raw_test = torch.load(cfg["test_path"], map_location="cpu", weights_only=True)

    def process_tensor(samples, labels):
        mask = (labels >= 0) & (labels < 5)
        s, l = samples[mask].float(), labels[mask].long()
        if s.ndim == 3 and s.shape[1] == 128 and s.shape[2] == 6:
            s = s.permute(0, 2, 1)
        return s, l

    x_pool, y_pool = process_tensor(raw_train["samples"], raw_train["labels"])
    x_test, y_test = process_tensor(raw_test["samples"], raw_test["labels"])

    return x_pool, y_pool, x_test, y_test, cfg["in_channels"]


def sample_k_shot_train_val(x_pool, y_pool, k, seed):
    """Trích xuất đúng k mẫu cho Train và k mẫu khác cho Val từ Target Pool."""
    rng = np.random.default_rng(seed)
    classes = torch.unique(y_pool).tolist()

    train_idx, val_idx = [], []
    for c in classes:
        c_idx = torch.where(y_pool == c)[0].numpy()
        rng.shuffle(c_idx)

        if len(c_idx) < 2 * k:
            raise ValueError(f"Không đủ mẫu cho lớp {c}. Cần {2 * k}, chỉ có {len(c_idx)}")

        train_idx.extend(c_idx[:k])
        val_idx.extend(c_idx[k:2 * k])

    return x_pool[train_idx], y_pool[train_idx], x_pool[val_idx], y_pool[val_idx]


def run_few_shot_for_pair(source_domain: str, target_domain: str):
    print(f"\n{'=' * 90}\n🔄 FEW-SHOT: [{source_domain.upper()}] ➔ [{target_domain.upper()}]\n{'=' * 90}")
    ssl_folder = METHOD_TO_FOLDER.get(args.method, args.method)
    source_ckpt = (PROJECT_ROOT / "checkpoints/ssl_pretrain" / ssl_folder /
                   source_domain / args.backbone /
                   f"{args.method}_{args.backbone}_encoder_pretrained_{source_domain}.pt")

    x_pool, y_pool, x_test, y_test, in_channels = load_and_prepare_target_data(target_domain)

    base_save_dir = (PROJECT_ROOT / "checkpoints" / "cross_domain_fewshot" / args.method /
                     args.backbone / f"{source_domain}_to_{target_domain}")
    base_save_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ModelEvaluator(class_names=COMMON_CLASS_NAMES, device=torch.device(DEVICE))
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    summary = {}

    for k in K_SHOTS:
        summary[f"{k}_shot"] = {}
        for proto in PROTOCOLS:
            proto_name = proto["name"]
            freeze_bb = proto["freeze_backbone"]

            print(f"\n▶ {k}-SHOT | CHIẾN LƯỢC: {proto_name.upper()} | Freeze: {freeze_bb}")

            k_save_dir = base_save_dir / f"{k}_shot" / proto_name
            k_save_dir.mkdir(parents=True, exist_ok=True)

            f1_list, acc_list = [], []
            best_run_f1 = -1.0
            best_run_model = None

            for seed in SEEDS:
                torch.manual_seed(seed)

                x_train, y_train, x_val, y_val = sample_k_shot_train_val(x_pool, y_pool, k, seed)

                train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=min(BATCH_SIZE, len(x_train)),
                                          shuffle=True)
                val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=min(BATCH_SIZE, len(x_val)),
                                        shuffle=False)

                acc, f1, model, _ = train_and_eval_finetune(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    encoder_checkpoint_path=source_ckpt,
                    encoder=build_encoder(backbone_type=args.backbone, in_channels=in_channels),
                    num_classes=NUM_COMMON_CLASSES,
                    in_channels=in_channels,
                    epochs=EPOCHS,
                    freeze_backbone=freeze_bb,
                    device=DEVICE
                )

                f1_list.append(f1)
                acc_list.append(acc)

                if f1 > best_run_f1:
                    best_run_f1 = f1
                    best_run_model = model

                print(f"   [Seed {seed:4d}] -> Acc: {acc:5.2f}% | F1: {f1:5.2f}%")

            mean_f1, std_f1 = float(np.mean(f1_list)), float(np.std(f1_list))
            mean_acc, std_acc = float(np.mean(acc_list)), float(np.std(acc_list))

            if best_run_model:
                torch.save(best_run_model.state_dict(), k_save_dir / "best_model.pt")
                evaluator.evaluate(best_run_model, test_loader, plot_save_path=str(k_save_dir / "confusion_matrix.png"))

            summary[f"{k}_shot"][proto_name] = {
                "macro_f1": f"{mean_f1:.2f} ± {std_f1:.2f}",
                "accuracy": f"{mean_acc:.2f} ± {std_acc:.2f}"
            }
            print(f"⭐ TỔNG KẾT {k}-SHOT: F1 = {mean_f1:.2f} ± {std_f1:.2f}% | Acc = {mean_acc:.2f} ± {std_acc:.2f}%")

            # 1. LƯU RIÊNG TỪNG K-SHOT NGAY SAU MỖI CHIẾN LƯỢC (Chống mất dữ liệu và chống ghi đè)
            k_shot_file = base_save_dir / f"{k}_shot" / "results.json"
            k_shot_data = {}
            if k_shot_file.exists():
                try:
                    with open(k_shot_file, "r", encoding="utf-8") as f:
                        k_shot_data = json.load(f)
                except Exception:
                    k_shot_data = {}
            k_shot_data[proto_name] = summary[f"{k}_shot"][proto_name]
            with open(k_shot_file, "w", encoding="utf-8") as f:
                json.dump(k_shot_data, f, indent=4)

    # 2. HỢP NHẤT TẤT CẢ VÀO FILE TỔNG (few_shot_results.json)
    results_json_path = base_save_dir / "few_shot_results.json"
    existing_summary = {}
    if results_json_path.exists():
        try:
            with open(results_json_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
        except Exception:
            existing_summary = {}

    for k_key, v_dict in summary.items():
        if k_key not in existing_summary:
            existing_summary[k_key] = {}
        existing_summary[k_key].update(v_dict)

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_summary, f, indent=4)


if __name__ == "__main__":
    for src, tgt in SELECTED_PAIRS:
        run_few_shot_for_pair(src, tgt)