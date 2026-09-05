"""
===============================================================================
SCRIPT: STANDALONE BASELINE EVALUATION
===============================================================================
Mục đích:
    - Nạp Checkpoint đã lưu để kiểm định độc lập trên tập Test (không cần train lại).
    - Tự động nạp đúng cấu hình kênh/lớp của MotionSense hoặc UCI-HAR.
    - Đo độ phức tạp tính toán (Params, FLOPs, Latency).
    - Xuất ma trận nhầm lẫn và báo cáo tổng kết vào thư mục standalone_eval.
===============================================================================
"""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from datasets.base_dataset import get_har_dataloaders
from models.encoders.cnn1d import StandardSensorEncoder1D
from models.heads.classifier import ClassifierHead
from models.har_classifier import HARClassifier
from engines.evaluation.evaluator import ModelEvaluator
from utils.complexity import measure_model_complexity, print_complexity_report


# =============================================================================
# KHỐI CẤU HÌNH ĐÁNH GIÁ (CHỈNH TRỰC TIẾP TẠI ĐÂY RỒI BẤM RUN)
# =============================================================================
@dataclass
class EvalConfig:
    dataset: str = "motionsense"           # 'motionsense' hoặc 'uci_har'
    checkpoint_path: Optional[str] = None  # None: tự tìm checkpoint mặc định
    feature_dim: int = 128
    batch_size: int = 64
    device: Optional[str] = None           # None: tự nhận diện cuda/cpu


CONFIG = EvalConfig(
    dataset="motionsense",
    checkpoint_path=None,                  # Hoặc điền đường dẫn cụ thể nếu muốn
    batch_size=64
)
# =============================================================================


@dataclass
class DatasetMetadata:
    data_dir: Path
    in_channels: int
    num_classes: int
    class_names: List[str]


def resolve_dataset_meta(dataset_name: str) -> DatasetMetadata:
    """Trích xuất thông số metadata và thư mục dữ liệu chuẩn từ file Config."""
    dataset_name = dataset_name.lower().strip()
    if dataset_name == "motionsense":
        return DatasetMetadata(
            data_dir=Path(MotionSenseConfig.PROCESSED_DIR),
            in_channels=int(MotionSenseConfig.IN_CHANNELS),
            num_classes=int(MotionSenseConfig.NUM_CLASSES),
            class_names=list(MotionSenseConfig.CLASS_NAMES)
        )
    elif dataset_name in ["uci_har", "ucihar"]:
        return DatasetMetadata(
            data_dir=Path(UCIHARConfig.DATA_DIR),
            in_channels=int(UCIHARConfig.IN_CHANNELS),
            num_classes=int(UCIHARConfig.NUM_CLASSES),
            class_names=list(UCIHARConfig.CLASS_NAMES)
        )
    else:
        raise ValueError(f"❌ Bộ dữ liệu '{dataset_name}' chưa được hỗ trợ!")


def main():
    device = torch.device(CONFIG.device) if CONFIG.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = CONFIG.dataset.lower().strip()
    meta = resolve_dataset_meta(dataset_name)

    # 1. Xác định checkpoint và thư mục xuất báo cáo
    if CONFIG.checkpoint_path is None:
        ckpt_path = PROJECT_ROOT / "experiments" / "baseline" / dataset_name / "checkpoint" / "best_baseline_model.pt"
    else:
        ckpt_path = Path(CONFIG.checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy Checkpoint tại: {ckpt_path}")

    eval_output_dir = PROJECT_ROOT / "experiments" / "baseline" / dataset_name / "standalone_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🔍 BẮT ĐẦU ĐÁNH GIÁ ĐỘC LẬP TỪ CHECKPOINT: {dataset_name.upper()}")
    print(f"💾 Checkpoint nạp vào: {ckpt_path}")
    print(f"💻 Thiết bị          : {device}")
    print("=" * 80)

    # 2. Nạp dữ liệu Test
    _, _, test_loader = get_har_dataloaders(
        data_dir=meta.data_dir,
        batch_size=CONFIG.batch_size
    )

    # 3. Khởi tạo mô hình đúng kích thước kênh/lớp và nạp trọng số
    encoder = StandardSensorEncoder1D(in_channels=meta.in_channels, feature_dim=CONFIG.feature_dim)
    head = ClassifierHead(feature_dim=CONFIG.feature_dim, num_classes=meta.num_classes)
    model = HARClassifier(encoder=encoder, classifier=head).to(device)

    checkpoint_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_dict, strict=False)
    print("✅ Đã nạp thành công trọng số từ Checkpoint.")

    # 4. Đo độ phức tạp tính toán
    input_shape = (1, meta.in_channels, 128)
    complexity_info = measure_model_complexity(model, input_size=input_shape, device=device)
    print_complexity_report(complexity_info)
    with open(eval_output_dir / "eval_complexity.json", "w", encoding="utf-8") as f:
        json.dump(complexity_info, f, indent=4)

    # 5. Kích hoạt ModelEvaluator Engine
    evaluator = ModelEvaluator(class_names=meta.class_names, device=device)
    cm_path = str(eval_output_dir / "eval_confusion_matrix.png")

    test_metrics = evaluator.evaluate(
        model=model,
        test_loader=test_loader,
        plot_save_path=cm_path,
        title_prefix=f"Standalone {dataset_name.upper()}"
    )

    # 6. Xuất báo cáo JSON
    summary_report = {
        "dataset": dataset_name,
        "checkpoint_evaluated": str(ckpt_path),
        "test_accuracy": float(test_metrics["accuracy"] * 100),
        "test_macro_f1": float(test_metrics["macro_f1"] * 100),
        "test_weighted_f1": float(test_metrics["weighted_f1"] * 100)
    }
    with open(eval_output_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4)

    print(f"✅ Hoàn tất đánh giá độc lập! Kết quả lưu tại: {eval_output_dir}")


if __name__ == "__main__":
    main()