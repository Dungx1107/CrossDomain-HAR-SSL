"""
===============================================================================
SCRIPT RUNNER: scripts/baseline/evaluate_baseline.py (EVALUATE STANDALONE)
===============================================================================
Mục đích:
    - Nạp checkpoint tốt nhất đã lưu để đánh giá trên Test Set (Subjects 19-24).
    - Đo lường độ phức tạp tính toán (Params, FLOPs, Latency) của mô hình.
    - Không cần huấn luyện lại, lưu log và confusion matrix vào experiments/.
===============================================================================
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.loader import get_motionsense_dataloaders
from models.baseline.supervised_model import SupervisedHARModel
from training.evaluator import ModelEvaluator
from utils.logger import ExperimentTracker
from utils.complexity import measure_model_complexity, print_complexity_report

CLASS_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]


def main():
    tracker = ExperimentTracker(
        exp_name="eval_standalone_baseline",
        config_obj=MotionSenseConfig,
        notes="Đánh giá độc lập trên Test Set từ Checkpoint đã huấn luyện"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints/baseline_cnn1d_motionsense_best.pt")

    try:
        # 1. Kiểm tra sự tồn tại của Checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Không tìm thấy Checkpoint tại: {checkpoint_path}")

        # 2. Nạp dữ liệu Test
        _, _, test_loader = get_motionsense_dataloaders()

        # 3. Khởi tạo mô hình và nạp trọng số đã lưu
        model = SupervisedHARModel(
            in_channels=MotionSenseConfig.IN_CHANNELS,
            num_classes=MotionSenseConfig.NUM_CLASSES
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Đã nạp thành công trọng số từ Checkpoint.")

        # 4. Đo độ phức tạp tính toán (Params, FLOPs, Latency) và lưu vào config.json
        input_sample_shape = (1, MotionSenseConfig.IN_CHANNELS, MotionSenseConfig.WINDOW_SIZE)
        complexity_info = measure_model_complexity(model, input_size=input_sample_shape, device=device)
        print_complexity_report(complexity_info)
        tracker.log_complexity(complexity_info)

        # 5. Kích hoạt Evaluator Engine và lưu Confusion Matrix vào thư mục plots
        evaluator = ModelEvaluator(class_names=CLASS_NAMES, device=device)
        cm_save_path = os.path.join(tracker.plot_dir, "confusion_matrix.png")

        evaluator.evaluate(
            model=model,
            test_loader=test_loader,
            plot_save_path=cm_save_path,
            title_prefix="Baseline 1D-CNN Test Set"
        )

    finally:
        tracker.close()


if __name__ == "__main__":
    main()