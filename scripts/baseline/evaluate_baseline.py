"""
===============================================================================
SCRIPT RUNNER: scripts/baseline/evaluate_baseline.py (EVALUATE STANDALONE)
===============================================================================
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.motionsense_config import MotionSenseConfig
from datasets.motionsense.loader import get_motionsense_loaders
from models.baseline.supervised_model import Supervised1DCNN
from training.evaluator import ModelEvaluator
from utils.logger import ExperimentTracker

CLASS_NAMES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]


def main():
    tracker = ExperimentTracker(
        exp_name="eval_standalone_baseline",
        base_dir="experiments",
        config_obj=MotionSenseConfig,
        notes="Đánh giá độc lập từ checkpoint đã lưu"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints/baseline_cnn1d_motionsense_best.pt")

    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ Không tìm thấy Checkpoint tại: {checkpoint_path}")

        _, _, test_loader = get_motionsense_loaders(batch_size=MotionSenseConfig.BATCH_SIZE)

        model = Supervised1DCNN(
            in_channels=MotionSenseConfig.IN_CHANNELS,
            num_classes=MotionSenseConfig.NUM_CLASSES
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        evaluator = ModelEvaluator(class_names=CLASS_NAMES, device=device)
        cm_save_path = os.path.join(tracker.run_dir, "confusion_matrix.png")

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