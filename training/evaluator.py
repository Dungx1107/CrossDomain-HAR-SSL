"""
===============================================================================
MỤC ĐÍCH:
    Cung cấp Engine đánh giá độc lập (Evaluation Engine) cho các bài toán
    phân loại hành động (HAR), hỗ trợ suy luận không tính Gradient,
    tính toán báo cáo đa chỉ số và tự động xuất Ma trận nhầm lẫn.
===============================================================================
"""

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from utils.visualization import plot_confusion_matrix


class ModelEvaluator:
    """
    Engine chuyên trách việc kiểm thử và đánh giá hiệu năng mô hình.
    """
    def __init__(self, class_names=None, device=None):
        self.class_names = class_names
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_inference(self, model, dataloader):
        """Thực hiện suy luận qua toàn bộ dataloader và thu về mảng nhãn thực tế / dự đoán."""
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        return np.array(all_targets), np.array(all_preds)

    def evaluate(self, model, test_loader, plot_save_path=None, title_prefix="Evaluation"):
        """
        Thực hiện đánh giá toàn diện, in kết quả ra màn hình và vẽ Confusion Matrix.
        """
        y_true, y_pred = self.run_inference(model, test_loader)

        # 1. Tính toán các chỉ số cốt lõi
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

        print("\n" + "=" * 75)
        print(f"{f'KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ({title_prefix})':^75}")
        print("=" * 75)
        print(f"🎯 Test Accuracy   : {acc * 100:.2f}%")
        print(f"🏆 Test Macro F1   : {macro_f1 * 100:.2f}% (Thước đo chính)")
        print(f"⚖️ Test Weighted F1 : {weighted_f1 * 100:.2f}%")
        print("=" * 75)

        # 2. In bảng thống kê chi tiết từng lớp
        target_names = self.class_names if self.class_names else [f"Class {i}" for i in range(len(np.unique(y_true)))]
        print("\n📋 BẢNG THỐNG KÊ CHI TIẾT TỪNG LỚP HÀNH ĐỘNG:")
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

        # 3. Tính toán Ma trận nhầm lẫn
        cm = confusion_matrix(y_true, y_pred)
        print("🔍 MA TRẬN NHẦM LẪN (SỐ LƯỢNG MẪU):")
        print(cm)

        # 4. Tự động vẽ và lưu hình ảnh nếu có đường dẫn
        if plot_save_path:
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plot_confusion_matrix(
                cm=cm,
                class_names=target_names,
                save_path=plot_save_path,
                title=f"{title_prefix} (Macro F1: {macro_f1*100:.2f}%)"
            )
            print(f"\n🖼️ Đã lưu Confusion Matrix tại: {plot_save_path}")

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm
        }