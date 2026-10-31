"""
===============================================================================
MODULE: COMPREHENSIVE MODEL EVALUATION ENGINE
===============================================================================
VAI TRÒ TRONG HỆ THỐNG:
    - Đơn vị đo lường và thẩm định độc lập cuối cùng cho toàn bộ hệ thống mô hình.
    - Không tham gia vào quá trình tính toán Gradient hay tối ưu tham số.

NHIỆM VỤ CHÍNH:
    1. Chạy suy luận (Batch Inference) trên tập kiểm thử mà không tích lũy đồ thị Gradient.
    2. Tính toán các chỉ số thống kê chuẩn mực trong nghiên cứu học máy:
       - Accuracy (Độ chính xác tổng thể).
       - Macro F1 (Trung bình cộng F1 của từng lớp - chỉ số vàng cho bài toán mất cân bằng dữ liệu).
       - Weighted F1 (F1 có trọng số theo tỷ lệ phân bố mẫu của các lớp).
    3. Trích xuất báo cáo phân loại chi tiết từng lớp hành vi (Precision, Recall, F1-Score).
    4. Tính toán và gọi module trực quan hóa Ma trận nhầm lẫn (Confusion Matrix).

HÀM DÙNG CHUNG (module-level) — nguồn duy nhất cho phần dự đoán + metric:
    - predict(model, loader, device)        -> (y_true, y_pred)
    - compute_metrics(y_true, y_pred, scale) -> dict các metric toàn cục
    Được dùng bởi: ModelEvaluator (dưới đây), engines/transfer/finetune_trainer.py,
    engines/evaluation/few_shot_report.py.

ĐẦU VÀO / ĐẦU RA:
    - Đầu vào: Mô hình PyTorch đã hoàn tất huấn luyện, Test DataLoader, danh sách tên nhãn.
    - Đầu ra: Dictionary chứa các chỉ số thực nghiệm, in kết quả ra terminal, lưu ảnh biểu đồ ma trận.
===============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score)
from utils.visualization import plot_confusion_matrix


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# HÀM DÙNG CHUNG
# =============================================================================

def predict(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chạy model (chế độ eval, không gradient) qua DataLoader.
    Trả về (y_true, y_pred) dạng np.ndarray.
    """
    model.eval()
    all_preds, all_targets = [], []

    # Tắt hoàn toàn bộ nhớ Gradient để giải phóng RAM/VRAM
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            logits = model(x_batch.to(device))  # (B, Num_Classes)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return np.array(all_targets), np.array(all_preds)


def compute_metrics(y_true, y_pred, scale: float = 100.0) -> Dict[str, float]:
    """
    Các metric toàn cục, nhân với `scale` (100.0 -> thang %, 1.0 -> thang 0-1).
    Macro/weighted tính trên các nhãn xuất hiện trong y_true ∪ y_pred.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred) * scale,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * scale,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0) * scale,
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * scale,
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * scale,
    }


# =============================================================================
# ENGINE
# =============================================================================

class ModelEvaluator:
    """
    Engine chuyên biệt để phân tích định lượng hiệu năng của mô hình phân loại HAR.
    """

    def __init__(self, class_names=None, device=None):
        """
        Khởi tạo Engine đánh giá.

        Tham số:
            class_names (list, optional): Danh sách tên chữ của các hoạt động.
                                         Mặc định sử dụng 5 lớp giao thoa:
                                         ['Downstairs', 'Upstairs', 'Walking', 'Sitting', 'Standing'].
            device (torch.device, optional): Thiết bị chạy suy luận.
        """
        self.class_names = class_names if class_names else ['Downstairs', 'Upstairs', 'Walking', 'Sitting', 'Standing']
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_inference(self,
                      model,
                      dataloader
                      ):
        """
        Thực thi toàn bộ lượt suy luận qua DataLoader để thu thập nhãn thực tế và nhãn dự đoán.
        Tham số:
            model (nn.Module): Mô hình cần đánh giá.
            dataloader (DataLoader): Nạp dữ liệu kiểm thử.
        Trả về:
            tuple: (np.ndarray y_true, np.ndarray y_pred)
        """
        return predict(model, dataloader, self.device)

    def evaluate(self,
                 model,
                 test_loader,
                 plot_save_path=None,
                 title_prefix="Evaluation"
                 ):
        """
        Thực hiện đánh giá toàn diện, in bảng biểu ra màn hình và lưu hình ảnh Confusion Matrix.

        Tham số:
            model (nn.Module): Mô hình cần kiểm định.
            test_loader (DataLoader): Dữ liệu Test kiểm định độc lập.
            plot_save_path (str, optional): Đường dẫn file .png để lưu biểu đồ ma trận nhầm lẫn.
            title_prefix (str): Tiêu đề mô tả kịch bản chạy (ví dụ: 'MotionSense Baseline' hay 'SSL Transfer 5%').

        Trả về:
            dict: Bảng tổng hợp các chỉ số kiểm thử phục vụ vẽ biểu đồ hoặc lập bảng kết quả
                  (accuracy / macro_f1 / weighted_f1 theo thang 0-1).
        """
        # Thu thập toàn bộ kết quả dự đoán
        y_true, y_pred = self.run_inference(model, test_loader)

        # ---------------------------------------------------------------------
        # 1. TÍNH TOÁN CÁC THƯỚC ĐO THỐNG KÊ TOÀN CỤC
        # ---------------------------------------------------------------------
        metrics = compute_metrics(y_true, y_pred, scale=1.0)
        acc = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]

        print("\n" + "=" * 75)
        print(f"{f'KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ({title_prefix})':^75}")
        print("=" * 75)
        print(f"🎯 Test Accuracy    : {acc * 100:.2f}%")
        print(f"🏆 Test Macro F1    : {macro_f1 * 100:.2f}% (Chỉ số nghiên cứu chính)")
        print(f"⚖️ Test Weighted F1  : {weighted_f1 * 100:.2f}%")
        print("=" * 75)

        # ---------------------------------------------------------------------
        # 2. BÁO CÁO CHI TIẾT TỪNG LỚP HÀNH VI
        # ---------------------------------------------------------------------
        # Hỗ trợ phân tích xem lớp nào mô hình nhận diện tốt (ví dụ Walking),
        # lớp nào hay bị nhầm lẫn do vị trí cảm biến khác biệt (ví dụ Upstairs vs Downstairs)
        print("\n📋 BẢNG THỐNG KÊ CHI TIẾT TỪNG LỚP HÀNH ĐỘNG:")

        # Luôn cố định đánh giá trên toàn bộ danh sách nhãn đã định nghĩa
        eval_labels = list(range(len(self.class_names)))
        eval_target_names = self.class_names

        print(classification_report(
            y_true,
            y_pred,
            labels=eval_labels,
            target_names=eval_target_names,
            digits=4,
            zero_division=0
        ))
        # ---------------------------------------------------------------------
        # 3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
        # ---------------------------------------------------------------------
        # labels=eval_labels: ma trận luôn đủ N x N dù tập test thiếu một lớp
        cm = confusion_matrix(y_true, y_pred, labels=eval_labels)
        print("🔍 MA TRẬN NHẦM LẪN (SỐ LƯỢNG MẪU):")
        print(cm)

        # ---------------------------------------------------------------------
        # 4. XUẤT BIỂU ĐỒ HÌNH ẢNH MINH HỌA
        # ---------------------------------------------------------------------
        if plot_save_path:
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plot_confusion_matrix(
                cm=cm,
                class_names=eval_target_names,
                save_path=plot_save_path,
                title=f"{title_prefix} (Macro F1: {macro_f1 * 100:.2f}%)"
            )
            print(f"\n🖼️ Đã lưu hình ảnh Confusion Matrix tại: {plot_save_path}")

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": cm
        }