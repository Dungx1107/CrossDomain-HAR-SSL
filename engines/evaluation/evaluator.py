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

ĐẦU VÀO / ĐẦU RA:
    - Đầu vào: Mô hình PyTorch đã hoàn tất huấn luyện, Test DataLoader, danh sách tên nhãn.
    - Đầu ra: Dictionary chứa các chỉ số thực nghiệm, in kết quả ra terminal, lưu ảnh biểu đồ ma trận.
===============================================================================
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visualization import plot_confusion_matrix


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

    def run_inference(self, model, dataloader):
        """
        Thực thi toàn bộ lượt suy luận qua DataLoader để thu thập nhãn thực tế và nhãn dự đoán.

        Tham số:
            model (nn.Module): Mô hình cần đánh giá.
            dataloader (DataLoader): Nạp dữ liệu kiểm thử.

        Trả về:
            tuple: (np.ndarray y_true, np.ndarray y_pred)
        """
        model.eval()
        all_preds = []
        all_targets = []

        # Tắt hoàn toàn bộ nhớ Gradient để giải phóng RAM/VRAM
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)

                # Lan truyền tiến nhận Logits: (B, Num_Classes)
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)

                # Chuyển dữ liệu từ GPU về CPU an toàn trước khi ép kiểu NumPy
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        return np.array(all_targets), np.array(all_preds)

    def evaluate(self, model, test_loader, plot_save_path=None, title_prefix="Evaluation"):
        """
        Thực hiện đánh giá toàn diện, in bảng biểu ra màn hình và lưu hình ảnh Confusion Matrix.

        Tham số:
            model (nn.Module): Mô hình cần kiểm định.
            test_loader (DataLoader): Dữ liệu Test kiểm định độc lập.
            plot_save_path (str, optional): Đường dẫn file .png để lưu biểu đồ ma trận nhầm lẫn.
            title_prefix (str): Tiêu đề mô tả kịch bản chạy (ví dụ: 'MotionSense Baseline' hay 'SSL Transfer 5%').

        Trả về:
            dict: Bảng tổng hợp các chỉ số kiểm thử phục vụ vẽ biểu đồ hoặc lập bảng kết quả.
        """
        # Thu thập toàn bộ kết quả dự đoán
        y_true, y_pred = self.run_inference(model, test_loader)

        # ---------------------------------------------------------------------
        # 1. TÍNH TOÁN CÁC THƯỚC ĐO THỐNG KÊ TOÀN CỤC
        # ---------------------------------------------------------------------
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")

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

        # Chỉ lấy đúng số lượng nhãn và tên nhãn thực tế xuất hiện
        num_classes_found = max(len(np.unique(y_true)), len(np.unique(y_pred)))
        eval_labels = list(range(num_classes_found))
        eval_target_names = self.class_names[:num_classes_found]

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
        cm = confusion_matrix(y_true, y_pred)
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
