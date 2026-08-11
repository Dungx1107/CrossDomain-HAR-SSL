"""
===============================================================================
MODULE CLASSIFIER HEAD (ĐẦU PHÂN LOẠI CHO BÀI TOÁN SUPERVISED / FINE-TUNE)
===============================================================================
Mục đích:
    Nhận Vector đặc trưng (128 chiều) từ Encoder và chuyển thành 6 giá trị Logits
    đại diện cho 6 hoạt động con người.
===============================================================================
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, feature_dim: int = 128, num_classes: int = 6, dropout_rate: float = 0.2):
        """
        Tham số:
            feature_dim (int): Kích thước Vector đầu vào từ Encoder (default: 128)
            num_classes (int): Số lượng nhãn phân loại (default: 6 cho MotionSense)
            dropout_rate (float): Tỷ lệ triệt tiêu ngẫu nhiên neuron để chống Overfitting
        """
        super(ClassifierHead, self).__init__()

        # Dropout giúp mô hình không bị quá phụ thuộc vào 1 vài neuron cố định
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Lớp Tuyến tính (Fully Connected / Dense) biến 128 chiều -> 6 chiều
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (Batch_Size, feature_dim) -> vd: (64, 128)
        Trả về logits shape: (Batch_Size, num_classes) -> vd: (64, 6)
        """
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
