"""
===============================================================================
MODULE NẠP DỮ LIỆU (DATASET & DATALOADER) CHO MOTIONSENSE
===============================================================================
Mục đích:
    - Duyệt qua từng thư mục thử nghiệm (dws_1, jog_9, wlk_7...) trong MotionSense.
    - Đọc các file sub_1.csv đến sub_24.csv.
    - Cắt chuỗi thời gian liên tục thành các Cửa sổ trượt (Windows) kích thước (6, 128).
    - Phân chia tập Train/Test theo ID người dùng (Subject ID).
    - Đóng gói thành PyTorch DataLoader sẵn sàng đưa vào mô hình Neural Network.
===============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config.motionsense_config import \
    MotionSenseConfig  # Import các tham số cấu hình từ file motionsense_config.py đã tạo ở trên


class MotionSenseDataset(Dataset):
    """
    Class Dataset tùy chỉnh đọc dữ liệu MotionSense cho PyTorch
    """

    def __init__(self,
                 data_dir,
                 subjects_list,
                 config=MotionSenseConfig
                 ):
        """
        Khởi tạo và tiền xử lý dữ liệu

        Tham số:
            data_dir (str): Đường dẫn đến thư mục data/raw/motion_sense/
            subjects_list (list): Danh sách ID người dùng cần nạp (vd: [1..18] cho Train, [19..24] cho Test)
            config: Class chứa tham số cấu hình
        """
        self.config = config
        self.windows = []  # Danh sách chứa các tensor cửa sổ dữ liệu (Kích thước mỗi phần tử: 6 x 128)
        self.labels = []  # Danh sách chứa nhãn tương ứng (Số nguyên từ 0 đến 5)

        # Gọi hàm nội bộ để load toàn bộ file csv và cắt cửa sổ
        self._load_and_process_data(data_dir, subjects_list)

        # Chuyển đổi danh sách Python thành Tensor của PyTorch
        # Shape của self.windows sau khi chuyển: (Tổng_Số_Windows, 6, 128)
        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        # Shape của self.labels sau khi chuyển: (Tổng_Số_Windows,)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def _load_and_process_data(self, data_dir, subjects_list):
        """
        Hàm nội bộ: Duyệt file, đọc dữ liệu và thực hiện Cắt cửa sổ trượt (Sliding Window)
        """
        # Tìm tất cả các thư mục hoạt động (dws_1, wlk_7, jog_9...)
        folder_paths = glob.glob(os.path.join(data_dir, "*_*"))
        print(f"DEBUG: Tìm thấy {len(folder_paths)} thư mục hoạt động tại đường dẫn: '{data_dir}'")
        for folder in folder_paths:
            folder_name = os.path.basename(folder)  # Lấy tên thư mục, ví dụ: 'wlk_7'

            # Bỏ qua nếu là file lẻ hoặc không đúng định dạng thư mục hoạt động
            if not os.path.isdir(folder):
                continue

            # Trích xuất mã hoạt động (vd: 'wlk_7' -> lấy chữ 'wlk')
            act_code = folder_name.split('_')[0]

            # Bỏ qua nếu mã hoạt động không nằm trong LABEL_MAP
            if act_code not in self.config.LABEL_MAP:
                continue

            # Lấy nhãn số tương ứng (vd: 'wlk' -> 2)
            label = self.config.LABEL_MAP[act_code]

            # Duyệt qua các người dùng trong danh sách yêu cầu (subjects_list)
            for sub_id in subjects_list:
                file_path = os.path.join(folder, f"sub_{sub_id}.csv")

                # Kiểm tra nếu file sub_X.csv tồn tại thì mới đọc
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    # Trích xuất đúng 6 kênh dữ liệu đã cấu hình: [userAcceleration + rotationRate]
                    # Matrix thu được có shape: (Tổng_Số_Samples_Trong_File, 6)
                    sensor_data = df[self.config.FEATURE_COLS].values

                    # -----------------------------------------------------------------
                    # THUẬT TOÁN CẮT CỬA SỔ TRƯỢT (SLIDING WINDOW)
                    # -----------------------------------------------------------------
                    num_samples = len(sensor_data)
                    window_size = self.config.WINDOW_SIZE  # 128
                    stride = self.config.STRIDE  # 64

                    # Vòng lặp trượt từ đầu đến cuối file CSV
                    for start in range(0, num_samples - window_size + 1, stride):
                        end = start + window_size

                        # Cắt 1 đoạn 128 mẫu -> Shape thu được: (128, 6)
                        window = sensor_data[start:end]

                        # CHUYỂN VỊ MA TRẬN: (128, 6) -> (6, 128) cho đúng chuẩn PyTorch Conv1D
                        window_transposed = window.T

                        # Lưu cửa sổ và nhãn tương ứng vào danh sách
                        self.windows.append(window_transposed)
                        self.labels.append(label)

    def __len__(self):
        """
        Hàm bắt buộc của PyTorch Dataset: Trả về tổng số lượng cửa sổ cắt được
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Hàm bắt buộc của PyTorch Dataset: Trả về cửa sổ và nhãn tại chỉ số idx

        Trả về:
            x (Tensor): Cửa sổ dữ liệu cảm biến shape (6, 128)
            y (Tensor): Nhãn số tương ứng (scatalar integer, vd: 2)
        """
        return self.windows[idx], self.labels[idx]
