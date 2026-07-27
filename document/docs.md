# CrossDomain-HAR-SSL

```text
CrossDomain-HAR-SSL/
│
├── data/                       # Thư mục chứa dữ liệu (đã ignore trên git)
│   ├── raw/                    # Dữ liệu gốc tải về (Zip, CSV, TXT)
│   │   ├── uci_har/
│   │   └── hhar/
│   └── processed/              # Dữ liệu đã tiền xử lý & cắt khung (.pt / .npy)
│       ├── uci_har_9ch.pt
│       └── hhar_6ch.pt
│
├── config/                     # Quản lý tham số (Hyperparameters & Setup)
│   ├── base_config.py          # Configuration chung (Batch size, Epochs, LR)
│   └── uci_config.py           # Config riêng cho từng dataset/domain
│
├── datasets/                   # PyTorch Dataset & DataLoader
│   ├── __init__.py
│   ├── base_dataset.py         # Class cơ sở cho DataLoader
│   ├── uci_har_loader.py       # Script load và preprocess UCI-HAR
│   └── hhar_loader.py          # Script load và preprocess HHAR
│
├── models/                     # Kiến trúc mạng (Architectures)
│   ├── __init__.py
│   ├── encoder1d.py            # 1D-CNN Encoder (Trích xuất đặc trưng)
│   ├── classifier.py           # Tầng phân loại (Linear Classifier)
│   └── ts_tcc_modules.py       # Các khối Transformer / Temporal Contextual
│
├── losses/                     # Định nghĩa các hàm Loss
│   ├── __init__.py
│   ├── contrastive_loss.py     # Loss cho SSL (Temporal / Contextual Contrastive)
│   └── prototype_loss.py       # Loss căn chỉnh nguyên mẫu (Prototype Alignment - Bài 2)
│
├── utils/                      # Tools bổ trợ
│   ├── __init__.py
│   ├── augmentations.py        # Các phép biến đổi dữ liệu 1D (Weak/Strong)
│   ├── metrics.py              # Tính Accuracy, F1-Score, Confusion Matrix
│   └── visualization.py        # Vẽ đồ thị tín hiệu 1D, t-SNE plot
│
├── scripts/                    # Scripts tự động tải & xử lý dữ liệu
│   ├── download_uci.sh         # Script tải tự động UCI-HAR
│   └── preprocess_all.py       # Run chuyển đổi tất cả dữ liệu gốc sang .pt
│
├── .gitignore                  # Bỏ qua folder data/ và checkpoint weights
├── requirements.txt            # Danh sách thư viện cần thiết
├── main.py                     # Script thực thi chính (Train SSL / Fine-tune)
└── README.md                   # Hướng dẫn chạy dự án
```