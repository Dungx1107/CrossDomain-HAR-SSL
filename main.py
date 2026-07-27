import sys
import time


def check_environment():
    print("=" * 60)
    print("      KIỂM TRA MÔI TRƯỜNG & THƯ VIỆN DỰ ÁN HAR-SSL")
    print("=" * 60)

    # 1. Kiểm tra Python Executable
    print(f"[✓] Python Executable : {sys.executable}")
    print(f"[✓] Python Version    : {sys.version.split()[0]}\n")

    # 2. Kiểm tra các thư viện cơ bản & Data Science
    libraries = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "tqdm"),
    ]

    print("-" * 60)
    print("1. KHỞI TẠO CÁC THƯ VIỆN XỬ LÝ DỮ LIỆU:")
    print("-" * 60)
    for module_name, display_name in libraries:
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "N/A")
            print(f"  • {display_name:<15}: OK (v{version})")
        except ImportError as e:
            print(f"  x {display_name:<15}: LỖI KHÔNG TÌM THẤY! ({e})")

    # 3. Kiểm tra PyTorch, Torchvision & GPU/CUDA
    print("\n" + "-" * 60)
    print("2. KIỂM TRA PYTORCH & CẤU HÌNH GPU (CUDA):")
    print("-" * 60)
    try:
        import torch
        import torchvision

        print(f"  • PyTorch Version   : {torch.__version__}")
        print(f"  • Torchvision Ver  : {torchvision.__version__}")

        is_cuda = torch.cuda.is_available()
        print(f"  • CUDA Available   : {is_cuda}")

        if is_cuda:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  • GPU Count        : {device_count}")
            print(f"  • Primary GPU      : {device_name}")

            # Thử nghiệm tính toán thực tế trên GPU (RTX 2050)
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print("  • GPU Tensor Test  : THÀNH CÔNG (Đã nhân ma trận trên GPU)")
        else:
            print("  ! CẢNH BÁO        : PyTorch đang chạy trên CPU!")

    except ImportError as e:
        print(f"  x PyTorch/Torchvision: LỖI! ({e})")

    # 4. Test nhanh một tác vụ nhỏ với TQDM
    print("\n" + "-" * 60)
    print("3. TEST TIẾN TRÌNH (TQDM):")
    print("-" * 60)
    from tqdm import tqdm

    for _ in tqdm(range(5), desc="  • Running Sanity Check"):
        time.sleep(0.05)

    print("\n" + "=" * 60)
    print("  => MÔI TRƯỜNG ĐÃ SẴN SÀNG CHO DỰ ÁN CROSSDOMAIN-HAR-SSL!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    check_environment()