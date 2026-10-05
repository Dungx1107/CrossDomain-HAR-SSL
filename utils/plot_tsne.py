import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Xác định thư mục gốc của dự án (lùi lại 1 cấp từ thư mục utils)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.builder import build_encoder

# Cấu hình đường dẫn dựa trên cây thư mục local
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "motionsense" / "dataset_all.pt"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "ssl_pretrain" / "prototype" / "motionsense"

# Tạo thư mục lưu ảnh
OUTPUT_DIR = PROJECT_ROOT / "figures" / "pretrain" / "prototype" / "tsne_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Jogging']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(data_path, max_samples=3000):
    print(f"Đang nạp dữ liệu từ: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        X = data.get("samples", data.get("X"))
        y = data.get("labels", data.get("y"))
    else:
        X, y = data[0], data[1]

    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long)

    if len(X) > max_samples:
        indices = torch.randperm(len(X))[:max_samples]
        X, y = X[indices], y[indices]

    return X, y


def plot_tsne_for_backbone(backbone_type, ckpt_name, X, y):
    ckpt_path = CHECKPOINT_DIR / backbone_type / ckpt_name
    if not ckpt_path.exists():
        print(f"❌ Không tìm thấy checkpoint: {ckpt_path}")
        return

    print(f"🚀 Đang xử lý t-SNE cho: {backbone_type.upper()}")

    encoder = build_encoder(backbone_type=backbone_type, in_channels=6).to(DEVICE)
    encoder.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    encoder.eval()

    features = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i + batch_size].to(DEVICE)
            feat = encoder(batch_x)
            features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)

    # Làm phẳng (flatten) nếu features là mảng 3D
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)

    labels = y.numpy()

    print("   ⏳ Đang nội suy không gian 2D với t-SNE (có thể mất 1-2 phút)...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=[CLASS_NAMES[lbl] for lbl in labels],
        palette=sns.color_palette("hls", len(CLASS_NAMES)),
        legend="full",
        alpha=0.7,
        s=40,
        edgecolor=None
    )
    plt.title(f"t-SNE Projection: {backbone_type.upper()} (SwAV Prototype)", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(title="Activity Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"tsne_{backbone_type}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Đã lưu đồ thị tại: {save_path}\n")


def main():
    print("=" * 60)
    print("BẮT ĐẦU VẼ T-SNE CHO CÁC MÔ HÌNH PROTOTYPE TỪ LOCAL CHECKPOINTS")
    print("=" * 60)

    X, y = load_data(DATA_PATH, max_samples=3000)
    print(f"📦 Đã nạp {len(X)} mẫu dữ liệu để trực quan hóa.\n")

    models_to_plot = [
        ("standard", "prototype_standard_encoder_pretrained_motionsense.pt"),
        ("cnn_transformer", "prototype_cnn_transformer_encoder_pretrained_motionsense.pt"),
        ("vit_1d", "prototype_vit_1d_encoder_pretrained_motionsense.pt")
    ]

    for backbone, ckpt in models_to_plot:
        plot_tsne_for_backbone(backbone, ckpt, X, y)


if __name__ == "__main__":
    main()