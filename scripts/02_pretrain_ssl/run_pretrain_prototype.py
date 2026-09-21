import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.motionsense_config import MotionSenseConfig
from config.uci_har_config import UCIHARConfig
from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from models.encoders.builder import build_encoder
from models.ssl.prototype.cluster_model import PrototypicalHARModel
from losses.swav_loss import SwAVPrototypeLoss

# ================== CẤU HÌNH THỰC NGHIỆM ==================
DATASETS = ["uci_har", "motionsense"]
# Đầy đủ 4 kiến trúc benchmark đồng bộ
BACKBONE_TYPES = ["standard", "tstcc", "cnn_transformer", "vit_1d"]

EPOCHS = 40
WARMUP_EPOCHS = 5
BATCH_SIZE = 64
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_PROTOTYPES = 45            # K = 45
TAU_S = 0.1                    # Softmax temperature
EPSILON = 0.05                 # Sinkhorn temperature
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_MAP = {
    "motionsense": (Path(MotionSenseConfig.DATA_ALL_PATH), int(MotionSenseConfig.IN_CHANNELS)),
    "uci_har": (Path(UCIHARConfig.DATA_ALL_PATH), int(UCIHARConfig.IN_CHANNELS)),
}


def detect_feature_dim(encoder: nn.Module, in_channels: int) -> int:
    """
    Dual-probe an toàn tuyệt đối:
    Chiều nào giữ nguyên khi thay đổi chiều dài thời gian (128 vs 256) chính là feature_dim.
    """
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        device = next(encoder.parameters()).device
        out1 = encoder(torch.randn(2, in_channels, 128, device=device))
        out2 = encoder(torch.randn(2, in_channels, 256, device=device))

        if out1.dim() == 2:
            detected = out1.size(1)
        elif out1.dim() == 3:
            # Chiều không đổi giữa out1 và out2 là feature dimension
            if out1.size(1) == out2.size(1):
                detected = out1.size(1)   # Format (B, D, L)
            elif out1.size(-1) == out2.size(-1):
                detected = out1.size(-1)  # Format (B, L, D)
            else:
                raise ValueError(f"Không thể xác định feature_dim giữa 2 probes: {out1.shape}, {out2.shape}")
        else:
            raise ValueError(f"Shape đầu ra không hợp lệ: {out1.shape}")

    if was_training:
        encoder.train()
    return detected


def adjust_lr(optimizer, epoch, total_epochs, base_lr, warmup_epochs):
    """Linear warmup + Cosine annealing."""
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def train_prototype_single_domain(domain_name: str, data_path: Path, in_channels: int, backbone_type: str):
    print("\n" + "=" * 85)
    print(f"🚀 PRETRAIN PROTOTYPE | DOMAIN: {domain_name.upper()} | BACKBONE: {backbone_type.upper()}")
    print(f"📁 Data: {data_path} | K = {NUM_PROTOTYPES} | Device: {DEVICE}")
    print("=" * 85)

    # 1. Dataset & DataLoader (bắt buộc drop_last=True cho Sinkhorn)
    dataset = ContrastiveDatasetWrapper(data_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 2. Xây dựng Backbone & Dò feature_dim bằng dual-probe
    backbone = build_encoder(backbone_type=backbone_type, in_channels=in_channels).to(DEVICE)
    detected_dim = detect_feature_dim(backbone, in_channels=in_channels)
    print(f"🧠 feature_dim xác thực = {detected_dim}")

    # 3. Model & Loss (channel_last=False cố định cho cả 4 backbone)
    model = PrototypicalHARModel(
        encoder=backbone,
        feature_dim=detected_dim,
        projection_dim=64,
        num_prototypes=NUM_PROTOTYPES,
        temperature=TAU_S,
        channel_last=False
    ).to(DEVICE)

    criterion = SwAVPrototypeLoss(epsilon=EPSILON).to(DEVICE)

    # 4. Tách biệt hai optimizer: network vs prototypes
    network_params = [p for n, p in model.named_parameters() if n != "prototypes"]
    opt_network = torch.optim.AdamW(network_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    opt_proto = torch.optim.AdamW([model.prototypes], lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    # 5. Đường dẫn checkpoint chuẩn hóa đồng bộ với pipeline downstream
    save_dir = PROJECT_ROOT / "checkpoints" / "ssl_pretrain" / "prototype" / domain_name / backbone_type
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"prototype_{backbone_type}_encoder_pretrained_{domain_name}.pt"

    max_entropy = math.log(NUM_PROTOTYPES)
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        cur_lr = adjust_lr(opt_network, epoch, EPOCHS, BASE_LR, WARMUP_EPOCHS)
        adjust_lr(opt_proto, epoch, EPOCHS, BASE_LR, WARMUP_EPOCHS)

        total_loss, entropy_acc, n_batches = 0.0, 0.0, 0

        for x_w, x_s in loader:
            x_w, x_s = x_w.to(DEVICE), x_s.to(DEVICE)

            opt_network.zero_grad()
            opt_proto.zero_grad()

            scores_w, scores_s = model(x_w, x_s)
            loss, q_w = criterion(scores_w, scores_s, return_q=True)

            loss.backward()

            # Cập nhật song song cả network và prototype ngay từ epoch 1
            opt_network.step()
            opt_proto.step()

            # Giám sát Cluster Entropy
            with torch.no_grad():
                avg_q = q_w.mean(dim=0)
                batch_entropy = -(avg_q * torch.log(avg_q + 1e-8)).sum().item()
                entropy_acc += batch_entropy

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_entropy = entropy_acc / max(1, n_batches)

        # Cảnh báo nếu có nguy cơ sụp cụm
        if avg_entropy < (0.3 * max_entropy):
            print(f"⚠️  Cảnh báo: Entropy thấp ({avg_entropy:.2f}/{max_entropy:.2f}) tại epoch {epoch}")

        # Tiêu chuẩn lưu Checkpoint: Dựa trên Loss tốt nhất sau warmup
        if epoch > WARMUP_EPOCHS and avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), ckpt_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"Epoch [{epoch:02d}/{EPOCHS:02d}] | LR: {cur_lr:.2e} | "
                f"Loss: {avg_loss:.4f} (Best: {best_loss:.4f}) | "
                f"Entropy: {avg_entropy:.2f}/{max_entropy:.2f}"
            )

    # Fallback lưu epoch cuối nếu chưa lưu
    if not ckpt_path.exists():
        torch.save(model.encoder.state_dict(), ckpt_path)
        print(f"⚠️ Fallback: Đã lưu checkpoint tại epoch cuối.")

    print(f"💾 Checkpoint hoàn tất: {ckpt_path}")

    # Thu hồi VRAM
    del model, backbone, criterion, opt_network, opt_proto, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    total_runs = len(BACKBONE_TYPES) * len(DATASETS)
    run_idx = 1

    for backbone in BACKBONE_TYPES:
        for name in DATASETS:
            print(f"\n[{run_idx}/{total_runs}] TIẾN TRÌNH: BACKBONE={backbone.upper()} | DATASET={name.upper()}")
            path, in_channels = DATASET_MAP[name]
            train_prototype_single_domain(name, path, in_channels, backbone_type=backbone)
            run_idx += 1

    print("\n🎉 HOÀN TẤT HUẤN LUYỆN PROTOTYPE CHO TOÀN BỘ 4 BACKBONES TRÊN TẤT CẢ TẬP DỮ LIỆU!")


if __name__ == "__main__":
    main()