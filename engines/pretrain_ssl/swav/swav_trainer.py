"""
===============================================================================
MODULE: SWAV TRAINER — VÒNG HUẤN LUYỆN PROTOTYPE / PHÂN CỤM
===============================================================================
"PHÂN CỤM" CỦA SWAV NẰM Ở 3 FILE, KHÔNG PHẢI RIÊNG TRAINER
    models/ssl/swav/swav_model.py   prototypes (K×D, tham số học được) + backbone + projection
                                     -> trả cosine (B, K)
    losses/swav_loss.py             Sinkhorn-Knopp (gán cụm đều) + swapped loss + entropy
    swav_trainer.py  (file này)     KHI NÀO và THẾ NÀO prototype được học:
                                     lịch LR, nhóm weight decay, đóng băng prototype,
                                     chọn checkpoint, ghi log

    Trainer KHÔNG làm toán phân cụm. Nó chỉ điều phối và quyết định "ai được cập nhật, lúc nào".

THỨ TỰ MỖI ITERATION  (đối chiếu hàm train() trong main_swav.py của facebookresearch/swav)
    (1) đặt learning rate     bản gốc: theo iteration | ở đây: theo EPOCH (_lr_at_epoch)
    (2) chuẩn hóa prototype   bản gốc: in-place, no_grad | ở đây: đã nằm trong model.forward (cách B)
    (3) forward               cos_w, cos_s = model(x_w, x_s)            (B, K) mỗi cái
    (4) Sinkhorn + loss       loss, entropy = criterion(cos_w, cos_s)
    (5) backward              loss.backward()
    (6) đóng băng prototype   bản gốc: iteration < 313 | ở đây: epoch <= FREEZE_PROTOTYPES_EPOCHS
    (7) cập nhật trọng số     optimizer.step()
    [LƯU Ý] Các mốc trên dựa theo trí nhớ về mã chính thức; bạn nên mở main_swav.py để đối chiếu từng dòng.

KHÁC BẢN GỐC (CÓ CHỦ Ý, phù hợp dữ liệu HAR nhỏ)
    - AdamW thay SGD + LARC; LR đổi theo epoch thay vì theo iteration.
    - 2 view (weak/strong) thay multi-crop; KHÔNG có queue.
    - Chuẩn hóa prototype trong forward thay vì in-place.
    - Chọn checkpoint "tốt nhất" theo loss, chỉ xét SAU warmup.

CÁC NÚM CHỈNH (config/methods/swav_config.py)
    NUM_PROTOTYPES (K)        số cụm. Giữ B/K >= ~1 (hiện 64/45 ≈ 1.4).
    TAU_S (τ)                 nhiệt độ softmax dự đoán; nhỏ -> sắc hơn.
    EPSILON (ε)               nhiệt độ Sinkhorn; nhỏ -> mã gần one-hot, lớn -> mã gần đều.
    SINKHORN_ITERS            số vòng Sinkhorn (3); nhiều hơn -> ép các cụm đều hơn.
    FREEZE_PROTOTYPES_EPOCHS  số epoch đầu không cập nhật prototype.
    WARMUP_EPOCHS             số epoch warmup LR (cũng là mốc bắt đầu chọn "best").
    WEIGHT_DECAY, LEARNING_RATE, BATCH_SIZE
    MAX_ENTROPY_COLLAPSE_THRESHOLD   ngưỡng cảnh báo = tỉ lệ × ln K.

CHÚ GIẢI THẺ: [ĐỐI CHIẾU] bài báo/mã gốc | [TÙY CHỈNH] chỗ có thể đổi | [LƯU Ý] bẫy hay gặp
              [MỞ RỘNG] chỗ gắn queue / multi-crop (xem thêm losses/swav_loss.py)

Bố cục thư mục: swav_trainer.py (luồng) / swav_reporting.py (ghi file + in log)
===============================================================================
"""

import math
from pathlib import Path
from typing import Dict, Union

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from config.methods.swav_config import SwAVConfig
from datasets.contrastive_dataset import ContrastiveDatasetWrapper
from losses.swav_loss import SwAVLoss
from models.encoders.builder import build_encoder
from models.ssl.swav.swav_model import SwAVModel
from engines.pretrain_ssl.swav import swav_reporting as rep
from utils.checkpoint_paths import ssl_checkpoint_name, ssl_checkpoint_dir, get_output_root

METHOD_NAME = "swav"  # tên method trong đường dẫn checkpoint: checkpoints/ssl_pretrain/swav/...
MIN_LR = 1e-6  # sàn LR của pha cosine


# =============================================================================
# 1. LỊCH LEARNING RATE
# =============================================================================
def _lr_at_epoch(epoch: int, total_epochs: int, base_lr: float, warmup_epochs: int, min_lr: float = MIN_LR) -> float:
    """
    LR của epoch `epoch` (đếm từ 1): tăng tuyến tính trong warmup rồi giảm cosine về min_lr.

    [ĐỐI CHIẾU] Bản gốc: warmup 10 epoch tuyến tính từ 0, sau đó cosine, cập nhật THEO ITERATION.
    [TÙY CHỈNH] Đổi WARMUP_EPOCHS ở config. Muốn theo iteration: tính LR từ (epoch, batch_idx)
                và gán trong vòng lặp batch thay vì mỗi epoch.
    """
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)  # 0 -> 1
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# 2. DỰNG CÁC THÀNH PHẦN
# =============================================================================
def _build_loader(data_path, batch_size: int, num_workers: int, device: str) -> DataLoader:
    """
    DataLoader trả về cặp (x_weak, x_strong), mỗi cái (B, C, T).

    [LƯU Ý] drop_last=True (khi dataset >= 1 batch) là QUAN TRỌNG cho SwAV: Sinkhorn ép mỗi cụm
            nhận ~B/K mẫu; một batch cuối rất nhỏ sẽ làm ràng buộc đó mất ý nghĩa.
    [MỞ RỘNG - multi-crop] Muốn thêm view: đổi ContrastiveDatasetWrapper để trả danh sách view.
    """
    dataset = ContrastiveDatasetWrapper(data_path)
    use_workers = num_workers if torch.cuda.is_available() else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(dataset) >= batch_size,
        num_workers=use_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(use_workers > 0),
    )


def _build_model(backbone_type: str, in_channels: int, device: str) -> SwAVModel:
    """
    Model chứa backbone + projection head + PROTOTYPES. Không còn nhiệt độ/loss ở đây.
    [LƯU Ý] FEATURE_DIM phải khớp đầu ra thật của encoder (build_encoder).
    """
    return SwAVModel(
        encoder=build_encoder(backbone_type=backbone_type, in_channels=in_channels),
        feature_dim=SwAVConfig.FEATURE_DIM,
        projection_dim=SwAVConfig.PROJECTION_DIM,
        num_prototypes=SwAVConfig.NUM_PROTOTYPES,
    ).to(device)


def _build_criterion(device: str) -> SwAVLoss:
    """
    Loss SwAV: Sinkhorn + swapped prediction. Toàn bộ siêu tham số phân cụm đọc từ config.
    [TÙY CHỈNH] Đổi TAU_S, EPSILON, SINKHORN_ITERS trong swav_config.py (không cần sửa code).
    """
    return SwAVLoss(
        tau=SwAVConfig.TAU_S,
        epsilon=SwAVConfig.EPSILON,
        sinkhorn_iters=SwAVConfig.SINKHORN_ITERS,
    ).to(device)


def _build_optimizer(model: SwAVModel, learning_rate: float) -> torch.optim.Optimizer:
    """
    AdamW với 2 nhóm: có weight decay / không weight decay.

    KHÔNG weight decay cho: prototype, bias, BatchNorm/LayerNorm (mọi tham số 1 chiều).
    Lý do: prototype chỉ có ý nghĩa theo HƯỚNG (đã chuẩn hóa L2 trong forward) nên co độ lớn là vô nghĩa;
    bias và norm-layer là tham số quy mô nhỏ, thường không nên bị kéo về 0.

    [ĐỐI CHIẾU] Bản gốc KHÔNG tách nhóm (SGD, wd=1e-6, bọc LARC). Việc tách nhóm ở đây là thực hành phổ biến
                với AdamW, không phải để "khớp bài báo". Ở AdamW (decoupled) wd=1e-4 tác động rất nhỏ.
    [TÙY CHỈNH] Đổi giá trị ở SwAVConfig.WEIGHT_DECAY. Muốn bỏ tách nhóm: dùng AdamW(model.parameters(), ...).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name == "prototypes":  # bias, norm-layer, và prototype (K×D, 2 chiều)
            no_decay.append(param)
        else:  # trọng số conv / linear
            decay.append(param)
    return AdamW(
        [{"params": decay, "weight_decay": SwAVConfig.WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate,
    )


# =============================================================================
# 3. MỘT EPOCH HUẤN LUYỆN — NƠI PROTOTYPE ĐƯỢC HỌC (đọc kỹ hàm này)
# =============================================================================
def _train_one_epoch(
        model: SwAVModel,
        criterion: SwAVLoss,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        freeze_prototypes: bool = False,
) -> Dict[str, float]:
    """
    Chạy 1 epoch. Trả về {"loss_total", "entropy"} trung bình theo batch.

    freeze_prototypes=True: KHÔNG cập nhật prototype trong epoch này (mạng vẫn học bình thường).
    """
    model.train()
    total_loss = total_entropy = 0.0

    for x_w, x_s in loader:
        x_w = x_w.to(device, non_blocking=True)  # (B, C, T) view weak
        x_s = x_s.to(device, non_blocking=True)  # (B, C, T) view strong

        optimizer.zero_grad()

        # ---- (3) FORWARD -------------------------------------------------------------
        # Mỗi view -> embedding L2 -> nhân với prototype đã L2 -> cosine THÔ.
        # [LƯU Ý] "THÔ" nghĩa là chưa chia τ hay ε. Chia nhiệt độ làm ở bước (4), bên trong criterion.
        cos_w, cos_s = model(x_w, x_s)  # mỗi cái (B, K), giá trị trong [-1, 1]

        # ---- (4) SINKHORN + LOSS -----------------------------------------------------
        # Bên trong criterion:
        #   q_w, q_s = Sinkhorn(cos/ε)       mã gán cụm MỤC TIÊU, không gradient, ép các cụm dùng đều
        #   p_w, p_s = softmax(cos/τ)        dự đoán của mạng, có gradient
        #   loss = ½[CE(q_s, p_w) + CE(q_w, p_s)]   view này dự đoán mã của view kia
        #   entropy                          entropy sử dụng cụm trung bình batch (chỉ để theo dõi)
        # [MỞ RỘNG - queue] Chèn ngay trước bước này: lấy embedding từ model.embed(...), nối queue vào.
        loss, entropy = criterion(cos_w, cos_s)

        # ---- (5) BACKWARD ------------------------------------------------------------
        # Gradient đi: loss -> log_softmax(cos/τ) -> cos -> (embedding, prototype)
        #              -> projection head -> backbone.
        # q KHÔNG nhận gradient (Sinkhorn chạy dưới torch.no_grad) — đúng ý bài báo:
        # mã gán cụm là "nhãn giả" cố định trong bước này.
        loss.backward()

        # ---- (6) ĐÓNG BĂNG PROTOTYPE (giai đoạn đầu) ---------------------------------
        # Đặt .grad = None: optimizer BỎ QUA tham số không có gradient nên prototype giữ nguyên.
        # Phải làm SAU backward() (lúc đó grad mới được tạo) và TRƯỚC step().
        # Vì sao: lúc đầu backbone còn ngẫu nhiên, gradient lên prototype chủ yếu là nhiễu.
        # [ĐỐI CHIẾU] Bản gốc: `if iteration < freeze_prototypes_niters: p.grad = None` cho tham số tên "prototypes".
        if freeze_prototypes:
            model.prototypes.grad = None

        # ---- (7) CẬP NHẬT TRỌNG SỐ ---------------------------------------------------
        optimizer.step()

        total_loss += loss.item()
        total_entropy += entropy.item()

    n = len(loader)
    return {"loss_total": total_loss / n, "entropy": total_entropy / n}


# =============================================================================
# 4. HÀM CHÍNH
# =============================================================================
def train_swav_encoder(
        data_path: Union[str, Path],
        dataset_name: str,
        backbone_type: str,
        in_channels: int,
        seq_len: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        num_workers: int,
        device: str,
        measure_complexity: bool = True,
        tag: str = "",
) -> None:
    device = str(device)
    # Tất cả output nằm CÙNG 1 thư mục: checkpoints/ssl_pretrain/swav/<dataset>/<backbone>_<tag>/
    ckpt_dir = ssl_checkpoint_dir(get_output_root(), METHOD_NAME, dataset_name, backbone_type, tag=tag)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = ssl_checkpoint_name(METHOD_NAME, backbone_type, dataset_name)
    checkpoint_path = ckpt_dir / checkpoint_name

    # ---- Dựng dữ liệu, model, loss, optimizer ------------------------------------------
    loader = _build_loader(data_path, batch_size, num_workers, device)
    total_samples = len(loader.dataset)
    model = _build_model(backbone_type, in_channels, device)
    criterion = _build_criterion(device)
    optimizer = _build_optimizer(model, learning_rate)

    # ---- Header + lưu config (config.json ghi lại đúng những gì sẽ chạy) ----------------
    rep.print_header(dataset_name, data_path, ckpt_dir, backbone_type, epochs, batch_size)
    rep.save_json(
        rep.build_config(
            data_path=data_path, total_samples=total_samples, in_channels=in_channels,
            seq_len=seq_len, backbone_type=backbone_type, batch_size=batch_size,
            learning_rate=learning_rate, epochs=epochs, num_workers=num_workers,
            save_dir=ckpt_dir, target_dir=ckpt_dir, checkpoint_name=checkpoint_name,
            device=device,
        ),
        ckpt_dir / "config.json",
    )

    # ---- Theo dõi sụp cụm ---------------------------------------------------------------
    # Entropy tối đa = ln K (các cụm dùng đều tuyệt đối). Thấp hơn ngưỡng -> cảnh báo.
    # [TÙY CHỈNH] Ngưỡng: SwAVConfig.MAX_ENTROPY_COLLAPSE_THRESHOLD (tỉ lệ so với ln K).
    max_entropy = math.log(SwAVConfig.NUM_PROTOTYPES)
    collapse_floor = SwAVConfig.MAX_ENTROPY_COLLAPSE_THRESHOLD * max_entropy

    best_loss, saved, history = float("inf"), False, []

    # ---- Vòng lặp epoch -----------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # (1) LR của epoch này, gán cho MỌI nhóm tham số (cả nhóm decay và no_decay).
        lr = _lr_at_epoch(epoch, epochs, learning_rate, SwAVConfig.WARMUP_EPOCHS)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Prototype đóng băng trong FREEZE_PROTOTYPES_EPOCHS epoch đầu.
        frozen = epoch <= SwAVConfig.FREEZE_PROTOTYPES_EPOCHS

        row = {"epoch": epoch,
               **_train_one_epoch(model, criterion, loader, optimizer, device, frozen),
               "lr": lr}
        history.append(row)

        # Cảnh báo sụp cụm: phân phối cụm trung bình quá lệch.
        if row["entropy"] < collapse_floor:
            print(f"⚠️  Cảnh báo sụp cụm: Entropy={row['entropy']:.2f} < {collapse_floor:.2f} tại epoch {epoch}")

        # Chọn checkpoint "tốt nhất" theo loss, chỉ xét SAU warmup (loss lúc warmup chưa đại diện).
        # Chỉ lưu ENCODER (không lưu prototype/projection): đây là thứ fine-tune cần.
        if epoch > SwAVConfig.WARMUP_EPOCHS and row["loss_total"] < best_loss:
            best_loss, saved = row["loss_total"], True
            torch.save(model.encoder.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            rep.log_epoch(epoch, epochs, row, best_loss if saved else row["loss_total"], max_entropy)

    # ---- Fallback: epochs <= warmup nên chưa lưu được -> lưu epoch cuối --------------------
    if not saved:
        best_loss = history[-1]["loss_total"]
        torch.save(model.encoder.state_dict(), checkpoint_path)
        print("   ⚠️ Fallback: epochs <= warmup, đã lưu checkpoint ở epoch cuối.")

    # ---- Lưu kết quả + tóm tắt -------------------------------------------------------------
    rep.save_training_outputs(
        history, ckpt_dir, checkpoint_path, backbone_type, dataset_name,
        in_channels, seq_len, device, measure_complexity,
    )
    rep.print_summary(checkpoint_path, best_loss, total_samples)

    del model, criterion, optimizer, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
