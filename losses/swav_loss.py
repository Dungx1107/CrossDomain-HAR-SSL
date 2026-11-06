"""
===============================================================================
MODULE: SWAV LOSS — Sinkhorn-Knopp + Swapped Prediction + giám sát sụp cụm
===============================================================================
VAI TRÒ
    Nhận 2 ma trận cosine (B, K) của 2 view (weak / strong) và trả về:
        loss    : swapped-prediction loss            (CÓ gradient)
        entropy : entropy phân phối cụm TRUNG BÌNH   (chỉ để theo dõi, KHÔNG gradient)

    File này KHÔNG chứa tham số học được và KHÔNG biết gì về backbone hay prototype.
    (prototype nằm ở models/ssl/swav/swav_model.py; lịch học nằm ở swav_trainer.py)

KÝ HIỆU
    B = batch size | K = số prototype | cos_* có shape (B, K), giá trị trong [-1, 1]

LUỒNG TÍNH (đối chiếu bài báo SwAV, NeurIPS 2020)
    cos_w, cos_s --Sinkhorn(cos/ε)--> q_w, q_s   "mã" gán cụm mục tiêu, KHÔNG gradient   [Eq. 3, 5]
    cos_w, cos_s --softmax(cos/τ) --> p_w, p_s   dự đoán của mạng,       CÓ gradient     [Eq. 2]
    loss = ½ · [ CE(q_s, p_w) + CE(q_w, p_s) ]   view này dự đoán mã của view kia        [Eq. 1]
        với CE(q, p) = -Σ_k q_k · log p_k

HAI NHIỆT ĐỘ — KHÔNG ĐƯỢC TRỘN LẪN (đây từng là nguồn lỗi tràn số)
    ε (epsilon) : CHỈ dùng trong Sinkhorn:      exp(cos / ε)
    τ (tau)     : CHỈ dùng trong cross-entropy: softmax(cos / τ)
    Cả hai đều nhận cosine THÔ. Tuyệt đối không chia τ trước rồi đưa vào Sinkhorn.

CHÚ GIẢI THẺ TRONG COMMENT
    [ĐỐI CHIẾU]  : tương ứng bài báo / mã chính thức facebookresearch/swav (main_swav.py)
    [TÙY CHỈNH]  : chỗ bạn có thể thay đổi
    [LƯU Ý]      : bẫy hay gặp
    [MỞ RỘNG]    : chỗ gắn thêm tính năng (queue, multi-crop)
===============================================================================
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. SINKHORN-KNOPP — sinh "mã" q: gán mỗi mẫu vào các cụm, ép các cụm được dùng đều
# =============================================================================
# [ĐỐI CHIẾU] Tương ứng hàm distributed_sinkhorn() trong main_swav.py (bản gốc có thêm
#             all_reduce cho nhiều GPU; ở đây 1 GPU nên bỏ). Dạng viết cố ý giống bản gốc
#             (Q /= sum; Q /= K ...) để bạn đặt hai file cạnh nhau mà đọc.
@torch.no_grad()  # [LƯU Ý] q là nhãn giả "mục tiêu": KHÔNG được có gradient chảy qua.
def sinkhorn(cos: torch.Tensor, epsilon: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """
    Args:
        cos       : (B, K) cosine THÔ giữa embedding và prototype (chưa chia nhiệt độ nào).
        epsilon   : nhiệt độ Sinkhorn ε.
        n_iters   : số vòng lặp chuẩn hóa hàng/cột (bài báo dùng 3).
    Returns:
        q : (B, K); mỗi hàng (mỗi mẫu) có tổng = 1, tức một phân phối xác suất trên K cụm.
    """
    # --- Bước 0: ổn định số học ------------------------------------------------
    # Trừ max TOÀN CỤC trước khi exp. Hệ số hằng này bị triệt tiêu ở bước "Q /= Q.sum()"
    # bên dưới nên KHÔNG đổi kết quả, chỉ tránh exp() tràn số.
    # [ĐỐI CHIẾU] bản gốc không trừ max vì cos/ε <= 1/0.05 = 20 => exp(20) ≈ 4.8e8 vẫn an toàn;
    #             trừ max chỉ là lớp bảo hiểm thêm.
    cos = cos - cos.max()

    # --- Bước 1: ma trận "độ phù hợp" dương -------------------------------------
    # Q[k, b] = exp(cos[b, k] / ε). Chuyển vị thành (K, B) giống ký hiệu bài báo (Eq. 5).
    # [TÙY CHỈNH] ε nhỏ  -> Q gần one-hot (gán cứng);  ε lớn -> Q gần đều (gán mềm, kém thông tin).
    Q = torch.exp(cos / epsilon).t()
    K, B = Q.shape

    # --- Bước 2: chuẩn hóa để toàn ma trận có tổng = 1 --------------------------
    # Coi Q là phân phối đồng thời (cụm, mẫu).
    Q /= Q.sum()

    # --- Bước 3: lặp chuẩn hóa hàng / cột ---------------------------------------
    # Mục tiêu: mỗi cụm nhận tổng khối lượng 1/K, mỗi mẫu có tổng khối lượng 1/B.
    # Ràng buộc "mỗi cụm 1/K" CHÍNH LÀ thứ chống sụp cụm (không thể dồn mọi mẫu vào 1 cụm).
    # [LƯU Ý] 3 vòng chỉ cho nghiệm XẤP XỈ: khi điểm số rất sắc, tải mỗi cụm vẫn lệch khoảng
    #         0.65–1.27 lần so với B/K (đã đo bằng số). Tăng n_iters để ép đều hơn.
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True)   # mỗi hàng (cụm) tổng = 1 ...
        Q /= K                            # ... rồi chia K  -> mỗi cụm tổng 1/K
        Q /= Q.sum(dim=0, keepdim=True)   # mỗi cột (mẫu) tổng = 1 ...
        Q /= B                            # ... rồi chia B  -> mỗi mẫu tổng 1/B

    # --- Bước 4: đưa về phân phối theo từng mẫu ---------------------------------
    # Sau vòng lặp mỗi cột tổng 1/B; nhân B để mỗi cột (mẫu) tổng = 1.
    Q *= B
    return Q.t()  # (B, K)


# =============================================================================
# 2. ENTROPY SỬ DỤNG CỤM — công cụ phát hiện sụp cụm
# =============================================================================
@torch.no_grad()
def cluster_usage_entropy(prob: torch.Tensor) -> torch.Tensor:
    """
    Entropy của phân phối cụm TRUNG BÌNH theo batch.

    Args:
        prob : (B, K) xác suất dự đoán (mỗi hàng tổng 1).
    Returns:
        Scalar trong [0, ln K]. ln(45) ≈ 3.81.

    Cách đọc:
        - Gần ln K : các cụm được dùng đều               -> khỏe.
        - Thấp     : phần lớn mẫu dồn vào ít cụm         -> SỤP CỤM.
    [LƯU Ý] Đây KHÔNG phải trung bình entropy từng mẫu. Entropy từng mẫu thấp chỉ nghĩa là
            mạng "tự tin", có thể đang rất khỏe, nên không dùng để phát hiện sụp cụm.
    """
    avg = prob.mean(dim=0)  # (K,) tỉ lệ dùng mỗi cụm, trung bình trên batch
    return -(avg * torch.log(avg + 1e-8)).sum()  # 1e-8 tránh log(0)


# =============================================================================
# 3. LOSS CHÍNH
# =============================================================================
class SwAVLoss(nn.Module):
    """
    Swapped prediction loss cho 2 view.

    [MỞ RỘNG - multi-crop] Bản gốc nhận NHIỀU view: mã q chỉ tính từ vài view lớn
        (crops_for_assign), nhưng MỌI view khác đều phải dự đoán mã đó. Bài toán 2 view
        của bạn là ca đặc biệt: view 0 dạy view 1 và ngược lại. Muốn mở rộng, đổi forward()
        nhận danh sách cos và lặp theo công thức:
            loss = mean_i [ mean_{j != i} CE(q_i, p_j) ]   với i thuộc các view lớn.
    [MỞ RỘNG - queue] Queue chèn ngay TRƯỚC lệnh sinkhorn(): nối thêm điểm số của các
        embedding cũ vào cos, chạy Sinkhorn trên (B + queue), rồi chỉ lấy B dòng cuối làm q.
        Cần model trả thêm embedding (xem SwAVModel.embed).
    """

    def __init__(self, tau: float = 0.1, epsilon: float = 0.05, sinkhorn_iters: int = 3):
        super().__init__()
        self.tau = tau                        # τ  : nhiệt độ của softmax dự đoán (Eq. 2)
        self.epsilon = epsilon                # ε  : nhiệt độ của Sinkhorn (Eq. 5)
        self.sinkhorn_iters = sinkhorn_iters  # số vòng lặp Sinkhorn

    def forward(self, cos_w: torch.Tensor, cos_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cos_w : (B, K) cosine THÔ của view weak   (từ SwAVModel.forward)
            cos_s : (B, K) cosine THÔ của view strong
        Returns:
            loss    : scalar, có gradient (chảy về backbone, projection head, prototype).
            entropy : scalar, không gradient, để theo dõi sụp cụm.
        """
        # --- (a) MÃ MỤC TIÊU q: Sinkhorn trên cosine THÔ, KHÔNG gradient -------------
        q_w = sinkhorn(cos_w, self.epsilon, self.sinkhorn_iters)  # (B, K)
        q_s = sinkhorn(cos_s, self.epsilon, self.sinkhorn_iters)  # (B, K)

        # --- (b) DỰ ĐOÁN p: softmax(cos / τ), CÓ gradient ---------------------------
        # Dùng log_softmax (ổn định hơn log(softmax(...))).
        # [TÙY CHỈNH] τ nhỏ -> dự đoán sắc, gradient mạnh vào cụm đang thắng; τ lớn -> mềm hơn.
        log_p_w = F.log_softmax(cos_w / self.tau, dim=1)  # (B, K)
        log_p_s = F.log_softmax(cos_s / self.tau, dim=1)  # (B, K)

        # --- (c) SWAPPED PREDICTION (Eq. 1) -----------------------------------------
        # View weak dự đoán mã của view strong, và ngược lại.
        # (q * log_p).sum(dim=1) = cross-entropy theo từng mẫu; .mean() lấy trung bình theo batch.
        loss_w_predicts_s = -(q_s * log_p_w).sum(dim=1).mean()
        loss_s_predicts_w = -(q_w * log_p_s).sum(dim=1).mean()
        # [ĐỐI CHIẾU] Bản gốc: mỗi crop_for_assign cộng sub-loss rồi chia (số crop - 1) và chia
        #             số crop_for_assign; với 2 view điều đó đúng bằng phép lấy trung bình này.
        loss = 0.5 * (loss_w_predicts_s + loss_s_predicts_w)

        # --- (d) GIÁM SÁT SỤP CỤM (không gradient) ----------------------------------
        # Phân phối dự đoán trung bình của hai view rồi tính entropy sử dụng cụm.
        entropy = cluster_usage_entropy(0.5 * (log_p_w.exp() + log_p_s.exp()))

        return loss, entropy


# =============================================================================
# SELF-TEST: python losses/swav_loss.py
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, K = 64, 45

    # Test 1: q là phân phối hợp lệ trên K cụm cho từng mẫu
    cos = torch.empty(B, K).uniform_(-0.1, 0.1)
    q = sinkhorn(cos)
    assert q.shape == (B, K), "shape q sai"
    assert torch.allclose(q.sum(dim=1), torch.ones(B), atol=1e-4), "mỗi mẫu phải có tổng xác suất = 1"
    assert (q >= 0).all(), "q có phần tử âm"
    print(f"Test 1 OK | tải mỗi cụm / (B/K): [{(q.sum(0).min() / (B / K)):.2f}, {(q.sum(0).max() / (B / K)):.2f}]")

    # Test 2: cosine rất sắc (khi huấn luyện đã hội tụ) vẫn hữu hạn — chống lại lỗi tràn số cũ
    sharp = torch.full((B, K), -0.1)
    sharp[torch.arange(B), torch.randint(0, K, (B,))] = 0.95
    assert torch.isfinite(sinkhorn(sharp)).all(), "Sinkhorn ra NaN/inf với cosine sắc"
    print("Test 2 OK | hữu hạn với cosine sắc")

    # Test 3: loss chạy được, gradient chảy về cos, và q KHÔNG có gradient
    cos_w = torch.empty(B, K).uniform_(-1, 1).requires_grad_(True)
    cos_s = torch.empty(B, K).uniform_(-1, 1).requires_grad_(True)
    criterion = SwAVLoss(tau=0.1, epsilon=0.05, sinkhorn_iters=3)
    loss, entropy = criterion(cos_w, cos_s)
    loss.backward()
    assert torch.isfinite(loss) and cos_w.grad is not None and cos_s.grad is not None
    assert not sinkhorn(cos_w.detach()).requires_grad
    print(f"Test 3 OK | loss={loss.item():.4f}, entropy={entropy.item():.3f} (tối đa ln K = {torch.log(torch.tensor(float(K))).item():.3f})")
    print("✅ swav_loss OK")