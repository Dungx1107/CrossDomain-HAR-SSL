import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, epsilon: float = 0.05, niters: int = 3) -> torch.Tensor:
    """
    Sinkhorn-Knopp với bảo vệ tràn số (Numerical Stability).
    """
    # 1. Ổn định số học: Trừ max theo từng hàng trước khi exp
    scores_stable = scores / epsilon
    scores_stable = scores_stable - torch.max(scores_stable, dim=1, keepdim=True)[0]

    Q = torch.exp(scores_stable).t()  # (K, B)
    K, B = Q.shape

    # Tránh chia cho 0
    sum_Q = torch.sum(Q)
    Q = torch.clamp(Q / (sum_Q + 1e-12), min=1e-12)

    for _ in range(niters):
        # Chuẩn hóa hàng
        row_sum = torch.sum(Q, dim=1, keepdim=True)
        Q = Q / (row_sum + 1e-12)
        Q = Q / K

        # Chuẩn hóa cột
        col_sum = torch.sum(Q, dim=0, keepdim=True)
        Q = Q / (col_sum + 1e-12)
        Q = Q / B

    Q = Q * B
    return Q.t()  # (B, K)


class SwAVPrototypeLoss(nn.Module):
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, scores_w: torch.Tensor, scores_s: torch.Tensor, return_q: bool = False):
        q_w = sinkhorn_knopp(scores_w, epsilon=self.epsilon)
        q_s = sinkhorn_knopp(scores_s, epsilon=self.epsilon)

        p_w = F.log_softmax(scores_w, dim=1)
        p_s = F.log_softmax(scores_s, dim=1)

        # Tránh nhân NaN/Inf nếu có giá trị cực tiểu
        loss_w = -torch.mean(torch.sum(q_w.detach() * p_s, dim=1))
        loss_s = -torch.mean(torch.sum(q_s.detach() * p_w, dim=1))

        loss = (loss_w + loss_s) * 0.5
        if return_q:
            return loss, q_w
        return loss

if __name__ == "__main__":
    # Test 1: Sinkhorn output là phân phối hợp lệ
    B, K = 16, 45
    scores = torch.randn(B, K) * 5.0  # scores ngẫu nhiên
    Q = sinkhorn_knopp(scores, epsilon=0.05, niters=3)

    print("Shape Q:", Q.shape)  # Phải là (16, 45)
    print("Tổng mỗi hàng:", Q.sum(dim=1)[:5])  # Phải ≈ 1.0
    print("Tổng mỗi cột:", Q.sum(dim=0)[:5])  # Phải ≈ B/K = 16/45 ≈ 0.356
    print("Min/Max Q:", Q.min().item(), Q.max().item())  # Phải > 0

    assert Q.shape == (B, K), "Shape sai"
    assert torch.allclose(Q.sum(dim=1), torch.ones(B), atol=1e-3), "Hàng không tổng = 1"
    assert (Q > 0).all(), "Có phần tử âm hoặc bằng 0"

    # Test 2: Loss chạy được, backprop OK
    criterion = SwAVPrototypeLoss(epsilon=0.05)
    scores_w = torch.randn(B, K, requires_grad=True)
    scores_s = torch.randn(B, K, requires_grad=True)
    loss, q_w = criterion(scores_w, scores_s, return_q=True)

    print("\nLoss:", loss.item())
    print("q_w shape:", q_w.shape)

    loss.backward()
    print("Gradient scores_w tồn tại:", scores_w.grad is not None)
    print("Gradient scores_s tồn tại:", scores_s.grad is not None)

    print("\n✅ FILE 1 OK")