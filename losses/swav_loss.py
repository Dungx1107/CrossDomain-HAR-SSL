import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_knopp(scores: torch.Tensor, epsilon: float = 0.05, niters: int = 3) -> torch.Tensor:
    """
    Sinkhorn-Knopp nguyên bản từ FAIR (SwAV).
    Trả về ma trận phân bổ cụm mềm Q (B, K).
    """
    Q = torch.exp(scores / epsilon).t()   # (K, B)
    K, B = Q.shape

    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for _ in range(niters):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()   # (B, K)


class SwAVPrototypeLoss(nn.Module):
    """
    Swapped Prediction loss với tùy chọn trả về Q để log entropy.
    """
    def __init__(self, epsilon: float = 0.05):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, scores_w, scores_s, return_q: bool = False):
        q_w = sinkhorn_knopp(scores_w, epsilon=self.epsilon)
        q_s = sinkhorn_knopp(scores_s, epsilon=self.epsilon)

        p_w = F.log_softmax(scores_w, dim=1)
        p_s = F.log_softmax(scores_s, dim=1)

        loss_w = -torch.mean(torch.sum(q_s * p_w, dim=1))
        loss_s = -torch.mean(torch.sum(q_w * p_s, dim=1))

        loss = (loss_w + loss_s) * 0.5
        if return_q:
            return loss, q_w
        return loss