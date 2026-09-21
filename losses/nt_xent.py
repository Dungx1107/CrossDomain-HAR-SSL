import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent / InfoNCE loss cho hai views của cùng một batch.

    Input:
        h_weak   : (B, D)
        h_strong : (B, D)

    Với mỗi anchor:
        positive = view còn lại của cùng sample
        negatives = tất cả views của các sample khác
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_weak, h_strong):
        batch_size = h_weak.size(0)

        # L2-normalize để dot product trở thành cosine similarity
        h_weak = F.normalize(h_weak, dim=1)
        h_strong = F.normalize(h_strong, dim=1)

        # (2B, D)
        representations = torch.cat(
            [h_weak, h_strong],
            dim=0
        )

        # (2B, 2B)
        similarity = torch.mm(
            representations,
            representations.t()
        ) / self.temperature

        # Loại self-similarity
        self_mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=representations.device
        )

        similarity = similarity.masked_fill(
            self_mask,
            torch.finfo(similarity.dtype).min
        )

        # Positive mapping:
        #
        # 0 ... B-1       -> B ... 2B-1
        # B ... 2B-1      -> 0 ... B-1
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size,
                         device=representations.device),
            torch.arange(0, batch_size,
                         device=representations.device)
        ])

        loss = self.criterion(similarity, targets)

        return loss
