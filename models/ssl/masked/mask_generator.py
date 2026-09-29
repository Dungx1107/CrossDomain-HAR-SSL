import torch
import torch.nn as nn


class SegmentMaskGenerator(nn.Module):
    """
    Sinh binary mask che các phân đoạn liên tục (continuous segments) trên chuỗi thời gian.
    Giá trị True (1) là vị trí bị mask (cần tái tạo), False (0) là giữ nguyên.
    Hỗ trợ cả dạng 3D (B, C, L) cho 1D-CNN và 4D (B, C, L, 1) cho 2D-CNN.
    """

    def __init__(self, mask_ratio: float = 0.15, max_segment_len: int = 16, min_segment_len: int = 4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.max_segment_len = max_segment_len
        self.min_segment_len = min_segment_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            b, c, l, _ = x.shape
            mask_1d = self._generate_mask(b, l, x.device)
            return mask_1d.unsqueeze(1).unsqueeze(-1).repeat(1, c, 1, 1)
        elif x.dim() == 3:
            b, c, l = x.shape
            mask_1d = self._generate_mask(b, l, x.device)
            return mask_1d.unsqueeze(1).repeat(1, c, 1)
        else:
            raise ValueError(f"Tensor đầu vào phải có 3 hoặc 4 chiều, nhận được: {x.dim()} chiều")

    def _generate_mask(self, batch_size: int, length: int, device: torch.device) -> torch.Tensor:
        num_mask_points = int(length * self.mask_ratio)
        mask = torch.zeros((batch_size, length), dtype=torch.bool, device=device)

        for i in range(batch_size):
            masked_indices = set()
            attempts = 0
            max_attempts = 200

            while len(masked_indices) < num_mask_points and attempts < max_attempts:
                attempts += 1
                remaining = num_mask_points - len(masked_indices)

                # Độ dài đoạn không vượt quá số điểm còn thiếu và không vượt max_segment_len
                upper_len = min(self.max_segment_len, remaining)
                lower_len = min(self.min_segment_len, upper_len)

                if upper_len < 1:
                    break

                seg_len = torch.randint(lower_len, upper_len + 1, (1,)).item()
                start_idx = torch.randint(0, length - seg_len + 1, (1,)).item()
                new_indices = set(range(start_idx, start_idx + seg_len))

                # Nếu bị trùng lặp với các vị trí đã mask trước đó thì bỏ qua để thử lại
                if new_indices & masked_indices:
                    continue

                masked_indices |= new_indices

            # Nếu chạm trần attempts mà vẫn chưa đủ, lấp đầy các vị trí còn trống ngẫu nhiên
            if len(masked_indices) < num_mask_points:
                unmasked = list(set(range(length)) - masked_indices)
                needed = num_mask_points - len(masked_indices)
                fallback = torch.randperm(len(unmasked))[:needed].tolist()
                for idx in fallback:
                    masked_indices.add(unmasked[idx])

            for idx in masked_indices:
                mask[i, idx] = True

        return mask