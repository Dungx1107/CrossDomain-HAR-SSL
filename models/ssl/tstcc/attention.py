"""
===============================================================================
MODULE: ATTENTION & TRANSFORMER FOR TS-TCC
===============================================================================
Mã nguồn gốc từ file `models/attention.py` của repository `emadeldeen24/TS-TCC`.

Module triển khai Transformer với các thành phần:
    - Residual: Kết nối phần dư (Residual Connection).
    - PreNorm: Chuẩn hóa LayerNorm trước khi thực hiện phép biến đổi.
    - FeedForward: Mạng truyền thẳng xử lý đặc trưng.
    - Attention: Cơ chế Multi-Head Self-Attention.
    - Transformer: Kết hợp Attention và FeedForward qua nhiều lớp.
    - Seq_Transformer: Thêm c_token để tổng hợp thông tin chuỗi thành
      Context Representation c_t.

Đầu vào của Seq_Transformer là chuỗi đặc trưng theo thời gian.
Đầu ra c_t là một vector ngữ cảnh tổng hợp thông tin từ chuỗi.

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class Residual(nn.Module):
    """
    Kết nối phần dư (Residual Connection).

    Cho đầu vào x đi qua hàm fn, sau đó cộng kết quả với x:

        output = fn(x) + x

    Mục đích:
        - Giúp thông tin từ đầu vào được truyền trực tiếp đến đầu ra.
        - Hỗ trợ lan truyền gradient khi huấn luyện mạng sâu.
        - Hạn chế khó khăn trong quá trình tối ưu hóa.

    Lưu ý:
        fn(x) phải có cùng kích thước tensor với x để phép cộng thực hiện
        được.
    """

    def __init__(self, fn):
        super().__init__()

        # Hàm biến đổi được sử dụng bên trong kết nối phần dư.
        # Trong Transformer, fn thường là Attention hoặc FeedForward.
        self.fn = fn

    def forward(self, x, **kwargs):
        # Thực hiện biến đổi trên x rồi cộng kết quả với đầu vào ban đầu.
        # **kwargs cho phép truyền thêm các tham số như mask.
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """
    Chuẩn hóa đầu vào trước khi đưa qua một khối xử lý.

    Luồng xử lý:

        x -> LayerNorm -> fn -> output

    LayerNorm chuẩn hóa các đặc trưng trên chiều cuối cùng của tensor.
    Với đầu vào có dạng (batch, sequence_length, dim), LayerNorm(dim)
    chuẩn hóa dim đặc trưng của từng vị trí trong chuỗi.

    PreNorm được sử dụng trước Attention và FeedForward trong Transformer.
    """

    def __init__(self, dim, fn):
        super().__init__()

        # Chuẩn hóa vector đặc trưng có số chiều dim.
        self.norm = nn.LayerNorm(dim)

        # Khối xử lý tiếp theo, chẳng hạn Attention hoặc FeedForward.
        self.fn = fn

    def forward(self, x, **kwargs):
        # Chuẩn hóa x trước khi đưa vào hàm fn.
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Mạng truyền thẳng (FeedForward Network - FFN).

    Kiến trúc:

        Linear(dim -> hidden_dim)
        -> ReLU
        -> Dropout
        -> Linear(hidden_dim -> dim)
        -> Dropout

    Vai trò:
        - Học các phép biến đổi phi tuyến trên đặc trưng.
        - Xử lý thông tin sau khi Attention đã tổng hợp thông tin
          giữa các vị trí trong chuỗi.

    Đầu vào và đầu ra có cùng số chiều dim để tương thích với
    Residual Connection.
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            # Mở rộng số chiều đặc trưng từ dim lên hidden_dim.
            nn.Linear(dim, hidden_dim),

            # Hàm kích hoạt phi tuyến.
            nn.ReLU(),

            # Ngẫu nhiên loại bỏ một phần neuron trong quá trình huấn luyện
            # nhằm hạn chế overfitting.
            nn.Dropout(dropout),

            # Chiếu đặc trưng trở lại số chiều ban đầu.
            # Điều này giúp kết quả có thể cộng với đầu vào qua Residual.
            nn.Linear(hidden_dim, dim),

            # Dropout sau lớp Linear cuối cùng.
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Xử lý độc lập trên chiều đặc trưng của từng vị trí trong chuỗi.
        return self.net(x)


class Attention(nn.Module):
    """
    Cơ chế Multi-Head Self-Attention.

    Mỗi vị trí trong chuỗi tạo ra ba vector:
        Q (Query): Đại diện cho thông tin đang cần tìm kiếm.
        K (Key):   Đại diện cho thông tin dùng để tính mức độ liên quan.
        V (Value): Thông tin được tổng hợp dựa trên trọng số Attention.

    Quy trình chính:

        1. Chiếu đầu vào thành Q, K, V.
        2. Chia Q, K, V thành nhiều attention head.
        3. Tính độ tương đồng giữa Q và K.
        4. Áp dụng Softmax để thu được trọng số Attention.
        5. Dùng trọng số kết hợp các vector V.
        6. Ghép các head và chiếu về số chiều ban đầu.

    Kích thước đầu vào:
        x: (batch_size, sequence_length, dim)

    Kích thước đầu ra:
        out: (batch_size, sequence_length, dim)
    """

    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()

        # Số attention head.
        self.heads = heads

        # Hệ số scale cho tích vô hướng giữa Q và K.
        # Công thức chuẩn thường dùng:
        #     Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        #
        # Lưu ý: code gốc sử dụng dim ** -0.5, tức 1/sqrt(dim),
        # thay vì 1/sqrt(dim_per_head).
        self.scale = dim ** -0.5

        # Một lớp Linear duy nhất tạo đồng thời Q, K, V.
        # Đầu ra có kích thước 3 * dim để sau đó chia thành ba phần.
        # bias=False nghĩa là lớp Linear không sử dụng bias.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        # Chiếu kết quả sau khi ghép các head về dim chiều.
        # Dropout giúp giảm overfitting.
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x có dạng (b, n, dim):
        #   b: batch_size
        #   n: số vị trí trong chuỗi
        #   dim: số chiều đặc trưng
        #
        # h là số attention head.
        b, n, _, h = *x.shape, self.heads

        # Tạo Q, K, V bằng một phép chiếu tuyến tính.
        # chunk(3, dim=-1) chia chiều cuối thành ba phần bằng nhau.
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Chuyển Q, K, V từ:
        #   (b, n, dim)
        # thành:
        #   (b, h, n, d)
        #
        # Trong đó d là số chiều đặc trưng của mỗi head.
        #
        # Điều kiện ngầm định: dim phải chia hết cho heads.
        q, k, v = map(
            lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h=h
            ),
            qkv
        )

        # Tính độ tương đồng giữa Query và Key.
        #
        # q: (b, h, n, d)
        # k: (b, h, n, d)
        #
        # dots: (b, h, n, n)
        #
        # dots[b, h, i, j] biểu thị mức độ liên quan giữa vị trí i
        # và vị trí j trong chuỗi, đối với head h.
        #
        # torch.einsum thực hiện tích vô hướng giữa q_i và k_j.
        dots = torch.einsum(
            'bhid,bhjd->bhij', q, k
        ) * self.scale

        # Nếu có mask, sử dụng mask để loại bỏ các vị trí không hợp lệ
        # khỏi quá trình tính Attention.
        if mask is not None:

            # Làm phẳng các chiều cuối của mask thành một chiều.
            # Thêm True ở đầu để tương ứng với vị trí c_token.
            mask = F.pad(
                mask.flatten(1),
                (1, 0),
                value=True
            )

            # Kiểm tra số vị trí của mask có khớp với Attention hay không.
            assert mask.shape[-1] == dots.shape[-1], \
                'mask has incorrect dimensions'

            # Tạo mask hai chiều:
            # mask[i, j] chỉ hợp lệ khi cả vị trí i và j đều hợp lệ.
            #
            # Thêm chiều để broadcast với tensor dots có dạng (b, h, n, n).
            mask = mask[:, None, :] * mask[:, :, None]

            # Các cặp vị trí không hợp lệ được gán -inf.
            # Sau Softmax, trọng số tương ứng sẽ bằng 0.
            dots.masked_fill_(~mask, float('-inf'))

            # Giải phóng tham chiếu tới tensor mask tạm thời.
            del mask

        # Chuẩn hóa điểm tương đồng thành trọng số Attention.
        # Tổng trọng số trên chiều các vị trí được chú ý bằng 1.
        attn = dots.softmax(dim=-1)

        # Kết hợp các vector V theo trọng số Attention.
        #
        # attn: (b, h, n, n)
        # v:    (b, h, n, d)
        # out:  (b, h, n, d)
        #
        # Mỗi vị trí nhận được thông tin tổng hợp từ các vị trí khác.
        out = torch.einsum(
            'bhij,bhjd->bhid', attn, v
        )

        # Ghép các head trở lại:
        # (b, h, n, d) -> (b, n, h*d)
        #
        # Vì h*d = dim nên kết quả có dạng (b, n, dim).
        out = rearrange(
            out, 'b h n d -> b n (h d)'
        )

        # Chiếu kết quả về không gian đặc trưng đầu ra.
        out = self.to_out(out)

        return out


class Transformer(nn.Module):
    """
    Transformer gồm nhiều lớp xử lý tuần tự.

    Mỗi lớp bao gồm hai khối:

        1. Multi-Head Self-Attention:
           Tổng hợp thông tin giữa các vị trí trong chuỗi.

        2. FeedForward:
           Biến đổi phi tuyến đặc trưng tại từng vị trí.

    Cả hai khối đều sử dụng:
        PreNorm + Residual Connection

    Luồng xử lý của một lớp:

        x
        -> LayerNorm -> Attention -> cộng residual
        -> LayerNorm -> FeedForward -> cộng residual

    Các lớp được xếp chồng lên nhau theo tham số depth.
    """

    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        # ModuleList lưu trữ các lớp Transformer.
        # Khác với list Python thông thường, các module trong ModuleList
        # được PyTorch đăng ký để quản lý tham số và huấn luyện.
        self.layers = nn.ModuleList([])

        for _ in range(depth):

            # Mỗi lớp có hai khối:
            #   - Attention với PreNorm và Residual.
            #   - FeedForward với PreNorm và Residual.
            self.layers.append(
                nn.ModuleList([
                    Residual(
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dropout=dropout
                            )
                        )
                    ),
                    Residual(
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout=dropout
                            )
                        )
                    )
                ])
            )

    def forward(self, x, mask=None):

        # Đưa tensor lần lượt qua từng lớp Transformer.
        for attn, ff in self.layers:

            # Bước 1: Self-Attention tổng hợp thông tin theo chuỗi.
            # Residual bên ngoài tự động cộng kết quả với đầu vào.
            x = attn(x, mask=mask)

            # Bước 2: FeedForward biến đổi đặc trưng.
            # Tiếp tục sử dụng PreNorm và Residual.
            x = ff(x)

        return x


class Seq_Transformer(nn.Module):
    """
    Transformer dùng c_token để trích xuất Context Representation c_t.

    Đầu vào:
        forward_seq: Tensor biểu diễn chuỗi đặc trưng theo thời gian.

    Đầu ra:
        c_t: Vector ngữ cảnh tổng hợp thông tin của chuỗi.

    Quy trình:

        1. Chiếu đặc trưng đầu vào sang không gian dim chiều.
        2. Thêm c_token vào đầu chuỗi.
        3. Đưa chuỗi qua Transformer.
        4. Lấy vector tại vị trí c_token làm biểu diễn ngữ cảnh c_t.

    c_token là một tham số có thể học được. Trong quá trình huấn luyện,
    nó học cách tổng hợp thông tin từ các vị trí trong chuỗi thông qua
    cơ chế Self-Attention.

    Lưu ý:
        c_token không phải đặc trưng đầu vào được đo trực tiếp từ cảm biến.
        Đây là một vector học được, đóng vai trò tổng hợp thông tin.
    """

    def __init__(
        self,
        *,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=1,
        dropout=0.1
    ):
        super().__init__()

        # Số chiều của một patch đầu vào.
        # Giả sử mỗi patch có patch_size giá trị trên mỗi channel:
        # patch_dim = channels * patch_size.
        patch_dim = channels * patch_size

        # Chiếu mỗi patch từ patch_dim chiều sang dim chiều.
        # Lớp Linear tác động lên chiều cuối của tensor.
        self.patch_to_embedding = nn.Linear(
            patch_dim, dim
        )

        # Khởi tạo c_token với kích thước (1, 1, dim).
        #
        # torch.randn khởi tạo các giá trị ngẫu nhiên theo phân phối chuẩn.
        # nn.Parameter giúp c_token trở thành tham số có thể học được.
        #
        # Hai chiều đầu là:
        #   1: batch giả, sẽ được mở rộng theo batch thực tế.
        #   1: vị trí đặc biệt của c_token trong chuỗi.
        self.c_token = nn.Parameter(
            torch.randn(1, 1, dim)
        )

        # Transformer dùng để học quan hệ giữa c_token và các vị trí
        # đặc trưng trong chuỗi.
        self.transformer = Transformer(
            dim, depth, heads, mlp_dim, dropout
        )

        # Identity không thực hiện biến đổi nào.
        # Được giữ nguyên theo thiết kế của mã nguồn gốc.
        self.to_c_token = nn.Identity()

    def forward(self, forward_seq):

        # Bước 1: Chiếu đặc trưng đầu vào sang không gian dim chiều.
        #
        # Nếu đầu vào có dạng (b, n, patch_dim), kết quả có dạng:
        # x: (b, n, dim)
        x = self.patch_to_embedding(forward_seq)

        # Lấy kích thước batch và số vị trí trong chuỗi.
        b, n, _ = x.shape

        # Bước 2: Sao chép c_token theo batch.
        #
        # c_token ban đầu: (1, 1, dim)
        # c_tokens sau repeat: (b, 1, dim)
        #
        # Mỗi mẫu trong batch có một c_token với cùng giá trị khởi tạo
        # từ tham số dùng chung.
        c_tokens = repeat(
            self.c_token,
            '() n d -> b n d',
            b=b
        )

        # Ghép c_token vào đầu chuỗi theo chiều sequence.
        #
        # c_tokens: (b, 1, dim)
        # x:        (b, n, dim)
        # Kết quả:  (b, n+1, dim)
        #
        # Vị trí đầu tiên là c_token; các vị trí tiếp theo là đặc trưng
        # của chuỗi đầu vào.
        x = torch.cat(
            (c_tokens, x),
            dim=1
        )

        # Bước 3: Đưa toàn bộ chuỗi qua Transformer.
        #
        # Nhờ Self-Attention, c_token có thể kết hợp thông tin từ các
        # vị trí đặc trưng trong chuỗi.
        #
        # Kết quả có dạng (b, n+1, dim).
        x = self.transformer(x)

        # Bước 4: Lấy vector ở vị trí đầu tiên.
        #
        # x[:, 0] có dạng (b, dim), chính là biểu diễn đã được cập nhật
        # của c_token sau khi đi qua Transformer.
        #
        # Vector này được sử dụng làm Context Representation c_t.
        c_t = self.to_c_token(x[:, 0])

        return c_t
