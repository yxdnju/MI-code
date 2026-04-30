import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import DeformConv2d
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim ** .5
    attn = F.softmax(scores, dim=-1)
    return attn
class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []

        for i in range(out_shape):
            index = i * self.stride
            input = x[:, :, index:index + self.kernel_size]
            output = torch.log(torch.clamp(input.var(dim=-1, keepdim=True), 1e-6, 1e6))
            out.append(output)
        out = torch.cat(out, dim=-1)

        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, window_size):
        super().__init__()
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.n_head = n_head
        self.register_buffer("base_mask", torch.ones(window_size, window_size))
        self.learnable_mask = nn.Parameter(torch.ones(window_size, window_size))
        with torch.no_grad():
            for i in range(window_size):
                for j in range(window_size):
                    distance = abs(i - j)
                    decay = 0.95 ** distance
                    self.learnable_mask[i, j] = decay
        self.w_q = nn.Linear(d_model, n_head * self.d_k)
        self.w_k = nn.Linear(d_model, n_head * self.d_k)
        self.w_v = nn.Linear(d_model, n_head * self.d_v)
        self.w_o = nn.Linear(n_head * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        B, T, C = query.shape
        attn = attention(q, k, v)
        mask = self.base_mask * torch.tanh(self.learnable_mask)

        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        mask = mask.expand(B, self.n_head, T, T)  # [B, H, T, T]


        attn = attn * mask


        strict_mask = self.base_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, T, T) == 0
        attn = attn.masked_fill(strict_mask,float())
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = rearrange(out, 'b h q d -> b q (h d)')
        out = self.dropout(self.w_o(out))

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop,64)
        self.feed_forward = FeedForward(embed_dim, embed_dim * fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, data):
        res = self.layernorm1(data)
        out = data + self.multihead_attention(res, res, res)

        res = self.layernorm2(out)
        output = out + self.feed_forward(res)
        return output

class TMFNet(nn.Module):
    def __init__(self, num_classes=4, num_channels=22, embed_dim=32, pool_size=50,
                 pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5,decay_factor = 0.96):
        super().__init__()
        self.decay_factor = decay_factor
        self.temp_conv1 = nn.Conv2d(1, 4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, 4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, 4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, 4, (1, 65), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.spatial_conv = nn.Conv2d(16, 16, (num_channels, 1))
        self.MTSA = MTSA(32)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu = nn.ELU()

        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.MaxPool1d(pool_size, pool_stride)



        self.dropout = nn.Dropout(0.5)

        self.TDAS = nn.ModuleList(
            [TransformerEncoder(embed_dim // 2, num_heads, fc_ratio, attn_drop, fc_drop) for i in range(depth)]
        )

        self.FC = nn.Sequential(
            nn.Conv2d(32, 32, (2, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            SEBlock1(32)
        )
        self.classify = nn.Linear(512, num_classes)
        self.down = nn.Conv2d(64, 32, (1, 1))

    def forward(self, x_):
        x = x_.unsqueeze(dim=1)
        x1 = self.temp_conv1(x)
        x2 = self.temp_conv2(x)
        x3 = self.temp_conv3(x)
        x4 = self.temp_conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = x.squeeze(dim=2)
        x1 = self.avg_pool(x)
        x2 = self.var_pool(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x1 = rearrange(x1, 'b d n -> b n d')
        x2 = rearrange(x2, 'b d n -> b n d')
        for encoder in self.TDAS:
            x1 = encoder(x1)
            x2 = encoder(x2)
        x1 = x1.unsqueeze(dim=2)
        x2 = x2.unsqueeze(dim=2)
        x = torch.cat((x1, x2), dim=2)
        x = self.down(x)
        x = self.MTSA(x)
        x = self.FC(x)
        x = x.reshape(x.size(0), -1)
        out = self.classify(x)
        return out





class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class SEBlock1(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock1, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PyramidFusion(nn.Module):
    def __init__(self, channels):
        super(PyramidFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv3x3 = DeformConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = DeformConv2d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.fusion = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        # self.dp = nn.Dropout()
    def forward(self, x):
        k3, k5 = 3, 5  # 3x3 和 5x5 的核大小
        offset3 = torch.zeros(x.shape[0], 2 * k3 * k3, x.shape[2], x.shape[3], device=x.device)
        offset5 = torch.zeros(x.shape[0], 2 * k5 * k5, x.shape[2], x.shape[3], device=x.device)
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x, offset3)
        x3 = self.conv5x5(x, offset5)
        x4 = self.dilated_conv(x)
        out = torch.cat([x1, x2, x3,x4], dim=1)

        return self.fusion(out)


class MTSA(nn.Module):
    def __init__(self, in_channels):
        super(MTSA, self).__init__()
        self.pyramid_fusion = PyramidFusion(in_channels)
        self.se_block = SEBlock(in_channels)

    def forward(self, x):
        x_fused = self.pyramid_fusion(x)
        x_fft = torch.fft.fft2(x_fused).abs()
        x_fft = self.se_block(x_fft)
        x_final = torch.fft.ifft2(x_fft).real
        return x_final
