
"""
Model3 模型
基于动态卷积和 Transformer 的 EEG 分类模型

主要组件:
- DynamicTemporalConv: 动态时间卷积，使用多个专家进行特征提取
- LiteDynamicMKCNN: 轻量级多核 CNN，提取时空特征
- DSCausalConv1d: 深度可分离因果卷积
- LightTCN: 轻量级时间卷积网络
- FrequencyAwareGQA: 频带感知分组查询注意力 (带 RoPE)
- Model3: 主模型，整合所有组件

参数说明:
- num_experts: 动态卷积专家数量
- freq_split_ratios: 频率分割比例 (beta, theta) -> mu = 1 - beta - theta
"""

from torch import nn, Tensor


from einops.layers.torch import Rearrange

import torch.nn.functional as F
import torch
import torch.nn as nn


class Conv1dWithConstraint(nn.Conv1d):
    """
    带权重约束的1D卷积层
    通过max_norm限制权重范数，防止梯度爆炸
    """
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)




class DynamicTemporalConv(nn.Module):
    """
    动态时间卷积 (Dynamic Temporal Convolution)
    根据输入特征动态聚合K个专家的权重

    工作原理:
    1. 通过全局平均池化压缩输入特征
    2. 注意力网络计算每个专家的权重
    3. 加权融合多个专家的卷积核
    4. 使用分组卷积高效应用不同权重
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_experts=4, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # Expect tuple (1, k)
        self.num_experts = num_experts

        # 1. Define Experts (Weight tensor: Experts x Out x In x H x W)
        # Note: We use register_parameter to make them learnable
        weight_shape = (num_experts, out_channels, in_channels, *kernel_size)
        self.weight = nn.Parameter(torch.Tensor(*weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        # 2. Attention Layer (Router) to calculate weights for experts
        # Compresses input to generate K scores
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(in_channels, max(4, in_channels // 4)),  # Reduction
            nn.ReLU(),
            nn.Linear(max(4, in_channels // 4), num_experts),
            nn.Softmax(dim=1)  # Normalize weights across experts
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        x: (Batch, In_Channels, H, W)
        """
        batch_size = x.shape[0]

        # Step 1: Calculate Attention Weights [Batch, num_experts]
        # Global Average Pooling to get context
        # Note: Since input is (B, 1, C, T), GAP works on C and T dimensions.
        # Ideally for dynamic temporal, we want context from the time series.
        pooled = self.avg_pool(x).view(batch_size, -1)
        attn_weights = self.attention(pooled)  # (B, K)

        # Step 2: Aggregate Weights
        # Resulting weight: (Batch * Out, In, H, W)
        # We reshape to use PyTorch's Grouped Conv trick for efficient per-sample weights

        # Reshape attention to (B, K, 1, 1, 1, 1) for broadcasting
        attn_weights = attn_weights.view(batch_size, self.num_experts, 1, 1, 1, 1)

        # Expert weights: (1, K, Out, In, H, W)
        all_experts = self.weight.unsqueeze(0)

        # Weighted sum: (B, Out, In, H, W)
        aggregate_weight = (attn_weights * all_experts).sum(dim=1)

        # Reshape for grouped convolution: (B*Out, In, H, W)
        aggregate_weight = aggregate_weight.view(batch_size * self.out_channels, self.in_channels, *self.kernel_size)

        # Step 3: Apply Convolution
        # Reshape input to (1, B*In, H, W)
        x_reshaped = x.view(1, batch_size * self.in_channels, x.shape[2], x.shape[3])

        # Use groups=batch_size to apply different weights to different samples
        output = F.conv2d(
            x_reshaped,
            aggregate_weight,
            bias=None,
            stride=1,
            padding=0,  # Padding is handled externally in your sequence
            groups=batch_size  # Crucial: separates the batch so each sample gets its own kernel
        )

        # Reshape output back to (B, Out, H, W)
        output = output.view(batch_size, self.out_channels, output.shape[2], output.shape[3])

        if self.bias is not None:
            # Handle dynamic bias similarly (omitted for simplicity as bias=False in original)
            pass

        return output


class LiteDynamicMKCNN(nn.Module):
    """
    Simplified Multi-Kernel CNN with Dynamic Convolution.
    Structure: [Multi-Scale Dynamic Temporal] -> [Spatial Depthwise] -> [Proj] -> [Pool]
    """

    def __init__(
            self,
            n_channels: int,  # EEG电极数量 (e.g., 22)
            d_model: int,  # 输出给Transformer的特征维度 (e.g., 48)
            temp_kernel_lengths: tuple = (20, 32, 64),  # 多尺度核
            F1: int = 16,  # 每个分支的滤波器数量 (适当减小，因为不用降维层了)
            D: int = 2,  # 深度卷积扩展系数
            final_pool: int = 8,  # 最终池化倍数
            dropout: float = 0.4,
            num_experts: int = 3  # 动态卷积专家数
    ):
        super().__init__()

        self.rearrange = Rearrange("b c seq -> b 1 c seq")

        # --- 1. 并行动态时间卷积 (Parallel Dynamic Temporal Convs) ---
        # 去掉了繁琐的Padding计算，直接利用 padding='same' (PyTorch 1.10+)
        # 或者在DynamicConv内部处理。这里假设您在上一步代码中处理了padding，
        # 或者我们简单地在这里用ConstantPad。
        self.temporal_branches = nn.ModuleList()
        for k in temp_kernel_lengths:
            # Padding计算：保证输出时间长度不变
            pad = (k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0)

            branch = nn.Sequential(
                nn.ConstantPad2d(pad, 0),
                DynamicTemporalConv(
                    in_channels=1,
                    out_channels=F1,
                    kernel_size=(1, k),
                    num_experts=num_experts,
                    bias=False
                ),
                nn.BatchNorm2d(F1)
                # 动态卷积自带非线性适应，这里可以不加激活，或加ELU
            )
            self.temporal_branches.append(branch)

        # 计算拼接后的通道数
        n_branches = len(temp_kernel_lengths)
        concat_channels = F1 * n_branches

        # --- 2. 空间深度卷积 (Spatial Depth-wise Conv) ---
        # 负责融合不同电极的信息
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                concat_channels,
                concat_channels * D,
                kernel_size=(n_channels, 1),
                groups=concat_channels,  # 深度卷积关键参数
                bias=False
            ),
            nn.BatchNorm2d(concat_channels * D),
            nn.ELU(),
        )

        # --- 3. 投影与池化 (Projection & Pooling) ---
        # 将通道数统一调整为Transformer需要的 d_model，并大幅降低时间分辨率
        self.proj_and_pool = nn.Sequential(
            nn.Dropout(dropout),
            # 1x1 卷积：融合特征并调整通道数到 d_model
            nn.Conv2d(concat_channels * D, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            # 最终池化：将时间维度压缩，生成Tokens
            nn.AvgPool2d((1, final_pool))
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = self.rearrange(x)  # (B, 1, C, T)

        # 1. 多分支并行提取
        branch_outputs = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(branch_outputs, dim=1)  # (B, F1*3, C, T)

        # 2. 空间滤波 (融合电极信息)
        x = self.spatial_conv(x)  # (B, F1*3*D, 1, T)

        # 3. 投影到 d_model 并池化
        x = self.proj_and_pool(x)  # (B, d_model, 1, T/pool)

        return x.squeeze(2)  # (B, d_model, T_new)


class Chomp1d(nn.Module):
    """ 切除右侧多余 padding，保证因果性 """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class DSCausalConv1d(nn.Module):
    """
    深度可分离因果卷积 (Depthwise Separable Causal Conv)
    参数量比标准卷积减少约 70% (取决于 kernel_size)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups_for_pointwise=1):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        # 1. Depthwise Conv: 提取时间特征
        # groups=in_channels 意味着每个通道独立进行卷积，互不干扰
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # [关键] 极大减少参数
            bias=False
        )

        self.chomp = Chomp1d(padding)

        # 2. Pointwise Conv: 融合通道特征
        # kernel_size=1，恢复通道间的交互
        # 保持 n_groups 参数，遵循原论文"组间独立"的设计
        self.pointwise = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1,
            groups=groups_for_pointwise,  # [关键] 保持原有的分组逻辑
            bias=True
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.chomp(x)
        x = self.pointwise(x)
        return x
class LightTCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 n_groups: int = 1, dropout: float = 0.3):
        super().__init__()

        # 使用深度可分离卷积替代标准卷积
        self.conv1 = DSCausalConv1d(
            n_filters, n_filters,
            kernel_size=kernel_length,
            dilation=dilation,
            groups_for_pointwise=n_groups
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = DSCausalConv1d(
            n_filters, n_filters,
            kernel_size=kernel_length,
            dilation=dilation,
            groups_for_pointwise=n_groups
        )
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.nonlinearity3 = nn.ELU()

        # 初始化 Pointwise 层的 bias (Depthwise层没有bias)
        nn.init.constant_(self.conv1.pointwise.bias, 0.0)
        nn.init.constant_(self.conv2.pointwise.bias, 0.0)

    def forward(self, input):
        # 结构保持完全一致：Conv -> BN -> ELU -> Dropout
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        # 残差连接
        x = self.nonlinearity3(input + x)
        return x


class LightTCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 n_groups: int = 1, dropout: float = 0.3):
        super(LightTCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            # 实例化轻量化 Block
            self.blocks.append(LightTCNBlock(kernel_length, n_filters, dilation, n_groups, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# class ClassificationHead(nn.Sequential):
#     def __init__(self, flatten_number, n_classes):
#         super().__init__()
#         self.fc = nn.Sequential(
#
#
#             nn.Dropout(0.5),
#             nn.Linear(flatten_number, n_classes)
#         )
#
#     def forward(self, x):
#         out = self.fc(x)
#         return out

class ClassificationHead(nn.Module):
    """
    Maps TCN features to class logits and optionally averages across groups.
    Expected input shape: (batch, d_model, 1)   ← after time-step selection.
    Output shape:        (batch, n_classes)
    """
    def __init__(
        self,
        d_features: int,
        n_groups: int,
        n_classes: int,
        kernel_size: int = 1,
        max_norm: float = 0.25,
    ):
        super().__init__()
        self.n_groups   = n_groups
        self.n_classes  = n_classes

        # self.drop = nn.Dropout(0.3)

        # point-wise (1 × 1) grouped conv = class projection per group
        self.linear = Conv1dWithConstraint(
            in_channels=d_features,
            out_channels=n_classes * n_groups,
            kernel_size=kernel_size,
            groups=n_groups,
            max_norm=max_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model, Tc)  →  logits: (B, n_classes)
        """
        # (B, n_classes*n_groups, 1) → squeeze last dim
        # x = self.drop(x) 
        x = self.linear(x).squeeze(-1)

        # (B, n_groups, n_classes) → mean over groups
        x = x.view(x.size(0), self.n_groups, self.n_classes).mean(dim=1)
        return x

class TCNHead(nn.Module):
    def __init__(self, d_features: int = 64, n_groups: int = 1, tcn_depth: int = 2, 
                 kernel_length: int = 4,  dropout_tcn: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.tcn = LightTCN(tcn_depth, kernel_length, d_features, n_groups, dropout_tcn)

        # self.linear = Conv1dWithConstraint(d_model, n_classes*n_groups, kernel_size=1, 
        #                                         groups=n_groups, max_norm=0.25)   
        self.classifier = ClassificationHead(
            d_features=d_features,
            n_groups=n_groups,
            n_classes=n_classes,
        )     
    def forward(self, x):
        x = self.tcn(x)
        x = x[:, :, -1:]

        x = self.classifier(x)   # (B, n_classes)
        # tcn_out = self.linear(tcn_out).squeeze(-1)

        # tcn_out = tcn_out.view(x.shape[0], self.n_groups, self.n_classes)
        # tcn_out = tcn_out.mean(dim=1) 

        return x
    
# ------------------------------------------------------------------------------- #
# MSCFormer
# ------------------------------------------------------------------------------- #


class Model3(nn.Module):
    def __init__(self,
            n_channels: int ,
            n_classes: int,
            temp_kernel_lengths=(16, 32, 64),
            pool_length_1: int = 8,
            pool_length_2: int = 7,
            D: int = 2,
            dropout_conv: float = 0.3,
            d_group: int = 16,
            tcn_depth: int = 2,
            kernel_length_tcn: int = 4,
            dropout_tcn: float = 0.3,
            kv_heads: int = 4,
            q_heads: int = 8,
            trans_dropout: float = 0.4,
            drop_path_max: float = 0.25,
            trans_depth: int = 5,
            num_experts: int = 3,  # [Sensitivity] Dynamic Conv Experts
            freq_split_ratios: tuple = (0.3, 0.3) # [Sensitivity] Freq Split (Beta, Theta)
        ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group*self.n_groups

        self.rearrange = Rearrange("b c seq -> b seq c")

        n_groups = len(temp_kernel_lengths)
        calculated_d_model = d_group * n_groups

        # 合并池化步长 (例如: 8 * 7 = 56)
        total_pool_stride = pool_length_1 * pool_length_2

        # === 2. 实例化 LiteDynamicMKCNN ===
        self.conv_block = LiteDynamicMKCNN(
            n_channels=n_channels,  # 保持不变
            d_model=calculated_d_model,  # [变化] 传入计算好的总维度
            temp_kernel_lengths=temp_kernel_lengths,  # 保持不变
            F1=8,  # [建议修改] 建议从 32 改为 16 或 8 (因为动态卷积专家增加了参数量)
            D=D,  # 保持不变
            final_pool=total_pool_stride,  # [变化] 传入合并后的池化值
            dropout=dropout_conv,  # 保持不变
            num_experts=num_experts  # [Sensitivity] Pass num_experts
        )
        self.mix = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=1,              # across channels only
                groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU()
        )

        # linearly increasing drop path rates from 0 to drop_path_max for deeper layers
        # drop_rates = torch.linspace(0.0, drop_path_max, trans_depth)
        # Quadratically increasing drop path rates from 0 to drop_path_max for deeper layers
        drop_rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max

        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self.transformer = nn.ModuleList([
            _TransformerBlock(self.d_model, q_heads, kv_heads, dropout=trans_dropout,
                              drop_path_rate=drop_rates[i].item(),
                              freq_split_ratios=freq_split_ratios)
            for i in range(trans_depth)
        ])

        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),            # 1. rearrange for Conv1d over channels
            nn.Conv1d(in_channels=self.d_model,     # 2. 1x1 conv over channel dim
                out_channels=d_group,
                kernel_size=1,
                groups=1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )
        #
        self.tcn_head = TCNHead(d_group*(self.n_groups+1), (self.n_groups+1), tcn_depth,
                                kernel_length_tcn, dropout_tcn, n_classes)



    def forward(self, x):      # x: [B, C_electrodes, T]
        conv_features = self.conv_block(x)
        B, C, T = conv_features.shape

        tokens = self.rearrange(self.mix(conv_features))
        cos, sin = self._rotary_cache(T , tokens.device)
        for blk in self.transformer:
            tokens = blk(tokens, cos, sin)
        tran_features = self.reduce(tokens)

        features = torch.cat((conv_features, tran_features), dim=1)
        out = self.tcn_head(features)

        # return features, out
        return out
    # ------------------------------------------------------------------
    # def _rotary_cache(self, seq_len: int, device: torch.device):
    #     if (self._cos is None) or (self._cos.shape[0] < seq_len):
    #         cos, sin = _build_rotary_cache(self.cls_token.shape[-1], seq_len, device)
    #         self._cos, self._sin = cos.to(device), sin.to(device)
    #     return self._cos, self._sin
    def _rotary_cache(self, seq_len: int, device: torch.device):
        """Build (or reuse) RoPE caches for the current sequence length."""
        head_dim = self.transformer[0].attn.head_dim  # use per‑head dimension, **not** d_model
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin

# -----------------------------------------------------------------------------
# 0.  helpers
# -----------------------------------------------------------------------------

def _xavier_zero_bias(module: nn.Module) -> None:
    """Apply Xavier‑uniform + zero bias to every conv/linear inside *module*."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# -----------------------------------------------------------------------------
# 1.  Rotary positional embedding utilities
# -----------------------------------------------------------------------------

# Adapted from GPT‑NeoX & LLaMA implementations

def _build_rotary_cache(head_dim: int, seq_len: int, device: torch.device):
    """Return cos & sin tensors of shape (seq_len, head_dim)."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)                 # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)             # duplicate for even/odd
    cos, sin = emb.cos(), emb.sin()
    return cos, sin                                     # each: (seq, head_dim)

def _rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):  # q/k: (B, h, T, d)
    def _rotate(x):                                        # half rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    q_out = (q * cos) + (_rotate(q) * sin)
    k_out = (k * cos) + (_rotate(k) * sin)
    return q_out, k_out


# -----------------------------------------------------------------------------
# 4.  Grouped‑Query Self‑Attention (GQA) with RoPE
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch import Tensor


# 假设外部引用
# from your_module import _rope, _xavier_zero_bias

class FrequencyAwareGQA(nn.Module):
    """
    Frequency-Aware GQA with Subject-Invariant Query Correction.

    改进点:
    1. 频带感知: Q heads 按 3:4:3 分组，KV heads 固定为 3 组。
    2. LOSO 适配: Q 投影后执行 Q = LayerNorm(Q) - E_s[Q].detach()。
       其中 E_s[Q] 计算为当前样本在时间维度上的均值，以去除被试特异性的绝对幅值偏置。
    """

    def __init__(self, d_model: int, num_q_heads: int, dropout: float = 0.3, 
                 split_ratios: tuple = (0.3, 0.3)):
        super().__init__()
        # --- 基础 GQA 设置 ---
        self.num_q_heads = num_q_heads
        self.head_dim = d_model // num_q_heads
        self.scale = self.head_dim ** -0.5

        # 3:4:3 分组逻辑 (可通过 split_ratios 调整)
        # split_ratios = (beta_ratio, theta_ratio) -> mu_ratio = 1 - beta - theta
        beta_ratio, theta_ratio = split_ratios
        h_beta = int(num_q_heads * beta_ratio)
        h_theta = int(num_q_heads * theta_ratio)
        h_mu = num_q_heads - h_beta - h_theta
        self.group_splits = [h_beta, h_mu, h_theta]

        assert d_model % num_q_heads == 0, "d_model must divide num_q_heads"
        self.num_kv_heads = 3  # 固定 3 组 KV

        # --- 投影层 ---
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, self.num_kv_heads * 2 * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # --- 新增: Subject-Invariant Correction ---
        # 对每个 Head 的 Query 向量进行 LayerNorm
        self.q_ln = nn.LayerNorm(self.head_dim)
        self.enable_subject_correction = True

        self.drop = nn.Dropout(dropout)

        # 初始化
        if '_xavier_zero_bias' in globals():
            _xavier_zero_bias(self)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.kv_proj.weight)
            nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, C = x.shape

        # 1. Query 投影 & 变形 -> (B, num_q_heads, T, d)
        # 注意：这里先 view 成 (B, T, H, d) 方便做 LayerNorm，然后再 transpose
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)

        # --- 新增操作: 去除被试特异性偏置 ---
        # a) LayerNorm: 标准化幅度
        q = self.q_ln(q)

        # b) 减去均值 (E_s[Q]): 去除绝对偏置
        # 计算维度: 沿 Time 维度 (dim=1) 求均值，得到该样本在该时间窗内的中心偏移
        # 形状: (B, T, H, d) -> mean(dim=1) -> (B, 1, H, d)
        q_mean = q.mean(dim=1, keepdim=True)

        # c) 减去 detach 后的均值
        if self.enable_subject_correction:
            q = q - q_mean.detach()
        # ----------------------------------

        # 转置为 Attention 需要的形状 -> (B, H, T, d)
        q = q.transpose(1, 2)

        # 2. KV 投影 -> (B, T, 3, 2, d)
        kv = self.kv_proj(x).view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv[..., 0, :].transpose(1, 2), kv[..., 1, :].transpose(1, 2)

        # 3. 频带感知广播 (3:4:3)
        k_groups = []
        v_groups = []
        k_slices = k.split(1, dim=1)
        v_slices = v.split(1, dim=1)

        for i, split_size in enumerate(self.group_splits):
            k_groups.append(k_slices[i].expand(-1, split_size, -1, -1))
            v_groups.append(v_slices[i].expand(-1, split_size, -1, -1))

        k = torch.cat(k_groups, dim=1)
        v = torch.cat(v_groups, dim=1)

        # 4. RoPE
        if '_rope' in globals():
            q, k = _rope(q, k, cos[:T, :], sin[:T, :])

        # 5. Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(out)


import torch
import torch.nn as nn
from torch import Tensor
import math






# -----------------------------------------------------------------------------
# DropPath (保持不变)
# -----------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# -----------------------------------------------------------------------------
# Transformer Block (更新以使用 FrequencyAwareGQA)
# -----------------------------------------------------------------------------
class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, q_heads: int, kv_heads: int = 4, mlp_ratio: int = 2, dropout=0.4, drop_path_rate=0.25,
                 window_size=17, freq_split_ratios=(0.3, 0.3)):
        super().__init__()
        self.q_heads = q_heads
        self.norm1 = nn.LayerNorm(d_model)

        # 修改点：不再传递 kv_heads，因为 FrequencyAwareGQA 内部固定为 3
        # 传递 freq_split_ratios
        self.attn = FrequencyAwareGQA(d_model, q_heads, dropout, split_ratios=freq_split_ratios)

        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # Pre-Norm + Attention + Residual + DropPath
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        # Pre-Norm + MLP + Residual + DropPath
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
# def main():
#     # Example usage: run benchmark with dummy input shape (batch, channels, time)
#    model = Model3(n_channels=22,n_classes=4)
#    from torchinfo import summary
#
#    total_params = sum(p.numel() for p in model.parameters())
#    print(f"总参数量: {total_params:,}")
