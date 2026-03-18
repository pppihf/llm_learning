"""
========================================
 第02课：Transformer 与注意力机制
========================================
Transformer 是大模型的核心架构。理解 Self-Attention 是面试必备。
本课从零实现注意力机制，并深入解析各种位置编码。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/02_attention_mechanism.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

print("=" * 60)
print(" 第02课：Transformer 与注意力机制")
print("=" * 60)

# ============================================================
# 1. 为什么需要 Attention？
# ============================================================
print("\n" + "=" * 60)
print("【1. 为什么需要 Attention？】")
print("=" * 60)
print("""
RNN 的问题：
  - 序列处理是串行的（无法并行）
  - 长距离依赖问题（信息在传递中衰减）
  - 梯度消失/爆炸

Attention 的核心思想：
  "让序列中的每个位置都能直接关注到其他所有位置"

公式: Attention(Q, K, V) = softmax(QK^T / √d_k) V

其中:
  Q (Query):  "我要查询什么信息"
  K (Key):    "我有什么信息可以提供"
  V (Value):  "我实际提供的信息内容"
  √d_k:      缩放因子，防止点积过大导致 softmax 饱和

【面试考点】
Q: 为什么要除以 √d_k？
A: 当 d_k 较大时，QK^T 的方差会随 d_k 增大。
   假设 Q, K 的元素独立同分布，均值0方差1：
   Var(q·k) = d_k  →  点积的方差随维度线性增长
   除以 √d_k 使方差归一化为1，避免 softmax 输入过大导致梯度消失。
""")

# ============================================================
# 2. 手动实现 Scaled Dot-Product Attention
# ============================================================
print("\n" + "=" * 60)
print("【2. 手动实现 Scaled Dot-Product Attention】")
print("=" * 60)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    手动实现缩放点积注意力

    Args:
        Q: (batch, seq_len_q, d_k)
        K: (batch, seq_len_k, d_k)
        V: (batch, seq_len_v, d_v)  # seq_len_k == seq_len_v
        mask: (batch, seq_len_q, seq_len_k) 或可广播的形状

    Returns:
        output: (batch, seq_len_q, d_v)
        attention_weights: (batch, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Step 1: 计算注意力分数 QK^T / √d_k
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores shape: (batch, seq_len_q, seq_len_k)

    # Step 2: 应用 mask（可选，用于 causal attention 或 padding）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: 加权求和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# 演示
batch_size, seq_len, d_model = 1, 4, 8
Q = torch.randn(batch_size, seq_len, d_model, device=device)
K = torch.randn(batch_size, seq_len, d_model, device=device)
V = torch.randn(batch_size, seq_len, d_model, device=device)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("Output shape:", output.shape)
print("Attention Weights shape:", weights.shape)
print("\nAttention Weights (每行表示一个 token 对其他 token 的注意力):")
print(weights[0].detach().cpu().numpy().round(3))
print("注意：每行之和 = 1（softmax 归一化）")
print("行和:", weights[0].sum(dim=-1).detach().cpu().numpy())

# ============================================================
# 3. Causal Mask（因果掩码）
# ============================================================
print("\n\n" + "=" * 60)
print("【3. Causal Mask（因果掩码）- GPT 解码的核心】")
print("=" * 60)
print("""
Decoder-only 模型（GPT/Llama/Qwen）使用因果掩码：
  每个 token 只能看到自己和前面的 token，不能看到未来。

数学上就是一个下三角矩阵：
  [[1, 0, 0, 0],
   [1, 1, 0, 0],
   [1, 1, 1, 0],
   [1, 1, 1, 1]]

【面试考点】
Q: 为什么 GPT 需要 causal mask，而 BERT 不需要？
A: GPT 是自回归模型（左→右生成），训练时预测下一个 token，
   如果能看到未来就"作弊"了。BERT 是双向的，用 [MASK] 替换来防泄露。
""")

# 创建因果掩码
seq_len = 6
causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
print(f"Causal Mask ({seq_len} × {seq_len}):")
print(causal_mask.int().cpu().numpy())

# 用因果掩码做 attention
Q = torch.randn(1, seq_len, d_model, device=device)
K = torch.randn(1, seq_len, d_model, device=device)
V = torch.randn(1, seq_len, d_model, device=device)

output_causal, weights_causal = scaled_dot_product_attention(
    Q, K, V, mask=causal_mask.unsqueeze(0))

print("\n使用 Causal Mask 后的 Attention Weights:")
print(weights_causal[0].detach().cpu().numpy().round(3))
print("注意：上三角全是 0，每个 token 只能看到自己和之前的 token")

# ============================================================
# 4. Multi-Head Attention（多头注意力）
# ============================================================
print("\n\n" + "=" * 60)
print("【4. Multi-Head Attention - 手动实现】")
print("=" * 60)
print("""
为什么需要多头？
  单个 attention 只能学到一种"关注模式"。
  多头让模型同时学习不同类型的依赖关系：
    - 一个头关注语法关系
    - 一个头关注语义相关性
    - 一个头关注位置邻近关系
    ...

多头注意力流程：
  1. 将 Q, K, V 分别投影到 h 个子空间
  2. 每个头独立做 attention
  3. 拼接所有头的输出
  4. 通过一个线性层融合

参数量：
  假设 d_model=768, num_heads=12, 则 d_k = 768/12 = 64
  参数量 = 3 × d_model × d_model (QKV投影) + d_model × d_model (输出投影)
         = 4 × 768 × 768 ≈ 2.36M
""")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # QKV 投影矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Step 1: 线性投影
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: 拆分成多头 reshape + transpose
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: 计算注意力（每个头独立计算）
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_q, seq_k) 广播到所有头
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Step 4: 拼接多头 (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        # Step 5: 输出投影
        output = self.W_o(attn_output)

        return output, attn_weights


# 演示
d_model, num_heads = 64, 8
mha = MultiHeadAttention(d_model, num_heads).to(device)

x = torch.randn(2, 10, d_model, device=device)  # batch=2, seq_len=10
output, weights = mha(x, x, x)  # Self-attention: Q=K=V=x

print("输入 shape:", x.shape)
print("输出 shape:", output.shape)
print("注意力权重 shape:", weights.shape, "(batch, heads, seq_q, seq_k)")
print(f"参数量: {sum(p.numel() for p in mha.parameters())}")

# ============================================================
# 5. 位置编码 (Positional Encoding)
# ============================================================
print("\n\n" + "=" * 60)
print("【5. 位置编码 - Sinusoidal vs RoPE】")
print("=" * 60)
print("""
Attention 是置换不变的（permutation invariant），不区分位置。
必须通过位置编码告诉模型每个 token 的位置。

┌──────────────────┬─────────────────────────────────────┐
│ Sinusoidal       │ 原始 Transformer：固定的三角函数     │
│ (绝对位置编码)   │ PE(pos,2i) = sin(pos/10000^(2i/d))  │
│                  │ PE(pos,2i+1) = cos(pos/10000^(2i/d)) │
├──────────────────┼─────────────────────────────────────┤
│ Learned          │ BERT/GPT-2：可学习的位置 embedding   │
│ (可学习位置编码) │ 简单但受限于最大训练长度              │
├──────────────────┼─────────────────────────────────────┤
│ RoPE             │ Llama/Qwen：旋转位置编码 ★★★         │
│ (旋转位置编码)   │ 相对位置信息，天然支持长度外推        │
│                  │ 核心：通过旋转矩阵编码相对位置        │
├──────────────────┼─────────────────────────────────────┤
│ ALiBi            │ BLOOM：注意力线性偏置                 │
│                  │ 直接在 attention score 上加位置偏置   │
└──────────────────┴─────────────────────────────────────┘

【面试高频考点】
Q: RoPE 为什么比绝对位置编码好？
A: 1) 编码的是相对位置，更符合语言的本质（语义取决于词间距离）
   2) 通过旋转矩阵实现，计算高效（只在 Q, K 上应用）
   3) 天然支持长度外推（结合 NTK-aware scaling 等技术）
""")


# 实现 Sinusoidal 位置编码
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 实现 RoPE (简化版)
class RotaryPositionalEncoding(nn.Module):
    """RoPE: 旋转位置编码 (简化实现)"""

    def __init__(self, d_model, max_len=5000, base=10000):
        super().__init__()
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # 预计算 cos 和 sin
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_len, d_model/2)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, q, k):
        """对 Q 和 K 应用旋转编码"""
        seq_len = q.size(-2)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq,d/2)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # 将 q, k 拆分为偶数和奇数位
        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        # 应用旋转: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
        q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        q_rot = q_rot.flatten(-2)
        k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        k_rot = k_rot.flatten(-2)

        return q_rot, k_rot


# 演示 RoPE
rope = RotaryPositionalEncoding(d_model=64).to(device)
q = torch.randn(1, 4, 10, 64, device=device)  # (batch, heads, seq, d_k)
k = torch.randn(1, 4, 10, 64, device=device)
q_rot, k_rot = rope(q, k)
print("RoPE 应用前 Q shape:", q.shape)
print("RoPE 应用后 Q shape:", q_rot.shape)

# 验证 RoPE 的相对位置特性
print("\n验证 RoPE 保持相对位置信息:")
q_test = torch.randn(1, 1, 5, 64, device=device)
k_test = q_test.clone()
q_r, k_r = rope(q_test, k_test)
scores = torch.matmul(q_r, k_r.transpose(-2, -1))
print("QK^T (应用RoPE后) - 对角线上最大，相对距离相同的分数相近:")
print(scores[0, 0].detach().cpu().numpy().round(2))

# ============================================================
# 6. 完整的 Transformer Block
# ============================================================
print("\n\n" + "=" * 60)
print("【6. 完整的 Transformer Decoder Block】")
print("=" * 60)
print("""
一个完整的 Decoder Block 包含：
  1. (Masked) Multi-Head Self-Attention
  2. LayerNorm
  3. Feed-Forward Network (FFN)
  4. LayerNorm
  5. Residual Connection

现代 LLM 的改进：
  ┌───────────────┬──────────────┬──────────────────────┐
  │ 改进点        │ 原始设计     │ 现代 LLM             │
  ├───────────────┼──────────────┼──────────────────────┤
  │ LayerNorm     │ Post-LN      │ Pre-LN (更稳定)      │
  │ 归一化方法    │ LayerNorm    │ RMSNorm (更快)       │
  │ FFN 激活      │ ReLU         │ SwiGLU (效果更好)    │
  │ 位置编码      │ Sinusoidal   │ RoPE                 │
  │ 注意力        │ MHA          │ GQA/MQA (更省显存)   │
  └───────────────┴──────────────┴──────────────────────┘
""")


class RMSNorm(nn.Module):
    """RMSNorm - Llama/Qwen 等模型使用的归一化方法"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN - 现代 LLM 的标配 FFN"""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)  # 典型比例
            d_ff = ((d_ff + 63) // 64) * 64  # 对齐到64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        # SwiGLU: w2(SiLU(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """现代 Transformer Decoder Block (Pre-LN + RMSNorm + SwiGLU)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-LN: Norm -> Attention -> Residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, mask)
        x = x + attn_out  # 残差连接

        # Pre-LN: Norm -> FFN -> Residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out  # 残差连接

        return x


# 演示完整的 Transformer Block
block = TransformerBlock(d_model=256, num_heads=8).to(device)
x = torch.randn(2, 20, 256, device=device)
output = block(x)

total_params = sum(p.numel() for p in block.parameters())
print("Transformer Block:")
print("  输入 shape:", x.shape)
print("  输出 shape:", output.shape)
print(f"  参数量: {total_params} ({total_params / 1000:.2f} K)")

# ============================================================
# 7. GQA (Grouped Query Attention)
# ============================================================
print("\n\n" + "=" * 60)
print("【7. GQA - Grouped Query Attention】")
print("=" * 60)
print("""
MHA vs MQA vs GQA:

MHA (Multi-Head Attention):
  Q: h 个头  K: h 个头  V: h 个头
  标准做法，每个头都有独立的 K, V

MQA (Multi-Query Attention):
  Q: h 个头  K: 1 个头  V: 1 个头
  所有 Q 头共享一个 K 和 V，大幅减少 KV-Cache

GQA (Grouped Query Attention):  ★★★ 当前主流
  Q: h 个头  K: g 个头  V: g 个头  (g < h, h 能被 g 整除)
  每组 Q 头共享一个 K, V 头
  是 MHA 和 MQA 的折中方案

例: Llama-3 70B: 64 Q heads, 8 KV heads (每8个Q头共享1个KV)

【面试考点】
Q: GQA 相比 MHA 有什么优势？
A: 1) KV-Cache 显存减少 h/g 倍（推理最关键的优化）
   2) 推理速度提升（KV-Cache 读取更少）
   3) 训练效果接近 MHA（通过更多 Q 头补偿信息容量）

Q: KV-Cache 是什么？为什么它是推理优化的关键？
A: 自回归生成时，之前 token 的 K、V 不变，缓存起来避免重复计算。
   KV-Cache 大小 = 2 × num_layers × num_kv_heads × d_k × seq_len × batch_size
   这就是为什么长序列推理很耗显存。GQA 通过减少 KV heads 来减少 cache 大小。
""")

# ============================================================
# 8. FlashAttention 简介
# ============================================================
print("\n" + "=" * 60)
print("【8. FlashAttention 简介】")
print("=" * 60)
print("""
FlashAttention 是一种 IO-aware 的精确注意力计算方法（非近似）。

核心问题：标准 Attention 的瓶颈不在计算，而在显存读写（IO）
  - 需要存储完整的 score 矩阵 (seq_len × seq_len)
  - 反复在 HBM (高带宽显存) 和 SRAM (片上缓存) 之间搬运数据

FlashAttention 的解决方案：
  1. Tiling: 将 Q, K, V 分块加载到 SRAM
  2. 在 SRAM 中完成计算（利用 online softmax 技巧）
  3. 不需要存储完整的 attention 矩阵
  4. 减少 HBM 访问次数

效果：
  - 显存节省: O(N²) -> O(N) 
  - 速度提升: 2-4x
  - 结果完全一致（精确计算，非近似）

代码使用:
  from torch.nn.functional import scaled_dot_product_attention
  # PyTorch 2.0+ 自动调用 FlashAttention（如果硬件支持）

【面试考点】
Q: FlashAttention 为什么能加速？
A: 不是减少了计算量（FLOPs 不变），而是减少了显存读写（IO）。
   通过 tiling + online softmax，将完整的 N×N attention 矩阵分块在 SRAM 中计算，
   避免了将完整矩阵写回 HBM 的开销。
""")

# 验证 PyTorch 的 scaled_dot_product_attention
print("测试 PyTorch 内置 scaled_dot_product_attention:")
q = torch.randn(2, 8, 64, 32, device=device)  # (batch, heads, seq, d_k)
k = torch.randn(2, 8, 64, 32, device=device)
v = torch.randn(2, 8, 64, 32, device=device)
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("  输入 shape: (2, 8, 64, 32)")
print("  输出 shape:", output.shape)
print("  (PyTorch 自动选择最优 backend: FlashAttention / Memory-Efficient / Math)")

# ============================================================
# 9. 观察真实模型的 Attention 结构
# ============================================================
print("\n\n" + "=" * 60)
print("【9. 查看真实模型的 Attention 结构】")
print("=" * 60)

from transformers import AutoConfig

configs_to_check = [
    ("/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    ("/publicdata/huggingface.co/Qwen/Qwen3-14B", "Qwen3-14B"),
    ("/publicdata/huggingface.co/Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct"),
]

for model_id, name in configs_to_check:
  try:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"\n{name} 配置:")
    print(f"  hidden_size (d_model): {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  num_attention_heads (Q): {getattr(config, 'num_attention_heads', 'N/A')}")
    print(f"  num_key_value_heads (KV): {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"  num_hidden_layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"  intermediate_size (FFN): {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"  vocab_size: {getattr(config, 'vocab_size', 'N/A')}")
    print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"  rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"  hidden_act: {getattr(config, 'hidden_act', 'N/A')}")

    # 计算参数量
    h = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    ffn = config.intermediate_size
    n_kv = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    n_q = config.num_attention_heads
    d_k = h // n_q

    # 估算参数量
    embedding_params = V * h
    attn_params_per_layer = h * (n_q * d_k + 2 * n_kv * d_k + h)  # QKV投影 + O投影
    ffn_params_per_layer = h * ffn * 3  # SwiGLU: gate + up + down
    norm_params_per_layer = h * 2  # 两个 RMSNorm
    total = embedding_params + L * (attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer)

    print("  --- 参数量估算 ---")
    print(f"  Embedding: {embedding_params / 1e6:.1f}M")
    print(f"  每层 Attention: {attn_params_per_layer / 1e6:.1f}M")
    print(f"  每层 FFN: {ffn_params_per_layer / 1e6:.1f}M")
    print(f"  总参数量估算: {total / 1e6:.1f}M")
    print(f"  GQA 比例: {n_q} Q heads / {n_kv} KV heads = {n_q // n_kv}x 共享")
  except Exception as e:
    print(f"{name} 加载失败: {e} (需要网络下载)")

# ============================================================
# 10. 总结
# ============================================================
print("\n\n" + "=" * 60)
print("【10. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. Attention(Q,K,V) = softmax(QK^T/√d_k) × V
  2. Causal Mask 实现自回归（左→右）
  3. Multi-Head 让模型学习多种注意力模式
  4. RoPE 通过旋转矩阵编码相对位置
  5. GQA 在效果和效率之间取得平衡
  6. FlashAttention 通过减少 IO 加速（非近似）

✅ 面试高频题：
  1. 为什么 Attention 要除以 √d_k？（方差归一化，防 softmax 饱和）
  2. Pre-LN vs Post-LN 的区别和优劣？
  3. RoPE 的原理和优势？
  4. GQA 和 MHA 的区别？KV-Cache 怎么计算？
  5. FlashAttention 的原理？为什么能加速？
  6. SwiGLU 比 ReLU FFN 好在哪？
  7. Transformer 的计算复杂度？（O(n²d) for attention, O(nd²) for FFN）

下一课：03_model_inference.py - 模型加载与推理
""")
