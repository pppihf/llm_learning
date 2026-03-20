"""
========================================
 第07课：推理优化
========================================
理解大模型推理的核心瓶颈（显存/带宽/计算），
掌握 KV-Cache、Continuous Batching、Paged Attention
等关键优化技术的原理。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/07_inference_optimization.py
"""
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print(" 第07课：推理优化")
print("=" * 60)

# ============================================================
# 1. 推理的两个阶段
# ============================================================
print("\n" + "=" * 60)
print("【1. 推理的两个阶段】")
print("=" * 60)
print("""
大模型推理分为两个截然不同的阶段：

  ┌────────────────────────────────────────────────────┐
  │ Prefill（预填充阶段）                              │
  │                                                    │
  │ 输入: 完整 prompt [t1, t2, ..., tn]                │
  │ 特点: 所有 token 可以并行处理                      │
  │ 瓶颈: 计算密集（Compute-bound）                    │
  │ 指标: TTFT (Time To First Token) — 首 token 延迟   │
  ├────────────────────────────────────────────────────┤
  │ Decode（解码阶段）                                 │
  │                                                    │
  │ 输入: 每次只有 1 个新 token                        │
  │ 特点: 自回归，一个一个生成                         │
  │ 瓶颈: 显存带宽密集（Memory-bound）                 │
  │ 指标: TPS (Tokens Per Second) — 生成速度            │
  └────────────────────────────────────────────────────┘

为什么 Decode 是带宽瓶颈？
  每生成一个 token，需要把整个模型参数从显存读一遍。
  7B 模型 fp16 = 14GB 参数，假设显存带宽 2TB/s：
  → 每个 token 至少需要 14GB / 2000GB/s = 7ms
  → 理论上限约 143 tokens/s

【面试考点】
Q: 为什么 Prefill 和 Decode 性能特征不同？
A: Prefill 处理 N 个 token，矩阵乘法可以充分利用 GPU 并行；
   Decode 只有 1 个 token，矩阵乘法退化为向量 × 矩阵，
   GPU 算力用不满，瓶颈变成从显存搬运模型参数的带宽。
""")

# ============================================================
# 2. 实际测量 TTFT 和生成速度
# ============================================================
print("\n" + "=" * 60)
print("【2. 实际测量 TTFT 和生成速度】")
print("=" * 60)

model_path = "/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"模型已加载: {model_path}")

prompt = "请用一段话解释什么是 KV-Cache："
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
max_new_tokens = 100

# warmup
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

# 测量
torch.cuda.synchronize()
t_start = time.perf_counter()
with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
torch.cuda.synchronize()
t_end = time.perf_counter()

generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
num_generated = len(generated_ids)
total_time = t_end - t_start

print(f"\n生成的文本:\n  {generated_text[:200]}...")
print(f"\n生成 token 数: {num_generated}")
print(f"总耗时: {total_time:.3f}s")
print(f"平均速度: {num_generated / total_time:.1f} tokens/s")
print(f"平均每 token: {total_time / num_generated * 1000:.1f}ms")

# ============================================================
# 3. KV-Cache 原理
# ============================================================
print("\n\n" + "=" * 60)
print("【3. KV-Cache 原理与显存计算】")
print("=" * 60)
print("""
自回归生成时，每个新 token 需要和前面所有 token 做 attention。
如果不缓存，每生成一个 token 都要重新计算前面所有 K 和 V。

KV-Cache: 把已计算过的 K、V 矩阵缓存下来，每步只算新 token 的 K、V。

  Without KV-Cache (每步都重算):
    Step 1: 计算 [t1] 的 K,V          → 1 次
    Step 2: 计算 [t1,t2] 的 K,V       → 2 次
    Step 3: 计算 [t1,t2,t3] 的 K,V    → 3 次
    ...总计算量 = 1 + 2 + 3 + ... + N = O(N²)

  With KV-Cache (只算新增的):
    Step 1: 计算 t1 的 K,V → 缓存
    Step 2: 计算 t2 的 K,V → 追加到缓存
    Step 3: 计算 t3 的 K,V → 追加到缓存
    ...总计算量 = N × 1 = O(N)
""")

# 计算 KV-Cache 显存


def estimate_kv_cache_memory(
    num_layers, num_kv_heads, head_dim, seq_len, batch_size=1, dtype_bytes=2
):
    """
    KV-Cache 显存 = 2 × layers × kv_heads × head_dim × seq_len × batch × dtype
                     ^K和V两份
    """
    memory_bytes = (
        2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    )
    return memory_bytes


# Qwen2.5-0.5B 参数
config = model.config
num_layers = config.num_hidden_layers
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
head_dim = config.hidden_size // config.num_attention_heads

print(f"模型参数: {num_layers} layers, {num_kv_heads} KV heads, "
      f"head_dim={head_dim}")

print("\n不同序列长度的 KV-Cache 显存 (FP16, batch=1):")
for seq_len in [512, 2048, 8192, 32768, 131072]:
    mem = estimate_kv_cache_memory(num_layers, num_kv_heads, head_dim, seq_len)
    print(f"  seq_len={seq_len:>7d} → KV-Cache = {mem / 1024**2:>8.1f} MB")

# 多 batch 场景
print("\n不同 batch size 的 KV-Cache 显存 (FP16, seq_len=4096):")
for bs in [1, 8, 32, 128]:
    mem = estimate_kv_cache_memory(num_layers, num_kv_heads, head_dim, 4096, bs)
    print(f"  batch={bs:>3d} → KV-Cache = {mem / 1024**3:.2f} GB")

print("""
【面试考点】
Q: 为什么长上下文场景 KV-Cache 是瓶颈？
A: KV-Cache 和 seq_len 成线性关系。对于 70B 模型 + 128K 上下文，
   单条 KV-Cache 就要几十 GB，严重挤占显存中本可以用于更大 batch 的空间。
""")

# ============================================================
# 4. GQA (Grouped Query Attention) 节省 KV-Cache
# ============================================================
print("\n" + "=" * 60)
print("【4. GQA — 降低 KV-Cache 的模型架构优化】")
print("=" * 60)
print("""
MHA vs GQA vs MQA:

  MHA (Multi-Head Attention):
    每个 attention head 都有独立的 K、V
    KV heads = Q heads
    KV-Cache 最大

  GQA (Grouped Query Attention):
    多个 Q head 共享一组 K、V
    KV heads = Q heads / group_size
    KV-Cache 减小为 MHA 的 1/group_size

  MQA (Multi-Query Attention):
    所有 Q head 共享同一组 K、V
    KV heads = 1
    KV-Cache 最小，但质量可能下降
""")

num_q_heads = config.num_attention_heads

print(f"Qwen2.5-0.5B 的配置:")
print(f"  Q heads:  {num_q_heads}")
print(f"  KV heads: {num_kv_heads}")

if num_kv_heads == num_q_heads:
    print(f"  类型: MHA（没有共享）")
elif num_kv_heads == 1:
    print(f"  类型: MQA（所有 Q head 共享 1 组 KV）")
else:
    group_size = num_q_heads // num_kv_heads
    print(f"  类型: GQA（每 {group_size} 个 Q head 共享 1 组 KV）")
    mha_cache = estimate_kv_cache_memory(
        num_layers, num_q_heads, head_dim, 4096
    )
    gqa_cache = estimate_kv_cache_memory(
        num_layers, num_kv_heads, head_dim, 4096
    )
    print(f"  如果用 MHA: KV-Cache = {mha_cache / 1024**2:.1f} MB")
    print(f"  GQA 实际:   KV-Cache = {gqa_cache / 1024**2:.1f} MB")
    print(f"  节省: {(1 - gqa_cache / mha_cache):.0%}")

# ============================================================
# 5. Continuous Batching
# ============================================================
print("\n\n" + "=" * 60)
print("【5. Continuous Batching（连续批处理）】")
print("=" * 60)
print("""
传统 Static Batching 的问题：
  一个 batch 里所有请求必须等最长的那个生成完才能一起返回，
  短请求白白等着，GPU 利用率低。

  请求1: [████████████████████████████]  (完成但在等)
  请求2: [████████████████]              (完成但在等)
  请求3: [████████████████████████████████████████]  (还在生成)
                                          ↑ 全部必须等到这里

Continuous Batching 的做法：
  某个请求一旦完成（遇到 EOS 或达到最大长度），
  立刻从 batch 中移除，空出的位置立刻插入新请求。

  请求1: [████████████████████████████]
  请求2: [████████████████]  → 请求4: [███████████████]
  请求3: [████████████████████████████████████████]
                             ↑ 请求2完成，请求4立刻加入

  效果：
  ✅ GPU 利用率大幅提升
  ✅ 吞吐量可提升 2-10 倍
  ✅ 短请求的延迟不再被长请求拖累
""")

# 模拟 Continuous Batching
import random
random.seed(42)

num_requests = 12
max_batch = 4
requests = [random.randint(5, 25) for _ in range(num_requests)]

# Static batching 模拟
def simulate_static_batching(requests, max_batch):
    total_steps = 0
    idx = 0
    while idx < len(requests):
        batch = requests[idx:idx + max_batch]
        total_steps += max(batch)  # 等最长的
        idx += max_batch
    return total_steps

# Continuous batching 模拟
def simulate_continuous_batching(requests, max_batch):
    total_steps = 0
    remaining = list(requests)
    active = []

    while remaining or active:
        # 填满 batch
        while len(active) < max_batch and remaining:
            active.append(remaining.pop(0))

        if not active:
            break

        # 每步所有 active 的请求各生成一个 token
        total_steps += 1
        active = [r - 1 for r in active]
        active = [r for r in active if r > 0]  # 移除完成的

    return total_steps

static_steps = simulate_static_batching(requests, max_batch)
cont_steps = simulate_continuous_batching(requests, max_batch)

print(f"模拟 {num_requests} 个请求，batch_size={max_batch}")
print(f"  各请求长度: {requests}")
print(f"  Static Batching:     {static_steps} steps")
print(f"  Continuous Batching: {cont_steps} steps")
print(f"  效率提升: {static_steps / cont_steps:.2f}x")

# ============================================================
# 6. Paged Attention (vLLM)
# ============================================================
print("\n\n" + "=" * 60)
print("【6. Paged Attention (vLLM 的核心技术)】")
print("=" * 60)
print("""
传统 KV-Cache 的显存管理问题：
  - 必须为每个请求预分配最大长度的连续显存
  - 实际生成可能只用很小一部分
  - 大量内部碎片（internal fragmentation）

Paged Attention 的灵感来自操作系统的虚拟内存：

  ┌─────────────────────────────────┐
  │ 物理显存被分成固定大小的 Block  │
  │ (类比操作系统的 Page Frame)     │
  ├─────────────────────────────────┤
  │ 每个请求维护一个 Block Table   │
  │ (类比操作系统的 Page Table)     │
  ├─────────────────────────────────┤
  │ Block 不需要连续，按需分配     │
  │ (类比操作系统的虚拟内存)       │
  └─────────────────────────────────┘

  传统方式 (连续预分配):
    请求1: [K1V1 K2V2 K3V3 K4V4 ___ ___ ___ ___]  ← 后面浪费了
    请求2: [K1V1 K2V2 ___ ___ ___ ___ ___ ___]      ← 更浪费
    → 显存利用率可能只有 30-50%

  Paged Attention (按需分配 Block):
    Block Pool: [B0][B1][B2][B3][B4][B5]...
    请求1 的 Block Table: [B0, B3, B5]     ← 只分配了需要的
    请求2 的 Block Table: [B1, B4]          ← 不连续但没关系
    → 显存利用率接近 100%

额外优势 — Prefix Cache（前缀共享）：
  多个请求共享相同的 system prompt 时，
  KV-Cache 的对应 Block 可以共享，不用重复计算和存储。

  请求1: [system_prompt] + "问题A"
  请求2: [system_prompt] + "问题B"
  → system_prompt 的 KV Block 只存一份，两个请求都指向它
""")

# ============================================================
# 7. Prefix Cache 验证
# ============================================================
print("\n" + "=" * 60)
print("【7. Prefix Cache 效果演示】")
print("=" * 60)

system_prompt = (
    "你是一个专业的 AI 助手，请简洁准确地回答用户的问题。"
    "回答时使用中文，保持专业和友好。"
)

questions = [
    "什么是梯度下降？",
    "解释一下 Transformer 的核心思想。",
    "Python 和 C++ 的主要区别？",
]

print("演示：相同 system prompt + 不同问题")
print(f"System prompt 长度: {len(tokenizer.encode(system_prompt))} tokens\n")

for i, question in enumerate(questions):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"问题 {i + 1}: {question}")
    print(f"  耗时: {(t1 - t0) * 1000:.0f}ms")
    print(f"  回答: {answer[:80]}...")
    print()

print("""
在 vLLM 等推理框架中，如果启用了 Prefix Cache：
  - 第一个请求正常计算 system prompt 的 KV
  - 后续请求直接复用 system prompt 的 KV Block，跳过 prefill
  - 当 system prompt 很长时，加速非常明显
""")

# ============================================================
# 8. Speculative Decoding（推测解码）
# ============================================================
print("\n" + "=" * 60)
print("【8. Speculative Decoding（推测解码）】")
print("=" * 60)
print("""
核心思想：用一个小模型快速"猜"接下来 K 个 token，
然后用大模型一次性验证，接受正确的，拒绝错误的。

  传统自回归（大模型串行生成）:
    大模型 → t1 → 大模型 → t2 → 大模型 → t3 → ...
    每步都要完整走一遍大模型

  Speculative Decoding:
    小模型快速猜 → [t1, t2, t3, t4, t5]   (很快)
    大模型一次验证 → [t1✅, t2✅, t3✅, t4❌, ...]  (一次并行)
    接受 t1, t2, t3，从 t4 重新采样

  为什么能加速？
    - 小模型猜得很快（参数少 10-100 倍）
    - 大模型验证是 prefill（可以并行），比自回归快很多
    - 如果小模型猜对率高（比如 70-80%），总步数大幅减少

  数学保证：
    即使小模型猜错，最终输出的分布仍然完全等价于大模型
    —— 通过拒绝采样（rejection sampling）保证无损

  适用场景：
    - 存在配套的 draft model（如同系列的小模型）
    - 生成内容有一定可预测性（代码、格式化文本等）
    - 不适合内容高度随机的场景（猜对率太低）
""")

# ============================================================
# 9. 推理服务框架对比
# ============================================================
print("\n" + "=" * 60)
print("【9. 主流推理框架对比】")
print("=" * 60)
print("""
┌────────────┬──────────────────────────────────────────────┐
│ 框架       │ 特点                                         │
├────────────┼──────────────────────────────────────────────┤
│ vLLM       │ Paged Attention, Continuous Batching          │
│            │ 高吞吐，支持 OpenAI API 兼容接口             │
│            │ 目前最主流的推理框架                          │
├────────────┼──────────────────────────────────────────────┤
│ TGI        │ HuggingFace 出品，Flash Attention             │
│            │ 适合 HF 生态，部署简单                       │
├────────────┼──────────────────────────────────────────────┤
│ TensorRT-  │ NVIDIA 优化，推理速度最快                    │
│ LLM        │ 但编译耗时长，灵活性差                       │
├────────────┼──────────────────────────────────────────────┤
│ SGLang     │ 程序化控制生成，RadixAttention               │
│            │ 适合复杂 prompt 管理场景                     │
├────────────┼──────────────────────────────────────────────┤
│ llama.cpp  │ CPU 推理，极简部署                           │
│            │ 适合个人/边缘设备                            │
├────────────┼──────────────────────────────────────────────┤
│ ms-swift   │ deploy 命令一键部署                          │
│  (deploy)  │ swift deploy --model xxx                     │
└────────────┴──────────────────────────────────────────────┘

选择推理框架的考量：
  1. 吞吐优先 → vLLM / TensorRT-LLM
  2. 延迟敏感 → TensorRT-LLM / Speculative Decoding
  3. 部署简单 → ms-swift deploy / TGI
  4. CPU/边缘 → llama.cpp
""")

# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 60)
print("【10. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. Prefill 是计算密集，Decode 是带宽密集
  2. KV-Cache 把生成复杂度从 O(N²) 降到 O(N)
  3. GQA 通过共享 KV heads 减少 KV-Cache 大小
  4. Continuous Batching 让短请求不等长请求，显著提升吞吐
  5. Paged Attention 用虚拟内存思想管理 KV-Cache，消除碎片
  6. Speculative Decoding 用小模型加速大模型，数学无损

✅ 面试高频题：
  1. 推理的瓶颈是什么？（Decode 阶段显存带宽）
  2. KV-Cache 大小怎么算？（2 × layers × kv_heads × head_dim × seq_len × dtype）
  3. GQA 和 MHA 的区别？（KV heads 数不同，GQA 节省显存）
  4. vLLM 的核心技术？（Paged Attention + Continuous Batching）
  5. Speculative Decoding 为什么无损？（拒绝采样保证分布一致）
  6. 怎么提高吞吐？（增大 batch, Continuous Batching, Prefix Cache）
  7. 怎么降低延迟？（Speculative Decoding, 量化, 更高显存带宽）

下一课：08_rag_basics.py - RAG 检索增强生成
""")
