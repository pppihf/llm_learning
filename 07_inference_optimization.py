"""
========================================
 第07课：推理优化与服务化
========================================
从系统角度理解 TTFT、TPS、连续批处理、前缀缓存、Paged Attention 等推理优化手段。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/07_inference_optimization.py
"""

from collections import deque

print("=" * 60)
print(" 第07课：推理优化与服务化")
print("=" * 60)

print("\n" + "=" * 60)
print("【1. 推理服务最关注什么指标？】")
print("=" * 60)

metrics = [
    ("TTFT", "Time To First Token", "用户看到首字的延迟"),
    ("TPS", "Tokens Per Second", "单请求生成速度"),
    ("Throughput", "吞吐", "单位时间能处理多少请求/多少 token"),
    ("P99 Latency", "尾延迟", "最慢那批请求是否可接受"),
]

for short_name, full_name, meaning in metrics:
    print(f"- {short_name} ({full_name}): {meaning}")

print("\n" + "=" * 60)
print("【2. 为什么静态 batch 往往不够好？】")
print("=" * 60)
print(
    """
在线服务里，请求长度不一致、到达时间不一致。
如果坚持静态 batch：
  - 短请求会被长请求拖慢
  - 空闲 token budget 无法复用
  - GPU 不能持续吃满

所以工业界通常会采用 continuous batching（连续批处理）。
"""
)


def simulate_continuous_batching(request_lengths, max_batch_tokens):
    queue = deque(enumerate(request_lengths, start=1))
    active = []
    step = 0

    while queue or active:
        step += 1

        while queue and sum(length for _, length in active) < max_batch_tokens:
            request_id, request_len = queue[0]
            if sum(length for _, length in active) + request_len > max_batch_tokens and active:
                break
            queue.popleft()
            active.append([request_id, request_len])

        print(f"Step {step}: active={[request_id for request_id, _ in active]}")

        next_active = []
        for request_id, remain in active:
            if remain > 1:
                next_active.append([request_id, remain - 1])
        active = next_active


print("连续批处理模拟:")
simulate_continuous_batching(request_lengths=[6, 2, 5, 3], max_batch_tokens=8)

print("\n" + "=" * 60)
print("【3. 前缀缓存和 Paged Attention】")
print("=" * 60)
print(
    """
前缀缓存（Prefix Cache）:
  如果多个请求共享 system prompt 或共享长前缀，KV-Cache 可以复用。

Paged Attention:
  把 KV-Cache 组织成“页”，避免连续大块内存申请造成碎片。
  这也是 vLLM 能把在线推理做得更稳的重要原因之一。

核心目标都一样：
  尽量少重复算，尽量少浪费显存，尽量让 GPU 始终有活干。
"""
)

print("\n" + "=" * 60)
print("【4. 除了量化，常见优化还有哪些？】")
print("=" * 60)

optimizations = [
    "Continuous Batching：提高整体吞吐",
    "Prefix Caching：复用共享上下文的 KV-Cache",
    "Paged Attention：降低 KV-Cache 内存碎片",
    "Speculative Decoding：用小模型草拟，大模型验证",
    "Prompt Compression：缩短无效上下文，减少 prefill 成本",
]

for item in optimizations:
    print(f"- {item}")

print("\n" + "=" * 60)
print("【5. 常见面试问法】")
print("=" * 60)
print(
    """
1. 在线服务为什么 often decode 比 prefill 更难优化？
   因为 decode 每次只生成一个 token，算力利用率低，容易被显存读写限制。

2. 为什么 vLLM 吞吐高？
   因为它在调度、KV-Cache 管理、Paged Attention、连续批处理上做得很强。

3. Prefix Cache 最适合什么场景？
   大量共享 system prompt、工具说明、长模板上下文的场景。

4. 只看平均延迟为什么不够？
   因为线上真正影响体验的是尾延迟，尤其是 P95 / P99。
"""
)

print("\n下一课：08_rag_basics.py - RAG 系统基础")