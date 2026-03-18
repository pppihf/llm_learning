"""
========================================
 第05课：分布式训练基础
========================================
理解大模型训练为什么必须分布式，以及 DP / TP / PP / ZeRO / FSDP 各自解决什么问题。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/05_distributed_training.py
"""

print("=" * 60)
print(" 第05课：分布式训练基础")
print("=" * 60)


def pretty_ratio(numerator, denominator):
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.2f}x"


print("\n" + "=" * 60)
print("【1. 为什么单卡训练会遇到瓶颈？】")
print("=" * 60)
print(
    """
单卡训练的三个典型瓶颈：
  1. 参数放不下：模型权重、梯度、优化器状态一起占显存
  2. 激活太大：长序列训练时，激活缓存会迅速膨胀
  3. 吞吐不够：即使放得下，训练速度也可能太慢

一个粗略估算：
  训练显存 ≈ 参数 + 梯度 + 优化器状态 + 激活
  AdamW 下，优化器状态通常是参数量的 2 倍左右。
"""
)

print("\n" + "=" * 60)
print("【2. 五种核心并行策略】")
print("=" * 60)

parallel_modes = [
    ("Data Parallel", "每张卡放完整模型，切不同 batch", "实现简单，但显存压力最大"),
    ("Tensor Parallel", "把单层矩阵切到多张卡上算", "适合超大层，但通信频繁"),
    ("Pipeline Parallel", "把不同层切到不同卡", "能扩模型深度，但有 pipeline bubble"),
    ("ZeRO", "分片参数/梯度/优化器状态", "大幅节省显存，是工程主流方案"),
    ("FSDP", "按模块分片参数并按需聚合", "PyTorch 原生，和 ZeRO 思路相近"),
]

for name, idea, tradeoff in parallel_modes:
    print(f"- {name}: {idea}；特点: {tradeoff}")


def plan_parallelism(num_gpus, tp=1, pp=1):
    dp = num_gpus // (tp * pp)
    return {
        "num_gpus": num_gpus,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "world_size": dp * tp * pp,
    }


print("\n" + "=" * 60)
print("【3. 并行度怎么拆？】")
print("=" * 60)
print(
    """
经验上常从这几个问题开始：
  1. 模型能否单卡放下？如果不能，优先考虑 ZeRO / FSDP
  2. 单层矩阵是否太大？如果是，考虑 TP
  3. 层数很多且跨卡可切分？考虑 PP
  4. 剩余卡数是否还能继续放大 batch？再考虑 DP
"""
)

plans = [
    plan_parallelism(num_gpus=8, tp=1, pp=1),
    plan_parallelism(num_gpus=8, tp=2, pp=1),
    plan_parallelism(num_gpus=8, tp=2, pp=2),
    plan_parallelism(num_gpus=16, tp=2, pp=4),
]

for plan in plans:
    print(
        f"- {plan['num_gpus']} 卡: TP={plan['tp']}, PP={plan['pp']}, "
        f"DP={plan['dp']}, world_size={plan['world_size']}"
    )

print("\n" + "=" * 60)
print("【4. ZeRO Stage 1/2/3 的区别】")
print("=" * 60)

zero_stages = [
    (1, "分片优化器状态", "先解决 optimizer state 太大的问题"),
    (2, "分片优化器状态 + 梯度", "继续降低反向阶段显存"),
    (3, "分片优化器状态 + 梯度 + 参数", "显存最省，但通信也最多"),
]

for stage, shard_target, effect in zero_stages:
    print(f"- ZeRO-{stage}: {shard_target}；作用: {effect}")

print("\n" + "=" * 60)
print("【5. 通信为什么会变成瓶颈？】")
print("=" * 60)

baseline_tokens = 1200
dp_tokens = 4200
print(f"单卡吞吐: {baseline_tokens} tokens/s")
print(f"8 卡数据并行后的理想吞吐上限: {baseline_tokens * 8} tokens/s")
print(f"假设实际测得吞吐: {dp_tokens} tokens/s")
print(f"线性加速比: {pretty_ratio(dp_tokens, baseline_tokens)}")
print(f"相对理想加速效率: {dp_tokens / (baseline_tokens * 8):.2%}")
print(
    """
没有线性扩展通常意味着：
  - AllReduce 开销过大
  - DataLoader 跟不上
  - 序列长度/微批设置不合理
  - 节点间带宽弱于卡间带宽
"""
)

print("\n" + "=" * 60)
print("【6. 常见面试问法】")
print("=" * 60)
print(
    """
1. ZeRO-2 和 ZeRO-3 的差别？
   ZeRO-3 连参数也分片，显存更省，但通信更重。

2. FSDP 和 ZeRO-3 的关系？
   思路相近，都是按需聚合参数再计算；一个偏 DeepSpeed 工程体系，一个偏 PyTorch 原生。

3. TP 和 DP 的本质区别？
   DP 复制模型切 batch；TP 不复制完整层，而是切单层计算。

4. 为什么多节点更难调？
   因为开始受网络带宽、拓扑、NCCL 配置、时钟漂移、节点负载不均等问题影响。
"""
)

print("\n下一课：06_alignment_and_dpo.py - 对齐训练与 DPO")