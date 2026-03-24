"""
========================================
 第05课：分布式训练基础
========================================
大模型训练离不开分布式。本课从显存估算出发，
理解 DP / TP / PP / ZeRO / FSDP 各自解决什么问题，
并用 PyTorch 代码实际演示梯度同步和梯度累积。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/05_distributed_training.py
"""
import torch
import torch.nn as nn

print("=" * 60)
print(" 第05课：分布式训练基础")
print("=" * 60)

# ============================================================
# 1. 训练显存到底花在哪？
# ============================================================
print("\n" + "=" * 60)
print("【1. 训练显存到底花在哪？】")
print("=" * 60)
print("""
一个参数在训练时要占多少显存？以 AdamW + BF16 混合精度为例：

┌──────────────────────┬──────────────┬───────────────────────────┐
│ 组成部分             │ 每参数字节   │  7B 模型估算              │
├──────────────────────┼──────────────┼───────────────────────────┤
│ 参数 (BF16)          │ 2 bytes      │  14 GB                    │
│ 梯度 (BF16)          │ 2 bytes      │  14 GB                    │
│ 优化器状态 (FP32)    │ 12 bytes     │  84 GB                    │
│   - 参数主副本       │   4 bytes    │    (AdamW 需要 FP32 副本) │
│   - 一阶动量 m       │   4 bytes    │    (FP32，保数值稳定)     │
│   - 二阶动量 v       │   4 bytes    │    (FP32，保数值稳定)     │
├──────────────────────┼──────────────┼───────────────────────────┤
│ 合计 (不含激活)      │ 16 bytes     │  ~112 GB                  │
│ + 激活 (视序列长度)  │ 变化大       │  额外 10~50+ GB           │
└──────────────────────┴──────────────┴───────────────────────────┘

结论：7B 模型全量训练至少要 ~112GB（不含激活），单张 80GB A100 放不下。
这就是为什么大模型训练必须分布式。

【面试考点】
Q: 为什么优化器状态占这么多？
A: AdamW 需要维护每个参数的一阶动量 m 和二阶动量 v，
   都以 FP32 存储（为了数值稳定），再加上 FP32 的参数主副本，
   所以是 4+4+4=12 bytes/param。加上 BF16 参数和梯度各 2 bytes，总共 16 bytes/param。
""")

# ============================================================
# 2. 动手算：训练显存估算器
# ============================================================
print("\n" + "=" * 60)
print("【2. 动手算：训练显存估算器】")
print("=" * 60)


def estimate_training_memory(
    num_params_billion,
    param_dtype_bytes=2,
    zero_stage=0,
    num_gpus=1,
    seq_len=2048,
    batch_size=1,
    hidden_size=4096,
    num_layers=32,
):
    """估算训练显存（GB），不含系统开销"""
    P = num_params_billion * 1e9

    param_mem = P * param_dtype_bytes
    grad_mem = P * param_dtype_bytes
    opt_mem = P * 8  # AdamW: FP32 副本 + m + v

    if zero_stage == 0:
        per_gpu = param_mem + grad_mem + opt_mem
    elif zero_stage == 1:
        per_gpu = param_mem + grad_mem + opt_mem / num_gpus
    elif zero_stage == 2:
        per_gpu = param_mem + grad_mem / num_gpus + opt_mem / num_gpus
    elif zero_stage == 3:
        per_gpu = (param_mem + grad_mem + opt_mem) / num_gpus
    else:
        per_gpu = param_mem + grad_mem + opt_mem

    # 激活估算（粗略）
    activation_per_layer = 2 * seq_len * batch_size * hidden_size * param_dtype_bytes
    activation_total = activation_per_layer * num_layers

    return {
        "params_gb": param_mem / 1e9,
        "grads_gb": grad_mem / 1e9,
        "optimizer_gb": opt_mem / 1e9,
        "activation_gb": activation_total / 1e9,
        "per_gpu_gb": per_gpu / 1e9,
        "total_gb": (per_gpu + activation_total) / 1e9,
    }


configs = [
    {"label": "7B 单卡 无ZeRO",  "num_params_billion": 7,  "zero_stage": 0, "num_gpus": 1},
    {"label": "7B 8卡 ZeRO-1",   "num_params_billion": 7,  "zero_stage": 1, "num_gpus": 8},
    {"label": "7B 8卡 ZeRO-2",   "num_params_billion": 7,  "zero_stage": 2, "num_gpus": 8},
    {"label": "7B 8卡 ZeRO-3",   "num_params_billion": 7,  "zero_stage": 3, "num_gpus": 8},
    {"label": "72B 8卡 ZeRO-3",  "num_params_billion": 72, "zero_stage": 3, "num_gpus": 8,
     "hidden_size": 8192, "num_layers": 80},
]

for cfg in configs:
    label = cfg.pop("label")
    result = estimate_training_memory(**cfg)
    print(f"\n{label}:")
    print(f"  参数 = {result['params_gb']:.1f} GB  梯度 = {result['grads_gb']:.1f} GB  "
          f"优化器 = {result['optimizer_gb']:.1f} GB")
    print(f"  每卡(不含激活) = {result['per_gpu_gb']:.1f} GB  "
          f"激活 ≈ {result['activation_gb']:.1f} GB  总计 ≈ {result['total_gb']:.1f} GB")

# ============================================================
# 3. 五种并行策略
# ============================================================
print("\n\n" + "=" * 60)
print("【3. 五种核心并行策略】")
print("=" * 60)
print("""
┌──────────────────┬──────────────────────────────────────────────────┐
│ Data Parallel    │ 每张卡放完整模型副本，切不同 mini-batch           │
│ (DP/DDP)         │ 前向+反向后 AllReduce 梯度，再各自更新           │
│                  │ ✅ 实现简单  ❌ 显存没省（每卡都有全量参数）      │
├──────────────────┼──────────────────────────────────────────────────┤
│ Tensor Parallel  │ 把单层的矩阵切到多卡上并行计算                   │
│ (TP)             │ 例: 4096×4096 的 Q_proj 切成 4 块 4096×1024      │
│                  │ ✅ 减少单层显存  ❌ 每层都要通信（AllReduce/P2P） │
├──────────────────┼──────────────────────────────────────────────────┤
│ Pipeline Parallel│ 把不同层分配到不同卡上                           │
│ (PP)             │ 例: Layer 0-15 在 GPU0, Layer 16-31 在 GPU1     │
│                  │ ✅ 减少单卡层数  ❌ pipeline bubble 降低利用率    │
├──────────────────┼──────────────────────────────────────────────────┤
│ ZeRO             │ 在 DP 基础上分片优化器状态/梯度/参数             │
│ (DeepSpeed)      │ Stage 1: 分片优化器 Stage 2: +梯度 Stage 3: +参数│
│                  │ ✅ 显存大幅节省  ❌ 通信开销随 Stage 增大         │
├──────────────────┼──────────────────────────────────────────────────┤
│ FSDP             │ PyTorch 原生的 ZeRO-3 等价实现                   │
│ (Fully Sharded)  │ 按模块分片参数，前向时按需 AllGather             │
│                  │ ✅ 原生集成  ❌ 和 ZeRO-3 一样通信开销重          │
└──────────────────┴──────────────────────────────────────────────────┘

【面试考点】
Q: DP 和 DDP 的区别？
A: DP 是 PyTorch 老接口，用单进程多线程 + GIL，效率差；
   DDP 每个 GPU 一个进程，用 NCCL AllReduce，是标准做法。

Q: ZeRO-2 和 ZeRO-3 怎么选？
A: 优先 ZeRO-2（通信少、速度快），放不下再用 ZeRO-3。
   ZeRO-3 每次前向都要 AllGather 参数，通信量约是 ZeRO-2 的 1.5x。
""")

# ============================================================
# 4. ZeRO Stage 1/2/3 详解
# ============================================================
print("\n" + "=" * 60)
print("【4. ZeRO Stage 1/2/3 详解】")
print("=" * 60)
print("""
假设有 N 张卡，模型参数量为 Φ：

           │ 每卡存储（不含激活）          │ 额外通信量(vs DDP)
───────────┼──────────────────────────────┼──────────────
无ZeRO(DDP)│ 2Φ + 2Φ + 12Φ = 16Φ bytes   │ 基准 (2Φ AllReduce)
───────────┼──────────────────────────────┼──────────────
ZeRO-1     │ 2Φ + 2Φ + 12Φ/N             │ 无额外通信
───────────┼──────────────────────────────┼──────────────
ZeRO-2     │ 2Φ + 2Φ/N + 12Φ/N           │ 无额外通信
───────────┼──────────────────────────────┼──────────────
ZeRO-3     │ (2Φ + 2Φ + 12Φ)/N = 16Φ/N   │ +1Φ AllGather
───────────┴──────────────────────────────┴──────────────

注意 ZeRO-1 和 ZeRO-2 的通信量和 DDP 一样！
只有 ZeRO-3 因为参数也分片了，需要额外 AllGather。

【面试考点】
Q: 为什么 ZeRO-2 通信量不增加却能省更多显存？
A: 因为它把 AllReduce 拆成 Reduce-Scatter + AllGather，
   每卡只保留自己负责的那一片梯度，通信量不变但显存少了。
""")

# ============================================================
# 5. 实际演示：梯度累积
# ============================================================
print("\n" + "=" * 60)
print("【5. 实际演示：梯度累积（Gradient Accumulation）】")
print("=" * 60)
print("""
梯度累积是分布式训练中的重要技巧：
  - 显存不够时，用小 batch 多次前向+反向，累积梯度后再更新
  - 效果等价于用大 batch 训练
  - 数学上：grad = (grad_1 + grad_2 + ... + grad_K) / K

例子：想用 batch_size=32 但显存只够 batch_size=4
  → 设 gradient_accumulation_steps=8
  → 每 8 步攒够梯度再 optimizer.step()
""")

model = nn.Linear(64, 16)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

accum_steps = 4
micro_batch = 2

print(f"\n--- 演示: micro_batch={micro_batch}, accum_steps={accum_steps}, "
      f"等效 batch={micro_batch * accum_steps} ---")

optimizer.zero_grad()
total_loss = 0.0

for step in range(accum_steps):
    x = torch.randn(micro_batch, 64)
    target = torch.randint(0, 16, (micro_batch,))
    logits = model(x)
    print(logits.shape, target.shape)
    loss = loss_fn(logits, target) / accum_steps  # ← 关键：除以累积步数

    loss.backward()  # 梯度自动累加到 .grad
    total_loss += loss.item()

    grad_norm = model.weight.grad.norm().item()
    print(f"  Step {step + 1}/{accum_steps}: loss={loss.item():.4f}, "
          f"累积梯度范数={grad_norm:.4f}")

optimizer.step()
optimizer.zero_grad()
print(f"\n  等效 batch loss: {total_loss:.4f}")
print("  optimizer.step() 只在累积结束后调用一次")

print("""
【关键注意事项】
  1. loss 要除以 accum_steps，否则等效学习率会偏大
  2. optimizer.step() 只在累积结束后调用一次
  3. 配合 DDP 使用时，中间步骤可以关闭梯度同步来加速：
     with model.no_sync():  # 中间步不 AllReduce
         loss.backward()
""")

# ============================================================
# 6. AllReduce 是什么？
# ============================================================
print("\n" + "=" * 60)
print("【6. AllReduce 是什么？为什么它是 DDP 的核心？】")
print("=" * 60)
print("""
AllReduce = Reduce + Broadcast

示意（4 张卡，每卡有一个梯度向量）：

  AllReduce 前:            AllReduce(sum) 后:
  GPU0: [1, 2]             GPU0: [16, 20]
  GPU1: [3, 4]      →     GPU1: [16, 20]
  GPU2: [5, 6]             GPU2: [16, 20]
  GPU3: [7, 8]             GPU3: [16, 20]

DDP 用 AllReduce(mean) 来保证每张卡拿到相同的平均梯度，
从而保证每张卡的参数更新一致。
""")

# 模拟 AllReduce
fake_grads = [
    torch.tensor([1.0, 2.0]),
    torch.tensor([3.0, 4.0]),
    torch.tensor([5.0, 6.0]),
    torch.tensor([7.0, 8.0]),
]

print("--- 模拟 AllReduce ---")
print("前:")
for i, g in enumerate(fake_grads):
    print(f"  GPU{i}: {g.tolist()}")

reduced = sum(fake_grads)
averaged = reduced / len(fake_grads)

print(f"\nAllReduce(sum):  {reduced.tolist()}")
print(f"AllReduce(mean): {averaged.tolist()}  ← DDP 用这个更新参数")

# ============================================================
# 7. 并行度怎么拆？
# ============================================================
print("\n\n" + "=" * 60)
print("【7. 并行度怎么拆？工程决策框架】")
print("=" * 60)
print("""
拿到 N 张卡后，决策顺序：

┌─────────────────────────────────────────────────────┐
│ Step 1: 模型能否单卡放下？                           │
│   能 → 直接 DDP（最简单、通信最少）                  │
│   不能 → 继续往下                                    │
├─────────────────────────────────────────────────────┤
│ Step 2: 用 ZeRO/FSDP 能否放下？                     │
│   ZeRO-2 通常够 → 优先 ZeRO-2                       │
│   ZeRO-2 不够 → 尝试 ZeRO-3 / FSDP                 │
├─────────────────────────────────────────────────────┤
│ Step 3: 单层矩阵是否超大？                           │
│   是（如 hidden_size > 8192）→ 加 TP                 │
│   否 → 通常不需要 TP                                 │
├─────────────────────────────────────────────────────┤
│ Step 4: 跨多节点？                                   │
│   节点内 TP（NVLink 快），节点间 DP/PP（网络慢）     │
│   典型配置: TP=8（一机）× DP=N/8（跨机）            │
└─────────────────────────────────────────────────────┘
""")


def plan_parallelism(num_gpus, tp=1, pp=1):
    dp = num_gpus // (tp * pp)
    return {"gpus": num_gpus, "TP": tp, "PP": pp, "DP": dp}


examples = [
    plan_parallelism(8, tp=1, pp=1),
    plan_parallelism(8, tp=2, pp=1),
    plan_parallelism(8, tp=2, pp=2),
    plan_parallelism(16, tp=4, pp=2),
    plan_parallelism(64, tp=8, pp=1),
]

print("--- 常见并行度拆分示例 ---")
for p in examples:
    print(f"  {p['gpus']:2d} 卡: TP={p['TP']}, PP={p['PP']}, DP={p['DP']}")

# ============================================================
# 8. DeepSpeed 配置速查
# ============================================================
print("\n\n" + "=" * 60)
print("【8. DeepSpeed ZeRO 配置速查】")
print("=" * 60)
print("""
最常用的 ZeRO-2 配置（JSON 格式）：
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 1
}

启动命令：
  # 单机 8 卡
  torchrun --nproc_per_node=8 train.py --deepspeed ds_config.json

  # 多机（2 节点 × 8 卡）
  torchrun --nnodes=2 --nproc_per_node=8 \\
    --node_rank=$RANK --master_addr=$MASTER train.py

ms-swift 封装（更简单）：
  swift sft --deepspeed zero2
  swift sft --deepspeed zero3

【面试考点】
Q: overlap_comm 是干什么的？
A: 让通信和计算重叠。反向传播时，已算完的层开始通信梯度，
   而其他层继续计算。这是隐藏通信延迟的关键优化。

Q: gradient_clipping 为什么重要？
A: 防止梯度爆炸。分布式训练时梯度可能异常大，
   裁剪到 max_norm=1.0 是标准做法。
""")

# ============================================================
# 9. 通信瓶颈分析
# ============================================================
print("\n" + "=" * 60)
print("【9. 通信瓶颈：加速效率怎么算？】")
print("=" * 60)

single_tps = 1200
perf_data = [
    ("2 卡 DDP",    2, 2300),
    ("4 卡 DDP",    4, 4400),
    ("8 卡 DDP",    8, 8200),
    ("8 卡 ZeRO-2", 8, 7800),
    ("8 卡 ZeRO-3", 8, 6500),
]

print(f"基准: 单卡 {single_tps} tokens/s\n")
for name, n, actual in perf_data:
    ideal = single_tps * n
    print(f"  {name}: {actual} t/s, 加速比 {actual / single_tps:.1f}x, "
          f"线性效率 {actual / ideal:.0%}")

print("""
效率不到 100% 的常见原因：
  1. AllReduce 通信开销（和参数量成正比）
  2. DataLoader I/O 跟不上 GPU 计算速度
  3. 节点间带宽 << 节点内 NVLink 带宽
  4. GPU 之间负载不均（数据长度差异大时）
  5. pipeline bubble（PP 模式下）
""")

# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 60)
print("【10. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. 训练显存 = 参数 + 梯度 + 优化器状态 + 激活
  2. AdamW 优化器状态 = 8 bytes/param（FP32 副本 + m + v）
  3. ZeRO 通过分片降低每卡负担，Stage 越高省越多通信越重
  4. 梯度累积用小 batch 模拟大 batch，loss 要除以累积步数
  5. AllReduce = Reduce + Broadcast，保证 DDP 参数一致

✅ 面试高频题：
  1. 训练 7B 模型需要多少显存？怎么估算？
  2. ZeRO-1/2/3 分别分片什么？通信量有何变化？
  3. DP vs DDP？为什么 DP 效率差？
  4. 梯度累积的 loss 为什么要除以步数？
  5. 多节点训练最大的瓶颈是什么？（节点间带宽）
  6. TP 和 PP 分别适用什么场景？

下一课：06_alignment_and_dpo.py - 对齐训练与 DPO
""")
