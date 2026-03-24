"""
========================================
 第06课：对齐训练与 DPO
========================================
理解从预训练到对齐的完整链路：SFT → Reward Model → RLHF/DPO。
手动实现 DPO loss，直观理解偏好优化的数学含义。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/06_alignment_and_dpo.py
"""
import torch
import torch.nn.functional as F

print("=" * 60)
print(" 第06课：对齐训练与 DPO")
print("=" * 60)

# ============================================================
# 1. 大模型为什么需要对齐？
# ============================================================
print("\n" + "=" * 60)
print("【1. 大模型为什么需要对齐？】")
print("=" * 60)
print("""
预训练模型只学了"预测下一个 token"，它不知道：
  - 什么回答是有帮助的
  - 什么内容应该拒绝
  - 什么风格是用户期望的

对齐训练的目标是让模型从"能说话"变成"说人话"。

完整训练链路：
  ┌──────────────────────────────────────────────────────────┐
  │ 1. 预训练 (PT)                                          │
  │    在海量文本上学习语言和知识                             │
  │    → 输出：base model（能续写，但不知道怎么回答问题）    │
  ├──────────────────────────────────────────────────────────┤
  │ 2. 监督微调 (SFT)                                       │
  │    在 instruction-response 数据上微调                     │
  │    → 输出：能回答问题，但质量参差不齐                    │
  ├──────────────────────────────────────────────────────────┤
  │ 3. 偏好对齐 (RLHF / DPO)                                │
  │    用人类偏好数据优化回答质量                             │
  │    → 输出：回答质量更高，更符合人类期望                  │
  └──────────────────────────────────────────────────────────┘

【面试考点】
Q: 为什么只做 SFT 不够？
A: SFT 数据中标注者的水平参差不齐，而且"写一个好回答"比
   "判断两个回答哪个更好"要难得多。偏好对齐让模型从"比较"中学习。
""")

# ============================================================
# 2. 偏好数据长什么样？
# ============================================================
print("\n" + "=" * 60)
print("【2. 偏好数据长什么样？】")
print("=" * 60)

preference_dataset = [
    {
        "prompt": "解释一下 Transformer 的核心思想",
        "chosen": "Transformer 的核心是自注意力机制（Self-Attention），它让序列中的每个位置"
                  "都能直接关注到其他所有位置，从而并行建模长距离依赖关系。"
                  "相比 RNN 的串行处理，Transformer 的并行性使其更适合大规模训练。",
        "rejected": "Transformer 就是一种很大的神经网络，效果很好。",
    },
    {
        "prompt": "Python 中 list 和 tuple 的区别？",
        "chosen": "list 是可变的（mutable），可以增删改元素；"
                  "tuple 是不可变的（immutable），创建后不能修改。"
                  "tuple 因为不可变所以可以作为字典的 key，而 list 不行。",
        "rejected": "list 用方括号，tuple 用圆括号，都差不多。",
    },
]

print("偏好数据示例：")
for i, sample in enumerate(preference_dataset):
    print(f"\n样本 {i + 1}:")
    print(f"  Prompt:   {sample['prompt']}")
    print(f"  Chosen:   {sample['chosen'][:60]}...")
    print(f"  Rejected: {sample['rejected'][:60]}...")

print("""
偏好数据的几种常见来源：
  1. 人工标注：标注者对比两个回答，选出更好的
  2. AI 反馈：用更强的模型（如 GPT-4）来判断偏好
  3. 规则筛选：根据长度、格式、关键词等自动构造偏好对
  4. 在线采样：用当前模型生成多个回答，人工/自动选最好和最差
""")

# ============================================================
# 3. RLHF 的完整流程
# ============================================================
print("\n" + "=" * 60)
print("【3. RLHF 的完整流程】")
print("=" * 60)
print("""
RLHF = Reinforcement Learning from Human Feedback

完整三步：
  Step 1: 训练 Reward Model (RM)
    - 输入: (prompt, response)
    - 输出: 标量分数
    - 训练目标: chosen 的分数 > rejected 的分数
    - Loss: -log(sigmoid(r_chosen - r_rejected))  ← Bradley-Terry

  Step 2: 用 PPO 优化策略模型
    - 策略模型生成回答 → Reward Model 打分 → PPO 更新策略
    - 加一个 KL 惩罚，防止策略偏离 SFT 模型太远

  Step 3: 迭代（可选）
    - 用新策略重新采样 → 重新标注 → 重新训练 RM → 继续 PPO

RLHF 的问题：
  ❌ 需要单独训练 Reward Model
  ❌ PPO 训练不稳定，超参敏感
  ❌ 需要同时维护4个模型：策略、参考、RM、价值网络
  ❌ 工程复杂度非常高
""")

# ============================================================
# 4. 手动实现 Reward Model Loss
# ============================================================
print("\n" + "=" * 60)
print("【4. 手动实现 Reward Model Loss】")
print("=" * 60)


def reward_model_loss(reward_chosen, reward_rejected):
    """
    Bradley-Terry 模型：chosen 的 reward 应该高于 rejected
    Loss = -log(sigmoid(r_w - r_l))
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


# 模拟 RM 输出
r_chosen = torch.tensor([2.5, 1.8, 3.1, 0.9])
r_rejected = torch.tensor([1.2, 0.5, 2.8, 1.5])

loss = reward_model_loss(r_chosen, r_rejected)
margins = r_chosen - r_rejected
accuracy = (margins > 0).float().mean()

print("Reward Model 训练示例:")
print(f"  Chosen rewards:  {r_chosen.tolist()}")
print(f"  Rejected rewards: {r_rejected.tolist()}")
print(f"  Margins (r_w - r_l): {margins.tolist()}")
print(f"  RM Loss: {loss.item():.4f}")
print(f"  排序正确率: {accuracy.item():.0%}")

print("""
当 margin > 0（chosen 分数更高）时，loss 小；
当 margin < 0（判断错误）时，loss 大。
RM 训练的目标就是让每对数据的 margin 都 > 0。
""")

# ============================================================
# 5. DPO：不需要 Reward Model 的对齐
# ============================================================
print("\n" + "=" * 60)
print("【5. DPO：不需要 Reward Model 的对齐方法】")
print("=" * 60)
print("""
DPO (Direct Preference Optimization) 的核心思想：

  把 RM + PPO 的两步合成一步，直接在偏好数据上优化策略模型。

DPO 论文证明了：RM 的最优解可以用策略本身来表示：
  r*(x, y) = β * log(π(y|x) / π_ref(y|x)) + constant

于是 DPO loss 变成：
  L = -log σ( β * [ log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) ] )

直觉理解：
  - 括号里的部分 = "当前模型相比参考模型，更偏好 chosen 的程度"
  - β 控制偏好的强度
  - Loss 鼓励模型增大 chosen 的相对概率，减小 rejected 的相对概率

DPO 的优势：
  ✅ 不需要训练 Reward Model
  ✅ 不需要 PPO（训练稳定得多）
  ✅ 只需要两个模型：策略模型 + 参考模型（冻结）
  ✅ 工程上就是一个分类 loss，实现简单
""")

# ============================================================
# 6. 手动实现 DPO Loss
# ============================================================
print("\n" + "=" * 60)
print("【6. 手动实现 DPO Loss】")
print("=" * 60)


def dpo_loss(
    policy_chosen_logps,     # 当前模型对 chosen 的 log π
    policy_rejected_logps,   # 当前模型对 rejected 的 log π
    ref_chosen_logps,        # 参考模型对 chosen 的 log π
    ref_rejected_logps,      # 参考模型对 rejected 的 log π
    beta=0.1,
):
    """
    DPO Loss 实现

    Args:
        *_logps: 各模型对各回答的 log 概率（标量或向量）
        beta: 温度参数，控制偏好强度

    Returns:
        loss, chosen_reward, rejected_reward
    """
    # 计算 log-ratio
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps

    # DPO 的隐式 reward
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss
    logits = beta * (policy_ratio - ref_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss, chosen_reward.mean(), rejected_reward.mean()


# 模拟 log probabilities
policy_chosen = torch.tensor([-1.2, -0.7, -0.4, -0.9])
policy_rejected = torch.tensor([-2.1, -1.3, -1.1, -1.5])
ref_chosen = torch.tensor([-1.4, -0.8, -0.5, -1.0])
ref_rejected = torch.tensor([-1.9, -1.0, -0.8, -1.2])

print("--- DPO Loss 计算 ---")
print(f"Policy log π(chosen):   {policy_chosen.tolist()}")
print(f"Policy log π(rejected): {policy_rejected.tolist()}")
print(f"Ref log π(chosen):      {ref_chosen.tolist()}")
print(f"Ref log π(rejected):    {ref_rejected.tolist()}")

for beta in [0.05, 0.1, 0.5, 1.0]:
    loss, c_reward, r_reward = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
    )
    margin = c_reward - r_reward
    print(f"\n  β={beta:.2f}: loss={loss.item():.4f}, "
          f"chosen_reward={c_reward.item():.4f}, "
          f"rejected_reward={r_reward.item():.4f}, "
          f"margin={margin.item():.4f}")

print("""
观察：
  1. β 越大，loss 对偏好差距更敏感
  2. margin > 0 说明模型确实更偏好 chosen
  3. 训练成功的标志是 chosen_reward 上升，rejected_reward 下降
""")

# ============================================================
# 7. DPO 训练中怎么算 log π(y|x)？
# ============================================================
print("\n" + "=" * 60)
print("【7. DPO 训练中怎么算 log π(y|x)？】")
print("=" * 60)
print("""
对于一个 response y = [t1, t2, ..., tn]：

  log π(y|x) = Σ log P(t_i | x, t_1, ..., t_{i-1})

也就是每个 token 的条件 log 概率之和。

实际实现：
  1. 把 prompt + response 拼起来输入模型
  2. 取模型输出的 logits → log_softmax
  3. 用 response 部分的 token ids 索引对应概率
  4. 求和得到整个 response 的 log π

代码框架（伪代码）：
  logits = model(input_ids).logits
  log_probs = F.log_softmax(logits, dim=-1)

  # 只取 response 部分
  response_log_probs = log_probs[prompt_len:, :]
  response_token_ids = input_ids[prompt_len:]

  # 每个 token 的 log 概率
  per_token_logps = response_log_probs.gather(-1, response_token_ids)

  # 整条 response 的 log 概率
  total_logp = per_token_logps.sum()
""")

# 演示 log probability 计算
print("--- 演示 log π 计算 ---")
vocab_size = 100
seq_len = 5

# 模拟模型输出的 logits
fake_logits = torch.randn(seq_len, vocab_size)
fake_token_ids = torch.randint(0, vocab_size, (seq_len,))
print(fake_token_ids)

# 计算 log probabilities
log_probs = F.log_softmax(fake_logits, dim=-1)
per_token_logps = log_probs.gather(-1, fake_token_ids.unsqueeze(-1)).squeeze(-1)
total_logp = per_token_logps.sum()

print(f"  序列长度: {seq_len}")
print(f"  每 token log π: {per_token_logps.tolist()}")
print(f"  整条 response log π(sum): {total_logp.item():.4f}")

# ============================================================
# 8. DPO 训练实战要点
# ============================================================
print("\n\n" + "=" * 60)
print("【8. DPO 训练实战要点】")
print("=" * 60)
print("""
数据准备：
  - 偏好对通常是 (prompt, chosen, rejected) 三元组
  - chosen 和 rejected 应该是对同一个 prompt 的不同回答
  - 数据量：通常 10K~100K 对即可见效

训练配置：
  - β (beta): 通常 0.1~0.5，太大容易过拟合偏好
  - 学习率: 通常 1e-6 ~ 5e-7（比 SFT 小一个量级）
  - Epoch: 通常 1~3 轮，过多容易退化
  - 参考模型: 冻结的 SFT 模型，不参与训练

ms-swift 中使用 DPO 训练：
  swift rlhf \\
    --rlhf_type dpo \\
    --model /path/to/sft_model \\
    --dataset hh_rlhf_cn \\
    --beta 0.1 \\
    --learning_rate 5e-7

监控指标：
  - loss 应该下降
  - chosen_reward 应该上升
  - rejected_reward 应该下降
  - reward_margin (chosen - rejected) 应该增大
  - reward_accuracy 应该趋近 100%
""")

# ============================================================
# 9. GRPO：不需要 Value Network 的 RL 对齐
# ============================================================
print("\n" + "=" * 60)
print("【9. GRPO：不需要 Value Network 的 RL 对齐】")
print("=" * 60)
print("""
GRPO (Group Relative Policy Optimization) 来自 DeepSeek，
用于 DeepSeek-R1 的训练，是目前最热门的对齐方法之一。

GRPO 的动机：
  PPO 需要 4 个模型（策略、参考、RM、价值网络），太重了。
  DPO 虽然简单，但它只用离线偏好数据，不做在线探索。
  GRPO 想要：在线 RL 的探索能力 + 不需要额外的价值网络。

核心思想 —— 用"组内相对排名"替代 Value Network：
  1. 对于每个 prompt，用当前策略采样 G 个回答
  2. 用 Reward（规则/RM/验证器）给每个回答打分
  3. 在这 G 个回答的分数内部做标准化 → 得到 advantage
  4. 用 advantage 做 PPO-style 的策略梯度更新

  传统 PPO:                GRPO:
  advantage = r - V(s)     advantage = (r - mean(r_group)) / std(r_group)
  需要 Value Network V     不需要 Value Network！

流程对比：
  ┌──────────────────────────────────────────────┐
  │ PPO:  策略 → 生成 → RM打分 → V(s)算advantage │
  │       → clip surrogate loss → 更新策略+V     │
  │       需要 4 个模型                           │
  ├──────────────────────────────────────────────┤
  │ GRPO: 策略 → 采样G个回答 → 打分              │
  │       → 组内标准化算advantage → clip loss    │
  │       → KL惩罚(vs 参考模型) → 更新策略       │
  │       需要 2 个模型（策略 + 参考）            │
  └──────────────────────────────────────────────┘
""")

# 手动实现 GRPO Loss
print("--- 手动实现 GRPO Loss ---")


def grpo_loss(
    log_probs,          # 当前策略对 G 个回答的 log π, shape (G,)
    old_log_probs,      # 采样时策略的 log π_old, shape (G,)
    ref_log_probs,      # 参考模型的 log π_ref, shape (G,)
    rewards,            # G 个回答的 reward, shape (G,)
    clip_eps=0.2,
    beta=0.01,
):
    """
    GRPO Loss 实现

    步骤：
      1. 组内标准化 reward → advantage
      2. 计算 importance ratio: π / π_old
      3. PPO-style clip surrogate loss
      4. 加 KL 惩罚（当前策略 vs 参考模型）
    """
    # Step 1: 组内标准化 → advantage
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Step 2: importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Step 3: PPO clip
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()

    # Step 4: KL 惩罚 (近似: log π - log π_ref)
    kl = (log_probs - ref_log_probs).mean()

    total_loss = policy_loss + beta * kl
    return total_loss, policy_loss, kl, advantage


# 模拟：一个 prompt 采样 G=8 个回答
G = 8
torch.manual_seed(42)
rewards = torch.tensor([0.2, 0.8, 0.5, 0.1, 0.9, 0.3, 0.7, 0.6])

# 模拟各模型的 log prob
old_log_probs = torch.randn(G) * 0.5 - 2.0   # 采样时的策略
log_probs = old_log_probs + torch.randn(G) * 0.1  # 更新后的策略（略有变化）
ref_log_probs = torch.randn(G) * 0.5 - 2.0   # 参考模型

loss, p_loss, kl, adv = grpo_loss(log_probs, old_log_probs, ref_log_probs, rewards)

print(f"  G={G} 个回答的 rewards: {rewards.tolist()}")
print(f"  组内标准化后 advantage: {[f'{a:.2f}' for a in adv.tolist()]}")
print(f"  Policy loss: {p_loss.item():.4f}")
print(f"  KL penalty:  {kl.item():.4f}")
print(f"  Total loss:  {loss.item():.4f}")

print("""
观察 advantage：
  - reward 高于组均值的回答 → advantage > 0 → 增加其概率
  - reward 低于组均值的回答 → advantage < 0 → 降低其概率
  - 不需要训练 Value Network，直接从组内比较得到信号
""")

print("--- GRPO 的 Reward 来源 ---")
print("""
GRPO 的 reward 可以来自多种来源（这是它的灵活之处）：

  1. 规则验证器（DeepSeek-R1 的主要方式）
     - 数学题：答案是否正确
     - 代码题：能否通过测试用例
     - 格式要求：是否按指定格式输出
     → 不需要训练 RM，reward 信号精确可靠

  2. Reward Model
     - 通用对话质量评分
     - 和 PPO 类似，但省掉了 Value Network

  3. 混合 reward
     - 规则分 + RM分 的加权组合
     - 例如: r = 0.7 * correctness + 0.3 * style_score

ms-swift 中使用 GRPO 训练：
  swift rlhf \\
    --rlhf_type grpo \\
    --model /path/to/sft_model \\
    --reward_funcs accuracy format \\
    --num_generations 8 \\
    --learning_rate 1e-6

【面试考点】
Q: GRPO 和 PPO 的核心区别？
A: PPO 用 Value Network 估计 baseline，GRPO 用同一 prompt 下
   多个采样回答的 reward 均值作 baseline，省掉了 Value Network。

Q: GRPO 和 DPO 的核心区别？
A: DPO 是离线方法，只用预先标注好的偏好对；
   GRPO 是在线方法，每步都让当前策略生成新回答并评分。
   在线探索让 GRPO 能发现更好的行为。

Q: 为什么 DeepSeek-R1 用 GRPO 而不是 PPO？
A: 数学/代码任务有确定性的验证器（答案对不对一目了然），
   不需要训练 RM；同时省掉 Value Network 降低了 50% 的
   额外模型开销，工程上更简单。
""")

# ============================================================
# 10. SFT、RLHF、DPO、GRPO 怎么选？
# ============================================================
print("\n" + "=" * 60)
print("【10. SFT、RLHF、DPO、GRPO 对比】")
print("=" * 60)
print("""
┌─────────┬─────────────────────────────────┬──────────────────┐
│ 方法    │ 什么时候用                       │ 工程复杂度       │
├─────────┼─────────────────────────────────┼──────────────────┤
│ SFT     │ 有高质量 instruction-response   │ ★☆☆ 低          │
│         │ 数据，先把格式学对               │                  │
├─────────┼─────────────────────────────────┼──────────────────┤
│ RLHF    │ 需要强控复杂偏好                │ ★★★ 高          │
│ (PPO)   │ 且工程资源充足                  │ 需 4 个模型       │
├─────────┼─────────────────────────────────┼──────────────────┤
│ DPO     │ 有 chosen/rejected 偏好对       │ ★★☆ 中          │
│         │ 想要简单有效的对齐               │ 需 2 个模型       │
├─────────┼─────────────────────────────────┼──────────────────┤
│ GRPO    │ 有可验证的 reward 信号          │ ★★☆ 中          │
│         │ 想要在线探索，不想训练 RM/V     │ 需 2 个模型       │
├─────────┼─────────────────────────────────┼──────────────────┤
│ SimPO   │ 不想维护 reference model        │ ★☆☆ 低          │
│ ORPO    │ 想合并 SFT + 偏好对齐           │ ★☆☆ 低          │
└─────────┴─────────────────────────────────┴──────────────────┘
""")

# ============================================================
# 11. 总结
# ============================================================
print("\n" + "=" * 60)
print("【11. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. 对齐训练让模型从"能说话"变成"说人话"
  2. RLHF 三步：训练 RM → PPO 优化策略 → 迭代
  3. DPO 把 RM + PPO 合成一步，直接优化偏好
  4. GRPO 用组内相对排名替代 Value Network，在线 RL 但更轻量
  5. DPO 是离线方法（用预标注数据），GRPO 是在线方法（即时采样+评分）
  6. β 控制偏好强度 / KL 惩罚强度，reference model 防止模型退化

✅ 面试高频题：
  1. DPO 和 RLHF 的区别？（不需要 RM 和 PPO，一个 loss 搞定）
  2. 为什么需要 reference model？（KL 约束，防止退化）
  3. β 太大/太小会怎样？（太大过拟合偏好，太小学不到偏好）
  4. DPO 的 log π(y|x) 怎么算？（token-level log prob 求和）
  5. GRPO 和 PPO 的区别？（组内 reward 均值替代 Value Network 做 baseline）
  6. GRPO 和 DPO 的区别？（在线 vs 离线，GRPO 有探索能力）
  7. DeepSeek-R1 为什么用 GRPO？（数学/代码有验证器，不需要 RM）
  8. RLHF 需要几个模型？（4个）PPO vs GRPO？（4个 vs 2个）

下一课：07_inference_optimization.py - 推理优化与服务化
""")
