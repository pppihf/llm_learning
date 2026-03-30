"""
========================================
 第11课：GRPO 强化学习微调
========================================
从理论到实战，完整掌握 GRPO（Group Relative Policy Optimization）。
本课结合实际训练 Qwen3-VL-8B 做 GeoQA 几何推理的经验，
融合数学推导与工程实践。

运行方式（学习用，不加载大模型）:
  python /workspace/huangzh14@xiaopeng.com/llm_learning/11_grpo_reinforcement_learning.py

实际训练见:
  /workspace/huangzh14@xiaopeng.com/llm_learning/train_grpo_geoqa.sh
"""
import math
import random
import re

print("=" * 60)
print(" 第11课：GRPO 强化学习微调")
print("=" * 60)

# ============================================================
# 1. 为什么需要 GRPO？DPO 还不够用吗？
# ============================================================
print("\n" + "=" * 60)
print("【1. 为什么需要 GRPO？DPO 还不够用吗？】")
print("=" * 60)
print("""
DPO 和 GRPO 都是对齐训练，但解决的问题不同：

┌─────────────────────────────────────────────────────────┐
│ DPO（Direct Preference Optimization）                   │
│  - 输入: (prompt, chosen, rejected) 三元组              │
│  - 目标: 让模型更喜欢 chosen，不喜欢 rejected           │
│  - 缺点: 需要人工标注的偏好对，成本高                   │
│  - 适用: 通用对话质量提升                               │
├─────────────────────────────────────────────────────────┤
│ GRPO（Group Relative Policy Optimization）              │
│  - 输入: (prompt, verifiable_answer) 二元组             │
│  - 目标: 让模型学会推理出正确答案                       │
│  - 优点: 奖励函数可以自动计算，无需人工标注！           │
│  - 适用: 数学、代码、逻辑推理等有客观答案的任务         │
└─────────────────────────────────────────────────────────┘

GRPO 最出名的应用是 DeepSeek-R1：
  - 用 GRPO 让模型在做题时自发产生"思维链"（Chain of Thought）
  - 模型通过与环境交互（答题→获得奖励）来学习推理
  - 这是真正的强化学习，而不只是监督学习的变体

核心区别：
  SFT/DPO → 告诉模型"你应该这么说"（模仿）
  GRPO    → 告诉模型"你的答案对不对"（试错）
""")

# ============================================================
# 2. GRPO 算法原理
# ============================================================
print("\n" + "=" * 60)
print("【2. GRPO 算法原理】")
print("=" * 60)
print("""
GRPO 由 DeepSeek 在 2024 年提出，是 PPO 的简化版本。

【PPO 的问题】
  PPO 需要单独训练一个「价值网络（Value Network/Critic）」，
  这个网络要和策略网络（Policy）一样大，内存开销翻倍！
  对于 8B、70B 这样的大模型，这是很大的障碍。

【GRPO 的解决方案】
  Group Relative Optimization —— 「组内相对比较」
  
  不需要 Critic！而是用同一批回答相互比较来估算优势：

  步骤 1: 对每个 prompt，生成 G 个不同回答
          prompt → [response₁, response₂, ..., responseG]

  步骤 2: 用奖励函数给每个回答打分
          r = [r₁, r₂, r₃, r₄]  例如：[1.0, 0.0, 1.0, 0.0]

  步骤 3: 计算组内相对优势（标准化）
          mean_r = (r₁ + r₂ + ... + rG) / G
          std_r  = std(r₁, r₂, ..., rG)
          Aᵢ = (rᵢ - mean_r) / (std_r + ε)

  步骤 4: 用优势值更新策略
          好于平均的回答(A>0) → 增大概率
          差于平均的回答(A<0) → 减小概率

  步骤 5: KL 散度约束（防止模型偏离太远）
          Loss = -E[Aᵢ · log π(responseᵢ|prompt)] + β·KL(π||π_ref)
""")

# 用代码直观展示优势计算
print("【优势计算的 Python 实现】")
rewards = [1.0, 0.0, 1.0, 0.0]  # G=4 个生成的奖励
mean_r = sum(rewards) / len(rewards)
std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
eps = 1e-8
advantages = [(r - mean_r) / (std_r + eps) for r in rewards]
print(f"  rewards    = {rewards}")
print(f"  mean       = {mean_r:.2f}")
print(f"  std        = {std_r:.4f}")
print(f"  advantages = {[round(a, 2) for a in advantages]}")
print("  → 正确答案的优势 > 0，模型会朝这个方向优化")
print("  → 错误答案的优势 < 0，这类回答的概率会下降")

# ============================================================
# 3. GRPO 的关键超参数
# ============================================================
print("\n" + "=" * 60)
print("【3. GRPO 的关键超参数】")
print("=" * 60)
print("""
G（num_generations）— 每个 prompt 生成几个回答
  - 太小（G=1）：没有组内比较，退化成 SFT
  - 太大（G=64）：显存爆炸，速度极慢
  - 推荐：G=4~8 （DeepSeek-R1 用了 G=8~64）
  - 实战经验：G=8 + 16张卡，速度比 G=4 + 8张卡相当
    （因为生成是主要瓶颈，G翻倍抵消了卡翻倍的加速）

β（kl_coef）— KL 散度惩罚系数
  - 控制模型偏离原始模型的程度
  - β 太大：模型不敢探索，学不到新东西
  - β 太小：模型乱说，奖励函数被 hack（reward hacking）
  - 默认值：0.01 ~ 0.1，ms-swift 默认用 0.0（通过 clip_ratio 隐式控制）
  - 实战观察：随着训练进行，kl 值从 ~0.001 逐渐升到 ~1.5

clip_ratio — PPO 的 clip 机制
  - 防止策略更新步子太大
  - clip_ratio/high 和 clip_ratio/low 都很小说明训练稳定
  - 如果 clip_ratio 突然飙高，说明学习率太大或 β 太小

max_completion_length — 生成的最大长度
  - GRPO 专用参数（SFT 用 max_new_tokens）
  - 会影响推理质量：太短则模型无法充分思考
  - 实战用 512，观察到 clipped_ratio ≈ 0.3~0.5（适度截断）

frac_reward_zero_std — 组内奖励方差为0的比例
  - 高的值（如0.8）意味着这批 prompt 里所有生成都答对/答错了
  - 这种 batch 学不到东西（优势值都是0）
  - DeepSeek-R1 论文里特别提到过滤这些 batch
""")

# ============================================================
# 4. 奖励函数设计 —— GRPO 的核心
# ============================================================
print("\n" + "=" * 60)
print("【4. 奖励函数设计 —— GRPO 的核心】")
print("=" * 60)
print("""
奖励函数的质量决定了 GRPO 的上限。

【常用奖励类型】

1. 准确率奖励（Accuracy Reward）— 二值奖励
   - 答案对：+1，答案错：0
   - 优点：简单、无歧义
   - 缺点：稀疏，答题很难时梯度信号微弱

2. 格式奖励（Format Reward）— 引导输出格式
   - 有 <think> + <answer>：+1.0
   - 只有 <answer>：+0.5
   - 没有格式：0
   - 作用：鼓励模型产生思维链（CoT）
   - 实战观察：格式奖励初始 0%，训练到 step 300+ 后逐渐上升到 40%

3. 长度惩罚（Length Penalty）— 防止 verbose
   - 奖励 = 准确率 - α * (length / max_length)
   - 防止模型用废话填充来"假装在思考"

4. 过程奖励（Process Reward）— 最难但最强
   - 不只看答案对不对，还打分推理步骤是否正确
   - 需要单独训练 PRM（Process Reward Model）
   - DeepSeek-R1 系列在研究这个方向

【多奖励叠加，ms-swift 写法】
  --reward_funcs geoqa_accuracy geoqa_format
  --reward_weights 1.0 0.3
  
  最终奖励 = 1.0 * accuracy + 0.3 * format
  weight 的选择：主奖励权重高（1.0），辅助奖励权重低（0.1~0.3）
""")

# 展示我们实际用的奖励函数
print("【实战：GeoQA 奖励函数核心逻辑】")
print("""
```python
class GeoQAAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs):
        # solution 字段直接从数据的 "solution": "B" 中读取
        rewards = []
        for completion, sol in zip(completions, solution):
            # 优先提取 <answer>X</answer>
            m = re.search(r'<answer>\\s*([A-Da-d])\\s*</answer>', completion)
            if not m:
                # 退路：选X、答案是X 等中文表达
                m = re.search(r'(?:选|答案[是为]?)\\s*([A-Da-d])', completion)
            pred = m.group(1).upper() if m else ""
            rewards.append(1.0 if pred == sol.strip().upper() else 0.0)
        return rewards

# 注册到 orms —— 这行至关重要！
orms['geoqa_accuracy'] = GeoQAAccuracy
```
""")

# ============================================================
# 5. 数据格式 —— GRPO 和 SFT 的区别
# ============================================================
print("\n" + "=" * 60)
print("【5. 数据格式 —— GRPO 和 SFT 的区别】")
print("=" * 60)
print("""
SFT 数据格式：
  {"messages": [user, assistant], "images": [...]}
  ↑ assistant 就是监督信号（模型要模仿的回答）

GRPO 数据格式：
  {"messages": [system, user], "images": [...], "solution": "B"}
  ↑ 没有 assistant！模型自己生成，用 solution 评分
  ↑ solution 字段通过 kwargs 传递给奖励函数

关键差别：
  - SFT 的 messages 必须包含 assistant 回复
  - GRPO 的 messages 只到 user，不能包含 assistant
  - GRPO 需要一个额外的 "ground truth" 字段（solution/answer/label 都行）

【ms-swift GRPO 数据加载机制】
  swift 会把数据中除"messages"和"images"之外的所有字段
  作为 kwargs 传给奖励函数的 __call__ 方法。
  所以奖励函数的签名要写:
    def __call__(self, completions, solution, **kwargs)
  其中 solution 对应数据里的 "solution" 键。
""")

# ============================================================
# 6. ms-swift GRPO 启动命令详解
# ============================================================
print("\n" + "=" * 60)
print("【6. ms-swift GRPO 启动命令详解】")
print("=" * 60)
print("""
NPROC_PER_NODE=16 \\
CUDA_VISIBLE_DEVICES=0,1,...,15 \\
swift rlhf \\
    --rlhf_type grpo \\              # 指定 GRPO 算法
    --model Qwen3-VL-8B-Instruct \\  # 基座模型
    --external_plugins reward.py \\ # 外部奖励函数文件（重要！）
    --reward_funcs accuracy format \\ # 奖励函数名（对应 orms 注册名）
    --reward_weights 1.0 0.3 \\     # 各奖励权重
    --num_generations 8 \\          # G：每 prompt 生成几个回答
    --max_completion_length 512 \\  # 最大生成长度（GRPO 专用）
    --loss_type grpo \\             # 损失函数类型（必须显式指定）
    --advantage_estimator grpo \\  # 优势估计方法（必须显式指定）
    --tuner_type lora \\            # 用 LoRA 节省显存
    --lora_rank 16 \\
    --freeze_vit true \\            # VLM 特有：冻结视觉编码器
    --freeze_aligner true \\        # VLM 特有：冻结视觉-语言对齐层
    --deepspeed zero2 \\           # 分布式策略
    --split_dataset_ratio 0         # 不自动切分 val，避免 eval 拖慢

【常见坑】
❌ 用了 --val_dataset 且 --eval_steps 过小
   → eval 每次遍历全量测试集，比训练慢 10 倍！
   → 要么删掉 val_dataset，要么把 eval_steps 调到 500+

❌ --max_new_tokens 用在 GRPO 里
   → 报错：参数不识别
   → 正确参数：--max_completion_length

❌ torchrun --nproc_per_node=8 swift rlhf
   → 旧写法，新版 ms-swift 用环境变量方式
   → 正确：NPROC_PER_NODE=8 swift rlhf

❌ 奖励函数文件用错 import 路径
   → 用 from swift.rewards import ORM, orms（包含注册表）
   → 不是 from swift.rewards.orm import ORM
""")

# ============================================================
# 7. 训练过程监控 — 看懂训练日志
# ============================================================
print("\n" + "=" * 60)
print("【7. 训练过程监控 — 看懂训练日志】")
print("=" * 60)
print("""
ms-swift GRPO 训练日志中每条记录的含义：

{
  'loss': 0.001,              # 策略梯度损失（GRPO loss）
  'grad_norm': 0.07,          # 梯度范数，>10 说明训练不稳定
  'learning_rate': 3e-06,     # 当前学习率（含 warmup）

  'completions/mean_length': 330,    # 平均生成长度
  'completions/clipped_ratio': 0.35, # 被 max_length 截断的比例
  
  'reward': 0.7,              # 当前 batch 的综合奖励均值
  'reward_std': 0.16,         # 奖励标准差（越小说明这批题越"单调"）
  'frac_reward_zero_std': 0.6,# 组内方差为0的比例（全对或全错的比例，越小越好学）

  'rewards/GeoQAAccuracy/mean': 0.69, # 准确率奖励
  'rewards/GeoQAFormat/mean': 0.03,   # 格式奖励
  
  'kl': 0.001,                # KL 散度（偏离初始模型的程度）
                              # 随训练增大，过大（>5）说明偏离危险
  'clip_ratio/high_mean': 0.0005,     # PPO clip 触发频率，越小越稳定
  
  'global_step/max_steps': '300/2622'  # 当前进度
  'remaining_time': '15h 30m'
  'train_speed(s/it)': 24.3   # 每步耗时，G=8 + 16卡约 24s
}

【关键指标变化趋势 — 我们实验的结果】
  step 1:    accuracy ≈ 50%（随机水平），format = 0%
  step 200:  accuracy ≈ 67%，format ≈ 3%（格式开始出现）
  step 600:  accuracy ≈ 72%，format ≈ 40%（格式学会了！）
  step 2622: accuracy ≈ 80-90%（峰值），format ≈ 45%
  
  格式奖励比准确率奖励学得慢——因为格式是"软约束"，
  权重只有0.3，而准确率权重1.0。
""")

# ============================================================
# 8. 为什么 GRPO 能产生 Chain-of-Thought？
# ============================================================
print("\n" + "=" * 60)
print("【8. 为什么 GRPO 能产生 Chain-of-Thought？（重要！）】")
print("=" * 60)
print("""
这是很多人困惑的问题：明明只奖励"答案对不对"，为什么模型会学会先思考？

【答案：格式奖励 + 答案奖励的协同作用】

假设两个回答：

  回答 A（无 CoT）:
    "<answer>B</answer>"
    → accuracy = 0.0（答错了），format = 0.5
    → 总奖励 = 0.5 × 0.3 = 0.15

  回答 B（有 CoT）:
    "<think>∠A = 80°，∠B = 60°，则∠ACB = 40°...
     因为 DE∥BC，所以∠CED = 180° - 40° = 140°</think>
     <answer>D</answer>"
    → accuracy = 1.0（答对了），format = 1.0
    → 总奖励 = 1.0 × 1.0 + 1.0 × 0.3 = 1.3

  两者优势差 = (1.3 - 0.15) = 1.15 → 有 CoT 的回答大幅领先！

【更深层的原因】
  对于需要多步推理的题目，模型通过 CoT 步骤来"减少词汇不确定性"。
  每多写一步推理，后续 token 的条件概率就更集中，
  答对概率也随之提升，因此 accuracy 奖励间接鼓励了 CoT。

  这就是为什么无需明确教模型写 <think>，
  它会通过强化学习自发涌现出思维链！
  （DeepSeek-R1 论文里称之为"Aha Moment"）

【我们实验的验证】
  训练初期（step <50）：模型直接输出答案，几乎无推理过程
  训练中期（step 200+）：开始出现推理段落，但没有 <think> 标签
  训练后期（step 600+）：格式奖励 40%，推理过程更规范
  
  模型的推理质量示例（step 2622 左右的表现）:
  ─────────────────────────────────────────
  题: AO 是圆锥的高, OB=0.7, AB=2.5, 求高 AO
  
  模型回答:
  "...AO⊥OB，在直角△AOB中：
   AO² + OB² = AB²
   AO² + 0.49 = 6.25
   AO = √5.76 = 2.4"
  <answer>A</answer>
  ─────────────────────────────────────────
  模型的数学推导是正确的！
""")

# ============================================================
# 9.  GRPO vs PPO vs DPO 横向对比
# ============================================================
print("\n" + "=" * 60)
print("【9. GRPO vs PPO vs DPO 横向对比】")
print("=" * 60)
print("""
┌────────────────┬──────────┬──────────┬──────────────┐
│ 特性            │ PPO      │ DPO      │ GRPO         │
├────────────────┼──────────┼──────────┼──────────────┤
│ 是否需要 Critic │ ✓ 需要   │ ✗ 不需要 │ ✗ 不需要     │
│ 数据需求        │ 奖励标注 │ 偏好对   │ 可验证答案   │
│ 显存开销        │ 2× model │ 2× model │ 1× model     │
│ 适合任务        │ 通用     │ 对话质量 │ 推理/数学/代码│
│ 涌现 CoT 能力   │ 有但弱   │ 无       │ 强！         │
│ 实现复杂度      │ 高       │ 低       │ 中           │
│ 代表作品        │ InstructGPT│Llama-2│ DeepSeek-R1  │
└────────────────┴──────────┴──────────┴──────────────┘

【什么情况用 GRPO？】
  ✓ 任务有客观正确答案（数学题、代码、问答）
  ✓ 希望模型学会推理而非只会背答案
  ✓ 标注数据贵，但验证答案便宜
  ✓ 在已有 SFT 模型上继续提升推理能力

【什么情况不用 GRPO？】
  ✗ 任务答案主观（写文章、配音风格）
  ✗ 相对偏好比绝对正确更重要的场景
  ✗ 没有可靠的奖励函数
""")

# ============================================================
# 10. 手动实现 GRPO Loss（简化版）
# ============================================================
print("\n" + "=" * 60)
print("【10. 手动实现 GRPO Loss（简化版，理解数学原理）】")
print("=" * 60)

import math

def grpo_loss_demo():
    """
    简化版 GRPO loss 计算，帮助理解数学原理。
    不涉及真实的模型推理，只展示核心公式。
    """
    print("\n假设 G=4 个生成，log_probs 是模型给每个 token 序列打的概率对数：")
    
    # 模拟：每个回答的 log_prob（对数概率之和）
    log_probs = [-10.2, -12.5, -9.8, -13.1]   # 当前策略 π_θ
    ref_log_probs = [-10.0, -12.0, -10.0, -12.5]  # 参考策略 π_ref
    rewards = [1.0, 0.0, 1.0, 0.0]            # 奖励

    # Step 1: 计算 KL 散度（每个 token 的 KL 之和）
    kl_penalties = [lp - rlp for lp, rlp in zip(log_probs, ref_log_probs)]
    beta = 0.04  # KL 惩罚系数

    # Step 2: 调整后的奖励
    adjusted_rewards = [r - beta * kl for r, kl in zip(rewards, kl_penalties)]

    # Step 3: 计算组内优势
    mean_r = sum(adjusted_rewards) / len(adjusted_rewards)
    std_r = (sum((r - mean_r)**2 for r in adjusted_rewards) / len(adjusted_rewards)) ** 0.5
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in adjusted_rewards]

    # Step 4: 计算 GRPO loss（最大化期望奖励 = 最小化负奖励）
    # Loss = -mean(Aᵢ × log_ratio)，其中 log_ratio = log π_θ - log π_ref
    log_ratios = [lp - rlp for lp, rlp in zip(log_probs, ref_log_probs)]
    
    # PPO-style clip
    eps_clip = 0.2
    clipped_losses = []
    for A, lr in zip(advantages, log_ratios):
        ratio = math.exp(lr)  # π_θ / π_ref
        loss1 = A * ratio
        loss2 = A * max(1 - eps_clip, min(1 + eps_clip, ratio))  # clipped
        clipped_losses.append(min(loss1, loss2))  # take min (pessimistic)

    grpo_loss = -sum(clipped_losses) / len(clipped_losses)

    print(f"  rewards       = {rewards}")
    print(f"  adjusted_r    = {[round(r, 4) for r in adjusted_rewards]}")
    print(f"  advantages    = {[round(a, 4) for a in advantages]}")
    print(f"  log_ratios    = {[round(lr, 4) for lr in log_ratios]}")
    print(f"  GRPO loss     = {grpo_loss:.6f}")
    print(f"\n  loss > 0 → 还有优化空间，模型继续更新")
    print(f"  loss ≈ 0 → 策略已接近最优（或训练坍缩了）")

grpo_loss_demo()

# ============================================================
# 11. VLM 上做 GRPO 的特殊注意事项
# ============================================================
print("\n\n" + "=" * 60)
print("【11. VLM 上做 GRPO 的特殊注意事项】")
print("=" * 60)
print("""
GRPO 最早用于纯文本 LLM，应用到 VLM（视觉语言模型）有额外挑战：

【架构特殊性】
  VLM = 视觉编码器（ViT）+ 对齐层（Aligner）+ 语言模型（LLM）
  
  训练时通常：
    --freeze_vit true      # 冻结 ViT，节省显存 + 防止视觉特征退化
    --freeze_aligner true  # 冻结对齐层，保持视觉-文本映射不变
    --tuner_type lora      # 只训练 LLM 部分的 LoRA 层

  为什么冻结 ViT？
  → GRPO 的训练信号来自文本奖励，ViT 的梯度信号很弱
  → 解冻 ViT 会消耗大量显存但收益很小
  → 实验表明冻结 ViT 几乎不影响 GRPO 效果

【数据格式中的图像】
  GRPO 多模态数据格式：
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<image>\\n题目文字"}
    ],
    "images": ["/path/to/image.png"],  # 图像路径
    "solution": "B"                     # 标准答案
  }
  
  注意 <image> token 必须在 user 消息的正文里，
  images 列表的顺序和文本中 <image> 出现的顺序对应。

【显存估算】
  模型参数（bfloat16）: 8B × 2bytes = 16GB
  LoRA 梯度 + 优化器状态: ~2GB
  G=8 个生成的 KV cache: 动态，约 1-3GB/卡
  每卡总显存: ~20GB（我们实验的实际观测值）
  
  8B 模型 + G=8 + 16卡 → 每卡 17.6GB / 97.9GB，约 18%
  显存还很充裕，可以增大 batch_size 或 max_completion_length
""")

# ============================================================
# 12. 结果分析与下一步
# ============================================================
print("\n" + "=" * 60)
print("【12. 实验结果与下一步探索】")
print("=" * 60)
print("""
【本次实验结果摘要】
  模型: Qwen3-VL-8B-Instruct
  数据: GeoQA3（3499 train，754 test，中文几何选择题）
  方法: GRPO + LoRA(rank=16) + 冻结 ViT/Aligner
  训练: G=8, 16卡, 3 epochs, 17h22m

  随机基线（瞎蒙 ABCD）: 25%
  基础模型（未训练，估算）: ~50-55%（Qwen3 本身很强）
  GRPO 微调后（20题测试）: 75%

  提升亮点：
  ✓ 准确率显著提升
  ✓ 模型学会了逐步推理（不再直接输出答案）
  ✓ 格式奖励从 0% 升到 40%，部分回答有 <think> 链
  ✓ 训练稳定，无 loss 爆炸

  不足之处：
  ✗ 格式规范率只有 40%，没有完全统一
  ✗ 部分题目 Pred=None（模型输出了推理但没有明确 <answer>）
  ✗ 测试只用了 20 题，需要全量 754 题的统计结果

【改进方向】

  1. 更强的格式奖励
     - 提高 format reward 权重：0.3 → 0.5
     - 加入"没有 <answer> 标签直接得 0"的严格惩罚

  2. 更多训练数据
     - GeoQA3 只有 3499 题，远少于 DeepSeek-R1 的规模
     - 可以混合其他数学推理数据（如 GSM8K、MATH）

  3. 更大的 G 值
     - 从 G=8 增大到 G=16，奖励信号更稳定
     - 需要 32GB/卡以上的显存

  4. 课程学习（Curriculum Learning）
     - 先训简单题（几何基础），再逐步加难
     - 防止初期因为太难导致奖励信号稀疏

  5. 奖励 hacking 防范
     - 监控 kl 值超过阈值（>3）时降低学习率
     - 使用 reward margin 替代直接奖励

  6. 与 SFT 组合（two-stage）
     - 先用 SFT 数据让模型学会基本格式（1 epoch）
     - 再用 GRPO 提升推理能力
     - 参考：OpenAI o1、DeepSeek-R1 都是多阶段训练
""")

print("\n" + "=" * 60)
print("【本课总结】")
print("=" * 60)
print("""
GRPO 是强化学习在 LLM 上的优雅实现：

  核心思想: 「不需要知道怎么做，只需要知道做对了没有」
  
  与 SFT 对比:
    SFT = 老师直接告诉你答案（模仿学习）
    GRPO = 自己做题、老师只批改对错（试错学习）
  
  实现要点:
    1. 数据带 solution 字段（ground truth）
    2. 奖励函数作为外部 plugin 注册到 orms
    3. G 个生成 → 组内标准化 → 策略梯度更新
    4. 去掉 val_dataset 避免评估拖慢训练
  
  结果观察:
    75% 准确率（vs 25% 随机基线），模型涌现出推理能力
    格式奖励引导模型逐步输出 <think> 链

GRPO 是当前「推理型 AI」的核心训练方法。
理解了 GRPO，就理解了 o1/R1/Gemini-Thinking 的训练范式。
""")
