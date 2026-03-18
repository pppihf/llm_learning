"""
========================================
 第06课：对齐训练与 DPO
========================================
理解 SFT、RLHF、DPO 的关系，并用一个最小例子直观看 DPO loss 的含义。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/06_alignment_and_dpo.py
"""

import math

import torch

print("=" * 60)
print(" 第06课：对齐训练与 DPO")
print("=" * 60)

print("\n" + "=" * 60)
print("【1. 对齐训练全景图】")
print("=" * 60)
print(
    """
常见对齐链路：
  1. 预训练：学语言和世界知识
  2. SFT：学“该怎么回答”
  3. Preference Learning：学“哪个回答更好”
  4. RLHF / DPO：让模型偏向人类偏好的回答

DPO 的优势是：
  - 不需要显式训练 reward model 再跑 PPO
  - 训练目标更直接
  - 工程复杂度通常低于 RLHF
"""
)

print("\n" + "=" * 60)
print("【2. 偏好数据长什么样？】")
print("=" * 60)

preference_sample = {
    "prompt": "解释一下 Transformer 的核心思想",
    "chosen": "Transformer 通过自注意力让每个 token 直接和其他位置交互。",
    "rejected": "Transformer 就是一个更大的 RNN。",
}

for key, value in preference_sample.items():
    print(f"- {key}: {value}")

print("\n" + "=" * 60)
print("【3. DPO loss 直觉】")
print("=" * 60)
print(
    """
DPO 想做的事很直接：
  - 提高模型对 chosen 回答的相对偏好
  - 降低模型对 rejected 回答的相对偏好

一种常见写法可以概括为：
  loss = -log(sigmoid(beta * ((log π(y_w) - log π(y_l)) - (log π_ref(y_w) - log π_ref(y_l)))))

其中：
  - y_w 是 chosen
  - y_l 是 rejected
  - π 是当前模型
  - π_ref 是参考模型
"""
)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    policy_gap = policy_chosen - policy_rejected
    ref_gap = ref_chosen - ref_rejected
    logits = beta * (policy_gap - ref_gap)
    return -torch.log(torch.sigmoid(logits))


policy_chosen = torch.tensor([-1.2, -0.7, -0.4])
policy_rejected = torch.tensor([-2.1, -1.3, -1.1])
ref_chosen = torch.tensor([-1.4, -0.8, -0.5])
ref_rejected = torch.tensor([-1.9, -1.0, -0.8])

for beta in [0.05, 0.1, 0.5]:
    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)
    print(f"beta={beta:.2f} -> 平均 DPO loss: {loss.mean().item():.4f}")

print("\n" + "=" * 60)
print("【4. SFT、RLHF、DPO 怎么选？】")
print("=" * 60)

choices = [
    ("SFT", "有高质量标准答案数据", "先把回答风格和格式学稳"),
    ("RLHF", "要强控复杂偏好，且工程资源充足", "效果强，但链路长、调参重"),
    ("DPO", "已经有 chosen/rejected 偏好对", "实现简单，是当前常用选项"),
]

for method, use_case, reason in choices:
    print(f"- {method}: 适用场景: {use_case}；原因: {reason}")

print("\n" + "=" * 60)
print("【5. 常见面试问法】")
print("=" * 60)
print(
    """
1. DPO 和 RLHF 的区别？
   DPO 直接在偏好对上优化，不需要 PPO 那套在线强化学习流程。

2. 为什么还要 reference model？
   因为要约束策略不要偏离初始模型太远，避免训练发散或语言质量退化。

3. beta 越大意味着什么？
   越强调 chosen 和 rejected 的偏好差距，更新会更“激进”。

4. 只有 SFT 数据，没有偏好对怎么办？
   先做 SFT；如果要继续对齐，再补偏好标注或用模型蒸馏生成偏好对。
"""
)

print("\n下一课：07_inference_optimization.py - 推理优化与服务化")