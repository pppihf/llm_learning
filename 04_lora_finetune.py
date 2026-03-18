"""
========================================
 第04课：LoRA 微调入门
========================================
LoRA 是最流行的高效微调方法。本课从原理到实践，
在 Qwen2-0.5B 上完成一次完整的 LoRA 微调。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/04_lora_finetune.py
"""
import torch
import torch.nn as nn
import os
import json
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(" 第04课：LoRA 微调入门")
print("=" * 60)

# ============================================================
# 1. 为什么需要高效微调？
# ============================================================
print("\n" + "=" * 60)
print("【1. 为什么需要高效微调 (PEFT)？】")
print("=" * 60)
print("""
全量微调 (Full Fine-tuning) 的问题：
  - 7B 模型，FP16 训练需要 ~14GB 参数 + ~14GB 梯度 + ~28GB 优化器状态
  - 总计需要 ~56GB+ 显存，单卡放不下
  - 每个下游任务都要保存一份完整模型副本

高效微调 (Parameter-Efficient Fine-Tuning, PEFT):
  只微调少量参数，冻结大部分预训练权重

主流 PEFT 方法：
  ┌──────────┬───────────────────────────────────────────┐
  │ LoRA     │ 低秩分解，在权重旁添加小矩阵 ★★★ 最常用  │
  │ QLoRA    │ 量化 + LoRA，进一步节省显存               │
  │ Adapter  │ 在层间插入小型网络                        │
  │ Prefix   │ 在输入前添加可学习的虚拟 token             │
  │ P-Tuning │ 类似 Prefix，但更灵活                     │
  │ IA³      │ 学习激活的缩放因子                        │
  └──────────┴───────────────────────────────────────────┘
""")

# ============================================================
# 2. LoRA 原理详解
# ============================================================
print("\n" + "=" * 60)
print("【2. LoRA 原理详解】")
print("=" * 60)
print("""
核心思想：预训练权重的更新矩阵是低秩的

  W_new = W_pretrained + ΔW
  ΔW = A × B  (低秩分解)

  W: (d × d) 原始矩阵
  A: (d × r) 降维矩阵  (用高斯随机初始化)
  B: (r × d) 升维矩阵  (用零初始化)
  r << d (rank, 通常 r=8, 16, 32, 64)

  训练时: 冻结 W，只训练 A 和 B
  推理时: 可以合并 W_new = W + A×B，无额外延迟

参数量对比：
  原始矩阵: d × d = 4096 × 4096 = 16.7M
  LoRA (r=16): d × r + r × d = 4096 × 16 × 2 = 131K
  参数量减少: 16.7M / 131K = 128x ！

通常对哪些层加 LoRA？
  - Attention 的 Q, K, V, O 投影矩阵 (最常见)
  - FFN 的 gate/up/down 矩阵 (可选)
  - 一般不对 Embedding 和 LM Head 加 LoRA

【面试高频考点】
Q: LoRA 的 rank r 怎么选？
A: 典型值 8-64。r 越大拟合能力越强但参数越多。
   简单任务 r=8 就够，复杂任务可用 r=32-64。
   经验法则：如果 loss 不下降尝试增大 r。

Q: LoRA 为什么初始化 B=0？
A: 保证训练开始时 ΔW = A×B = 0，模型输出和原始模型一致，
   从一个好的起点开始微调（不会破坏预训练知识）。

Q: LoRA 和 全量微调 效果差多少？
A: 论文证明在多数任务上 LoRA 和全量微调效果接近甚至相当。
   对于简单的下游适配任务，LoRA 通常足够。
   对于需要学习全新领域知识的任务，全量微调仍有优势。
""")

# ============================================================
# 3. 手动实现 LoRA Layer
# ============================================================
print("\n" + "=" * 60)
print("【3. 手动实现 LoRA Layer】")
print("=" * 60)


class LoRALinear(nn.Module):
    """手动实现 LoRA 层，帮助理解原理"""

    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 冻结原始权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        # 原始输出 + LoRA 增量
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output

    def merge(self):
        """合并 LoRA 权重到原始层（推理时使用，无额外开销）"""
        with torch.no_grad():
            merged_weight = (self.lora_A @ self.lora_B) * self.scaling
            self.original_layer.weight.data += merged_weight.T


# 演示手动 LoRA
print("手动 LoRA 演示:")
original = nn.Linear(512, 512)
lora = LoRALinear(original, rank=16, alpha=32)

original_params = sum(p.numel() for p in [original.weight, original.bias]
                      if p is not None)
lora_params = sum(p.numel() for p in [lora.lora_A, lora.lora_B])
print(f"  原始层参数: {original_params}")
print(f"  LoRA 参数: {lora_params}")
print(f"  参数比例: {lora_params / original_params * 100:.2f}%")

x = torch.randn(1, 10, 512)
y = lora(x)
print("  输入 shape:", x.shape)
print("  输出 shape:", y.shape)

# ============================================================
# 4. 使用 PEFT 库进行 LoRA 微调
# ============================================================
print("\n\n" + "=" * 60)
print("【4. 使用 PEFT 库进行 LoRA 微调】")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

# 加载模型
MODEL_PATH = "/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
print("\n加载 Qwen2.5-0.5B-Instruct 模型...")
print("模型路径:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": device},
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 查看可以添加 LoRA 的目标模块
print("\n--- 模型中的 Linear 层（候选 LoRA target）---")
linear_modules = set()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # 提取模块名（去掉层号前缀）
        parts = name.split('.')
        module_name = parts[-1]
        linear_modules.add(module_name)
print("可选 target modules:", sorted(linear_modules))

# 配置 LoRA
print("\n--- 配置 LoRA ---")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,     # 任务类型: 因果语言模型
    r=16,                              # LoRA rank
    lora_alpha=32,                     # 缩放因子 (alpha/r)
    lora_dropout=0.05,                 # Dropout 防过拟合
    target_modules=[                   # 在哪些层添加 LoRA
        "q_proj", "k_proj", "v_proj",  # Attention QKV
        "o_proj",                      # Attention 输出
        "gate_proj", "up_proj",        # FFN gate/up (SwiGLU)
        "down_proj",                   # FFN down
    ],
    bias="none",                       # 不训练 bias
)

print("LoRA 配置:")
print("  rank (r):", lora_config.r)
print("  alpha:", lora_config.lora_alpha)
print("  scaling:", lora_config.lora_alpha / lora_config.r)
print("  target_modules:", lora_config.target_modules)
print("  dropout:", lora_config.lora_dropout)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 查看参数量变化
print("\n--- LoRA 后参数统计 ---")
model.print_trainable_parameters()

# ============================================================
# 5. 准备训练数据
# ============================================================
print("\n\n" + "=" * 60)
print("【5. 准备训练数据】")
print("=" * 60)
print("""
SFT (Supervised Fine-Tuning) 数据格式：

1. Alpaca 格式:
   {"instruction": "...", "input": "...", "output": "..."}

2. ShareGPT 格式:
   {"conversations": [
     {"from": "human", "value": "..."},
     {"from": "gpt", "value": "..."}
   ]}

3. 对话格式:
   {"messages": [
     {"role": "system", "content": "..."},
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."}
   ]}

关键注意事项：
  - 只对 assistant 的回复计算 loss（不对 user 的输入计算）
  - 这通过 labels 中的 -100 实现（PyTorch 的 CrossEntropyLoss 会忽略 -100）
""")

# 创建一个小型演示数据集
training_data = [
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI技术助手。"},
            {"role": "user", "content": "什么是大语言模型？"},
            {"role": "assistant", "content": "大语言模型(LLM)是一种基于Transformer架构的深度学习模型，"
             "通过在大规模文本数据上进行预训练，学习语言的统计规律和知识。"
             "它具有强大的文本理解和生成能力，能够进行对话、翻译、摘要、编程等多种任务。"
             "代表模型包括GPT系列、LLaMA、Qwen等。"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI技术助手。"},
            {"role": "user", "content": "解释一下注意力机制"},
            {"role": "assistant", "content": "注意力机制(Attention)是Transformer的核心组件。"
             "它通过计算Query和Key的相似度来确定每个位置应该关注输入序列的哪些部分，"
             "再用这个权重对Value进行加权求和。"
             "公式为Attention(Q,K,V) = softmax(QK^T/√d_k)V。"
             "多头注意力让模型同时学习不同类型的依赖关系。"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI技术助手。"},
            {"role": "user", "content": "LoRA微调是什么？"},
            {"role": "assistant", "content": "LoRA(Low-Rank Adaptation)是一种高效的模型微调方法。"
             "核心思想是：预训练权重的更新矩阵是低秩的，可以分解为两个小矩阵的乘积ΔW=AB。"
             "训练时冻结原始权重W，只训练A和B，参数量可减少100倍以上。"
             "推理时可以合并W+AB，不引入额外延迟。"
             "是目前最流行的微调方法。"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI技术助手。"},
            {"role": "user", "content": "什么是KV-Cache？"},
            {"role": "assistant", "content": "KV-Cache是LLM推理加速的关键技术。"
             "在自回归生成过程中，每一步只生成一个新token，"
             "但需要重新计算所有之前token的Key和Value，这造成了大量重复计算。"
             "KV-Cache将之前计算过的K和V缓存起来，每步只需要计算新token的K和V并拼接到缓存中。"
             "这将每步的计算复杂度从O(n)降低到O(1)。"},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "你是一个专业的AI技术助手。"},
            {"role": "user", "content": "RoPE位置编码的优势是什么？"},
            {"role": "assistant", "content": "RoPE(旋转位置编码)通过旋转矩阵来编码位置信息，具有三个关键优势：\n"
             "1. 编码相对位置而非绝对位置，更符合语言的本质特征。\n"
             "2. 只应用在Q和K上，计算高效，V保持不变。\n"
             "3. 天然支持长度外推，结合NTK-aware scaling等技术可以推广到更长序列。\n"
             "目前Llama、Qwen等主流模型都采用RoPE。"},
        ]
    },
]


def prepare_training_examples(data, tokenizer, max_length=512):
    """将对话数据转换为训练格式"""
    examples = {"input_ids": [], "attention_mask": [], "labels": []}

    for item in data:
        messages = item["messages"]

        # 使用 chat template 格式化
        try:
            # 构建完整对话文本
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)

            # 构建只到 user 的文本（用于确定 loss mask 边界）
            user_messages = messages[:-1]
            user_text = tokenizer.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True)

            # 编码
            full_encoded = tokenizer(full_text, truncation=True,
                                      max_length=max_length, padding=False)
            user_encoded = tokenizer(user_text, truncation=True,
                                      max_length=max_length, padding=False)

            input_ids = full_encoded["input_ids"]
            attention_mask = full_encoded["attention_mask"]

            # 构建 labels：user 部分设为 -100（不计算 loss）
            labels = input_ids.copy()
            user_len = len(user_encoded["input_ids"])
            for j in range(min(user_len, len(labels))):
                labels[j] = -100

            examples["input_ids"].append(input_ids)
            examples["attention_mask"].append(attention_mask)
            examples["labels"].append(labels)

        except Exception as e:
            print("数据处理失败:", e)
            continue

    return examples


training_examples = prepare_training_examples(training_data, tokenizer)
print(f"准备了 {len(training_examples['input_ids'])} 条训练样本")

# 展示一条样本的结构
print("\n--- 样本示例 ---")
sample_ids = training_examples["input_ids"][0]
sample_labels = training_examples["labels"][0]
print("Input IDs 长度:", len(sample_ids))
print(f"Labels 中 -100 的数量（不计算 loss 的部分）: {sample_labels.count(-100)}")
print(f"Labels 中有效 token 数量: {len(sample_labels) - sample_labels.count(-100)}")

# 可视化 loss mask
tokens = tokenizer.convert_ids_to_tokens(sample_ids)
print("\nToken-Level Loss Mask 可视化 (前40个token):")
for i in range(min(40, len(tokens))):
    mask = "●" if sample_labels[i] != -100 else "○"
  print(f"  {mask} {tokens[i]}", end="")
    if (i + 1) % 5 == 0:
        print()
print("\n  ● = 计算 loss, ○ = 不计算 loss (user/system 部分)")

# ============================================================
# 6. 简单训练循环
# ============================================================
print("\n\n" + "=" * 60)
print("【6. 训练循环（简化版）】")
print("=" * 60)

from torch.optim import AdamW

# 准备数据
optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
model.train()

print("开始训练（3轮，演示用mini数据集）...\n")
num_epochs = 3
total_steps = 0

for epoch in range(num_epochs):
    epoch_loss = 0

    for i in range(len(training_examples["input_ids"])):
        input_ids = torch.tensor([training_examples["input_ids"][i]],
                                  device=device)
        attention_mask = torch.tensor([training_examples["attention_mask"][i]],
                                       device=device)
        labels = torch.tensor([training_examples["labels"][i]],
                               device=device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        total_steps += 1

    avg_loss = epoch_loss / len(training_examples["input_ids"])
  print(f"  Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

# ============================================================
# 7. 保存和加载 LoRA 权重
# ============================================================
print("\n\n" + "=" * 60)
print("【7. 保存和加载 LoRA 权重】")
print("=" * 60)

save_dir = "/workspace/huangzh14@xiaopeng.com/llm_learning/lora_demo_output"
os.makedirs(save_dir, exist_ok=True)

# 保存 LoRA 权重（只保存 LoRA 参数，非常小）
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 查看保存的文件
print("保存目录:", save_dir)
saved_files = os.listdir(save_dir)
for f in saved_files:
    size = os.path.getsize(os.path.join(save_dir, f))
    if size > 1024:
    print(f"  {f}  ({size / 1024:.1f} KB)")
    else:
    print(f"  {f}  ({size} bytes)")

print("""
注意：LoRA 权重通常只有几 MB，而完整模型可能有 GB 级别。
这是 LoRA 的另一个优势：存储和分发成本极低。

LoRA 权重包含：
  - adapter_config.json: LoRA 配置信息
  - adapter_model.safetensors: LoRA 参数（A 和 B 矩阵）
""")

# ============================================================
# 8. LoRA 合并与推理
# ============================================================
print("\n" + "=" * 60)
print("【8. LoRA 合并与推理】")
print("=" * 60)
print("""
LoRA 推理有两种方式：

方式1: 不合并，保持 adapter（灵活切换）
  model = AutoModelForCausalLM.from_pretrained(base_model)
  model = PeftModel.from_pretrained(model, lora_path)
  # 可以动态加载/卸载不同 LoRA

方式2: 合并权重（无额外推理延迟）
  model = model.merge_and_unload()
  # W_new = W + A×B，之后和普通模型一样
  # 不能再切换 LoRA

【面试考点】
Q: LoRA 合并后和全量微调的模型有什么区别？
A: 数学上完全等价（都是在权重上加了 ΔW）。
   但 LoRA 约束了 ΔW 是低秩的（rank=r），
   这意味着 LoRA 的表达能力受限于 r 的大小。
""")

# 用微调后的模型推理
model.eval()
prompt = "什么是大语言模型？"
messages = [
    {"role": "system", "content": "你是一个专业的AI技术助手。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=150,
                             do_sample=True, temperature=0.7, top_p=0.9)
response = tokenizer.decode(output[0][input_ids.shape[1]:],
                             skip_special_tokens=True)
print("\n微调后的回答:")
print(f"  问: {prompt}")
print(f"  答: {response[:300]}")

# ============================================================
# 9. 总结与进阶方向
# ============================================================
print("\n\n" + "=" * 60)
print("【9. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. LoRA 通过低秩分解 ΔW=AB 实现高效微调
  2. 只训练 A, B（冻结原始 W），参数量减少 100x+
  3. B 初始化为零确保训练起点等价于原模型
  4. 推理时可合并 W+AB，无额外延迟
  5. Loss Mask (-100) 确保只对 assistant 回复计算 loss

✅ 面试高频题：
  1. LoRA 的数学原理？W_new = W + AB
  2. rank r 怎么选？r=8~64，取决于任务复杂度
  3. alpha 和 r 的关系？scaling = alpha/r
  4. LoRA 可以应用在哪些层？QKV+FFN 都可以
  5. LoRA vs QLoRA？QLoRA = 4bit量化 + LoRA + 分页优化器
  6. SFT 数据中为什么要 loss mask？只让模型学习生成回复

✅ 进阶方向：
  - DoRA：分解权重为方向和大小，比 LoRA 效果更好
  - LoRA+：对 A 和 B 使用不同学习率
  - rsLoRA：根据 rank 调整 scaling
  - GaLore：全参数低秩训练

🎉 基础四课完成！你已经掌握了 LLM 的核心知识：
   Tokenizer → Attention → 推理 → 微调
""")

# 清理
del model
torch.cuda.empty_cache()
print("\n已释放模型显存。")
