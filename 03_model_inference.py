"""
========================================
 第03课：模型加载与推理基础
========================================
从加载预训练模型到理解生成过程，掌握 LLM 推理的完整链路。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/03_model_inference.py
"""
import torch
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(" 第03课：模型加载与推理基础")
print("=" * 60)

# ============================================================
# 1. 模型加载方式
# ============================================================
print("\n" + "=" * 60)
print("【1. 模型加载方式】")
print("=" * 60)
print("""
加载模型的几种方式：

1. AutoModelForCausalLM（最常用）
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

2. 指定精度加载（节省显存）
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2-0.5B",
       torch_dtype=torch.float16,  # 半精度
       device_map="auto",          # 自动分配到 GPU
   )

3. 量化加载（进一步节省显存）
   from transformers import BitsAndBytesConfig
   bnb_config = BitsAndBytesConfig(load_in_4bit=True)
   model = AutoModelForCausalLM.from_pretrained(
       "model_name", quantization_config=bnb_config)

精度与显存对照：
  ┌───────────┬─────────────┬───────────────────────┐
  │ 精度      │ 每参数字节  │ 7B 模型显存估算       │
  ├───────────┼─────────────┼───────────────────────┤
  │ FP32      │ 4 bytes     │ ~28 GB                │
  │ FP16/BF16 │ 2 bytes     │ ~14 GB                │
  │ INT8      │ 1 byte      │ ~7 GB                 │
  │ INT4      │ 0.5 bytes   │ ~3.5 GB               │
  └───────────┴─────────────┴───────────────────────┘

【面试考点】
Q: BF16 和 FP16 有什么区别？训练时用哪个？
A: BF16 指数位8位（和FP32相同），尾数7位；FP16 指数5位，尾数10位。
   BF16 数值范围更大（不容易溢出），但精度稍低。
   训练推荐 BF16（稳定），推理两者都行。
""")

# ============================================================
# 2. 加载并探索模型结构
# ============================================================
print("\n" + "=" * 60)
print("【2. 加载并探索模型结构】")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
print("正在加载 Qwen2.5-0.5B-Instruct（约1GB显存）...")
print("模型路径:", MODEL_PATH)
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map={"": device},
    trust_remote_code=True,
)
load_time = time.time() - start_time
print(f"加载完成！耗时: {load_time:.1f} 秒")

# 查看模型结构
print("\n--- 模型结构概览 ---")
print(model)

# 参数量统计
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n--- 参数量统计 ---")
print(f"总参数量: {total_params} ({total_params / 1e6:.2f} M)")
print(f"可训练参数: {trainable_params} ({trainable_params / 1e6:.2f} M)")

# 显存占用
mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
print("\n--- 显存占用 ---")
print(f"已分配: {mem_allocated:.2f} GB")
print(f"已预留: {mem_reserved:.2f} GB")

# 各层参数分布
print("\n--- 各组件参数量 ---")
param_groups = {}
for name, param in model.named_parameters():
    group = name.split('.')[1] if '.' in name else name
    if group not in param_groups:
        param_groups[group] = 0
    param_groups[group] += param.numel()

for group, count in sorted(param_groups.items(), key=lambda x: -x[1]):
    print(f"  {group}: {count / 1e6:.2f} M ({count / total_params * 100:.1f}%)")

# ============================================================
# 3. 理解自回归生成过程
# ============================================================
print("\n\n" + "=" * 60)
print("【3. 理解自回归生成过程】")
print("=" * 60)
print("""
自回归生成 (Autoregressive Generation):
  每一步预测下一个 token，将预测结果加入输入，循环往复。

过程示意：
  输入: "今天天气"
  Step 1: model("今天天气") -> "真"
  Step 2: model("今天天气真") -> "好"
  Step 3: model("今天天气真好") -> "！"
  Step 4: model("今天天气真好！") -> <eos>  → 停止

手动实现 vs model.generate():
  - 手动循环更灵活，理解原理
  - model.generate() 封装了各种采样策略
""")

# 手动实现自回归生成
print("\n--- 3.1 手动实现自回归生成 ---")
prompt = "人工智能的未来发展方向包括"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(f"输入文本: '{prompt}'")
print("输入 Token IDs:", input_ids[0].tolist())

generated_ids = input_ids.clone()
max_new_tokens = 30

print("\n逐步生成过程:")
model.eval()
with torch.no_grad():
    for step in range(max_new_tokens):
        # 前向传播，获取 logits
        outputs = model(generated_ids)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # 取最后一个位置的 logits（预测下一个 token）
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

        # Greedy: 取概率最高的 token
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 拼接到已生成序列
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        # 解码当前 token
        token_text = tokenizer.decode(next_token_id[0])

        # 检查是否生成了结束标记
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"  Step {step + 1:2d}: [EOS] -> 生成结束")
            break

        if step < 10:  # 只打印前10步
            # 获取 top-5 候选
            top5_values, top5_indices = torch.topk(
                torch.softmax(next_token_logits, dim=-1), 5)
            top5_tokens = [tokenizer.decode(idx) for idx in top5_indices[0]]
            top5_probs = top5_values[0].tolist()

            top5_text = ", ".join(
                f"'{token}'({prob:.2f})" for token, prob in zip(top5_tokens, top5_probs)
            )
            print(f"  Step {step + 1:2d}: '{token_text}' (top5: {top5_text})")

full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n完整生成结果:")
print("  " + full_text)

# ============================================================
# 4. 采样策略详解
# ============================================================
print("\n\n" + "=" * 60)
print("【4. 采样策略详解 - generate() 参数】")
print("=" * 60)
print("""
┌──────────────────┬───────────────────────────────────────┐
│ 参数             │ 说明                                  │
├──────────────────┼───────────────────────────────────────┤
│ do_sample=False  │ Greedy：总是选概率最高的（确定性）     │
│ do_sample=True   │ 开启随机采样                          │
├──────────────────┼───────────────────────────────────────┤
│ temperature      │ 温度：<1 更保守，>1 更随机             │
│                  │ logits / temperature → 控制分布锐利度  │
├──────────────────┼───────────────────────────────────────┤
│ top_k            │ 只从概率最高的 k 个 token 中采样       │
├──────────────────┼───────────────────────────────────────┤
│ top_p            │ 核采样：从累积概率达到 p 的最小集合采样 │
│ (nucleus)        │ 例如 top_p=0.9：选择直到概率之和≥0.9   │
├──────────────────┼───────────────────────────────────────┤
│ repetition_      │ 重复惩罚：>1 降低已出现 token 的概率   │
│ penalty          │ 避免模型反复说同一句话                 │
├──────────────────┼───────────────────────────────────────┤
│ num_beams        │ Beam Search：保留 n 个最优候选         │
│                  │ 更适合翻译等任务，生成对话不太用       │
└──────────────────┴───────────────────────────────────────┘

【面试考点】
Q: temperature 的数学原理？
A: softmax(logits / T)
   T→0：分布趋向 one-hot（贪心）
   T=1：原始分布
   T→∞：趋向均匀分布（完全随机）

Q: top_p 和 top_k 的区别？
A: top_k 固定候选数量（不管概率分布如何）
   top_p 动态调整候选数量（根据概率分布）
   top_p 更灵活：对于高确定性的位置自动收窄候选集
""")

# 对比不同采样策略
print("\n--- 对比不同采样策略 ---")
prompt = "大语言模型最重要的突破是"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

strategies = [
    {"name": "Greedy", "params": {"do_sample": False, "max_new_tokens": 50}},
    {"name": "Temperature=0.3 (保守)", "params": {
        "do_sample": True, "temperature": 0.3, "max_new_tokens": 50}},
    {"name": "Temperature=1.0 (标准)", "params": {
        "do_sample": True, "temperature": 1.0, "top_p": 0.9, "max_new_tokens": 50}},
    {"name": "Temperature=1.5 (创意)", "params": {
        "do_sample": True, "temperature": 1.5, "top_p": 0.95, "max_new_tokens": 50}},
]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

for strategy in strategies:
    output = model.generate(input_ids, **strategy["params"])
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n[{strategy['name']}]")
    print("  " + text[:120])

# ============================================================
# 5. KV-Cache 原理与演示
# ============================================================
print("\n\n" + "=" * 60)
print("【5. KV-Cache - 推理加速的关键】")
print("=" * 60)
print("""
问题：自回归生成时，每一步都要重新计算所有 token 的 K, V
  Step 1: 计算 K,V for [t1, t2, t3]           → 3 次计算
  Step 2: 计算 K,V for [t1, t2, t3, t4]       → 4 次计算 (t1,t2,t3 重复!)
  Step 3: 计算 K,V for [t1, t2, t3, t4, t5]   → 5 次计算 (t1-t4 重复!)

KV-Cache: 缓存之前的 K, V，只计算新 token 的 K, V
  Step 1: 计算 K,V for [t1, t2, t3]           → 3 次 → cache
  Step 2: 计算 K,V for [t4]                   → 1 次 → 和 cache 拼接
  Step 3: 计算 K,V for [t5]                   → 1 次 → 和 cache 拼接

复杂度: O(n²) → O(n) 每步

KV-Cache 显存计算:
  cache_size = 2 × layers × kv_heads × d_k × seq_len × batch × dtype_bytes
  例: 7B 模型, 32 layers, 8 kv_heads, d_k=128, seq=4096, batch=1, fp16:
  = 2 × 32 × 8 × 128 × 4096 × 1 × 2 bytes = 512 MB
""")

# 对比有无 KV-Cache 的速度
print("\n--- 速度对比: 有/无 KV-Cache ---")
prompt = "请列举人工智能的主要应用领域"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
gen_len = 100

# 使用 KV-Cache（默认行为）
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=gen_len, do_sample=False,
                            use_cache=True)
torch.cuda.synchronize()
time_with_cache = time.time() - start
text_cached = tokenizer.decode(output[0], skip_special_tokens=True)

# 不使用 KV-Cache
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=gen_len, do_sample=False,
                            use_cache=False)
torch.cuda.synchronize()
time_without_cache = time.time() - start

print(f"无 KV-Cache: {time_without_cache:.2f} 秒")
print(f"有 KV-Cache: {time_with_cache:.2f} 秒")
print(f"加速比: {time_without_cache / time_with_cache:.2f}x")
print(f"\n生成内容: {text_cached[:150]}")

# ============================================================
# 6. Prefill 和 Decode 两阶段
# ============================================================
print("\n\n" + "=" * 60)
print("【6. Prefill 与 Decode 两阶段】")
print("=" * 60)
print("""
LLM 推理分两个阶段：

1. Prefill（预填充阶段）:
   - 处理整个 prompt
   - 并行计算所有 token（矩阵乘法）
   - 计算密集型（Compute-bound）
   - 生成第一个 token 的延迟 = TTFT (Time To First Token)

2. Decode（解码阶段）:
   - 一次生成一个 token
   - 每步只计算一个 token（但要读取完整 KV-Cache）
   - 内存密集型（Memory-bound）
   - 吞吐量关键指标：tokens/second

性能指标：
  - TTFT: 首 token 延迟（用户体验关键）
  - TPS: 每秒生成 token 数
  - Throughput: 批次吞吐（服务端关键）

【面试考点】
Q: 为什么 Prefill 是计算密集，Decode 是内存密集？
A: Prefill 一次处理 N 个 token，矩阵乘法充分利用 GPU 算力。
   Decode 每次只处理 1 个 token，但需要读取所有 KV-Cache，
   GPU 算力利用率低，瓶颈在显存带宽。
""")

# 测量 Prefill 和 Decode 时间
prompt_long = "请详细介绍一下机器学习、深度学习、自然语言处理和计算机视觉这几个人工智能领域的发展历程和关键技术突破"
input_ids = tokenizer.encode(prompt_long, return_tensors="pt").to(device)
print(f"Prompt 长度: {input_ids.shape[1]} tokens")

# Prefill
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
torch.cuda.synchronize()
prefill_time = time.time() - start
print(f"Prefill 耗时: {prefill_time:.3f} 秒 (TTFT)")

# Decode
decode_steps = 50
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    for _ in range(decode_steps - 1):
        out = model(next_id, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
torch.cuda.synchronize()
decode_time = time.time() - start
print(f"Decode {decode_steps} tokens 耗时: {decode_time:.3f} 秒")
print(f"Decode 速度: {decode_steps / decode_time:.1f} tokens/sec")

# ============================================================
# 7. 对话推理完整示例
# ============================================================
print("\n\n" + "=" * 60)
print("【7. 对话推理完整示例】")
print("=" * 60)

messages_list = [
    [
        {"role": "system", "content": "你是一个专业的AI助手。"},
        {"role": "user", "content": "用一句话解释什么是Transformer。"},
    ],
    [
        {"role": "system", "content": "你是一个专业的AI助手。"},
        {"role": "user", "content": "LoRA微调的核心思想是什么？"},
    ],
]

for i, messages in enumerate(messages_list):
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:],
                                     skip_special_tokens=True)
        print(f"\n对话 {i + 1}:")
        print(f"  问: {messages[-1]['content']}")
        print(f"  答: {response[:200]}")
    except Exception as e:
        print(f"对话 {i + 1} 失败: {e}")

# ============================================================
# 8. 总结
# ============================================================
print("\n\n" + "=" * 60)
print("【8. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. 模型加载：精度选择(FP16/BF16/INT4)直接决定显存占用
  2. 自回归生成：每步预测下一个 token，循环往复
  3. 采样策略：temperature/top_p/top_k 控制生成多样性
  4. KV-Cache：缓存 K,V 避免重复计算，每步 O(1) 而非 O(n)
  5. Prefill vs Decode：两阶段特性不同，优化策略也不同

✅ 面试高频题：
  1. FP16 vs BF16 的区别？训练用哪个？
  2. temperature 的数学原理？
  3. top_p 和 top_k 哪个更好？为什么？
  4. KV-Cache 的显存怎么计算？
  5. 为什么 batch decode 比单条 decode 更高效？
  6. TTFT 和 TPS 分别受什么因素影响？

下一课：04_lora_finetune.py - LoRA 微调入门
""")

# 清理显存
del model
torch.cuda.empty_cache()
print("\n已释放模型显存。")
