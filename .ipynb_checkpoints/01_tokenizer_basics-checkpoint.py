"""
========================================
 第01课：Tokenizer 基础与实践
========================================
大模型不直接处理文本，需要先将文本转换为数字序列（token ids）。
Tokenizer 是连接人类语言和模型的桥梁，理解它是学习 LLM 的第一步。

运行方式:
  fuyao shell --job-name=bifrost-2026031209470900-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/01_tokenizer_basics.py
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("=" * 60)
print(" 第01课：Tokenizer 基础与实践")
print("=" * 60)

# ============================================================
# 1. 什么是 Tokenization？
# ============================================================
print("\n" + "=" * 60)
print("【1. 什么是 Tokenization？】")
print("=" * 60)
print("""
文本 -> Tokenizer -> Token IDs -> 模型 -> Token IDs -> Tokenizer -> 文本

三种主流分词算法：
┌─────────────┬────────────────────────────────────────────┐
│ BPE         │ 从字符开始，反复合并最高频的相邻对          │
│ (GPT系列)   │ 优点：无 OOV，压缩率好                     │
├─────────────┼────────────────────────────────────────────┤
│ WordPiece   │ 类似BPE，但用似然度而非频率决定合并         │
│ (BERT)      │ 使用 ## 前缀标记子词                       │
├─────────────┼────────────────────────────────────────────┤
│ Unigram     │ 从大词表开始，逐步删减低概率的子词          │
│ (T5/Llama)  │ 基于概率模型，通常配合 SentencePiece 使用  │
└─────────────┴────────────────────────────────────────────┘

【面试考点】
Q: BPE 和 WordPiece 的核心区别？
A: BPE 按 频率 合并 pair；WordPiece 按 互信息（似然增益）合并。
   BPE: score(A,B) = count(AB)
   WordPiece: score(A,B) = count(AB) / (count(A) * count(B))
""")

# ============================================================
# 2. 动手实现：最简版 BPE
# ============================================================
print("\n" + "=" * 60)
print("【2. 动手实现 BPE 分词（简化版）】")
print("=" * 60)


def simple_bpe(text, num_merges=10):
    """最简版 BPE 实现，帮助理解核心原理"""
    # Step 1: 将文本拆成字符序列，每个词末尾加 </w> 标记
    words = text.split()
    vocab = {}
    for word in words:
        chars = list(word) + ["</w>"]
        key = " ".join(chars)
        vocab[key] = vocab.get(key, 0) + 1

    print("初始词表（字符级）:")
    for word, count in vocab.items():
        print("  '%s' × %d" % (word, count))

    merge_rules = []

    for i in range(num_merges):
        # Step 2: 统计所有相邻 pair 的频率
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j + 1])
                pairs[pair] = pairs.get(pair, 0) + freq

        if not pairs:
            break

        # Step 3: 找到频率最高的 pair
        best_pair = max(pairs, key=pairs.get)
        merge_rules.append(best_pair)
        print("\n第 %d 次合并: '%s' + '%s' -> '%s' (频率: %d)" % (
            i + 1, best_pair[0], best_pair[1],
            best_pair[0] + best_pair[1], pairs[best_pair]))

        # Step 4: 在词表中执行合并
        new_vocab = {}
        merged = best_pair[0] + " " + best_pair[1]
        replacement = best_pair[0] + best_pair[1]
        for word in vocab:
            new_word = word.replace(merged, replacement)
            new_vocab[new_word] = vocab[word]
        vocab = new_vocab

    print("\n最终词表:")
    for word, count in vocab.items():
        print("  '%s' × %d" % (word, count))

    return merge_rules


corpus = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider"
print("输入文本:", corpus)
print()
rules = simple_bpe(corpus, num_merges=10)
print("\n合并规则:", rules)

# ============================================================
# 3. 使用 HuggingFace Tokenizer
# ============================================================
print("\n\n" + "=" * 60)
print("【3. 使用 HuggingFace Tokenizer 实践】")
print("=" * 60)

from transformers import AutoTokenizer

# 使用本地 Qwen2.5 Tokenizer（公共路径，无需下载）
MODEL_PATH = "/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
print("\n--- 加载 Qwen2.5 Tokenizer ---")
print("模型路径:", MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model_name = "Qwen2.5-0.5B-Instruct"

print("使用 Tokenizer:", model_name)
print("词表大小:", tokenizer.vocab_size)
print("特殊 token:", tokenizer.all_special_tokens)

# 3.1 基础编码/解码
print("\n--- 3.1 基础编码与解码 ---")
texts = [
    "Hello, world!",
    "你好，世界！",
    "大模型是人工智能的重要方向",
    "The transformer architecture uses self-attention mechanism.",
    "GPT-4 和 LLaMA-3 是当前主流的大语言模型",
]

for text in texts:
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    decoded = tokenizer.decode(token_ids)
    print("\n原文: %s" % text)
    print("Token IDs (%d个): %s" % (len(token_ids), token_ids))
    print("Tokens: %s" % tokens)
    print("解码回文本: %s" % decoded)

# 3.2 特殊 Token
print("\n\n--- 3.2 特殊 Token 详解 ---")
print("""
特殊 Token 的作用：
┌──────────┬────────────────────────────────────────┐
│ [CLS]    │ BERT 分类任务的起始标记                 │
│ [SEP]    │ BERT 句子分隔符                        │
│ [PAD]    │ 填充到相同长度                         │
│ [UNK]    │ 未知 token（好的 tokenizer 应该没有）   │
│ <|endoftext|> │ GPT 系列文本结束标记              │
│ <|im_start|>  │ Qwen 对话格式开始                 │
│ <|im_end|>    │ Qwen 对话格式结束                 │
│ <s> / </s>    │ Llama 系列的 BOS/EOS              │
└──────────┴────────────────────────────────────────┘
""")

print("当前 Tokenizer 的特殊 token:")
special_tokens = {
    "bos_token": tokenizer.bos_token,
    "eos_token": tokenizer.eos_token,
    "pad_token": tokenizer.pad_token,
    "unk_token": tokenizer.unk_token,
}
for name, token in special_tokens.items():
    if token:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print("  %s: '%s' (id=%d)" % (name, token, token_id))

# 3.3 Batch 编码与 Padding
print("\n\n--- 3.3 Batch 编码与 Padding ---")
batch_texts = [
    "短文本",
    "这是一段稍微长一些的文本内容",
    "大语言模型的训练需要大量的计算资源和高质量的数据集",
]

# 设置 pad_token（有些模型没有默认的 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

encoded = tokenizer(
    batch_texts,
    padding=True,         # 填充到最长
    truncation=True,      # 超过最大长度时截断
    max_length=64,        # 最大长度
    return_tensors="pt",  # 返回 PyTorch tensor
)

print("Batch 编码结果:")
print("  input_ids shape:", encoded["input_ids"].shape)
print("  attention_mask shape:", encoded["attention_mask"].shape)
for i, text in enumerate(batch_texts):
    print("\n  文本 %d: '%s'" % (i, text))
    print("  input_ids:", encoded["input_ids"][i].tolist())
    print("  attention_mask:", encoded["attention_mask"][i].tolist())
    print("  (1=真实token, 0=padding)")

# 3.4 中英文分词对比
print("\n\n--- 3.4 中英文分词效率对比 ---")
print("""
【面试考点】
Q: 为什么中文在英文模型中 token 数量更多？
A: 英文 tokenizer 训练语料以英文为主，中文字符多被拆成字节级别的 token。
   这导致：1) 中文推理更慢（序列更长）；2) 中文能力可能更弱。
   Qwen/GLM 等中文模型专门扩展了中文词表来缓解这个问题。
""")

test_text_en = "Large language models are transforming the AI industry."
test_text_zh = "大语言模型正在变革人工智能行业。"

tokens_en = tokenizer.encode(test_text_en)
tokens_zh = tokenizer.encode(test_text_zh)
print("英文 (%d 字符) -> %d tokens" % (len(test_text_en), len(tokens_en)))
print("中文 (%d 字符) -> %d tokens" % (len(test_text_zh), len(tokens_zh)))
print("中/英 token 比: %.2f" % (len(tokens_zh) / len(tokens_en)))

# ============================================================
# 4. Chat Template（对话模板）
# ============================================================
print("\n\n" + "=" * 60)
print("【4. Chat Template（对话模板）】")
print("=" * 60)
print("""
不同模型有不同的对话格式（Chat Template），这是面试常考点：

Qwen (ChatML 格式):
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  你好<|im_end|>
  <|im_start|>assistant
  你好！有什么可以帮你的吗？<|im_end|>

Llama 3 格式:
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are a helpful assistant.<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  你好<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>

【面试考点】
Q: 为什么对话模板很重要？
A: 训练时用什么格式，推理时就必须用相同格式，否则模型效果会严重下降。
   模板定义了模型区分 system/user/assistant 角色的方式。
""")

messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "什么是 Transformer？"},
]

try:
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Chat Template 格式化结果:")
    print(formatted)
except Exception as e:
    print("当前 tokenizer 不支持 chat template:", e)

# ============================================================
# 5. 总结与面试要点
# ============================================================
print("\n\n" + "=" * 60)
print("【5. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. Tokenizer 将文本转为 token ids，是模型处理文本的第一步
  2. BPE 是最常用的分词算法（GPT/Llama 系列都用它的变体）
  3. 词表大小是个 trade-off：太大→embedding 参数多，太小→序列太长
  4. 特殊 token 控制模型行为（BOS/EOS/PAD）
  5. Chat Template 定义对话格式，训练和推理必须一致

✅ 面试高频题：
  1. BPE vs WordPiece vs Unigram 的区别和各自特点
  2. 为什么需要 subword 分词？（解决 OOV + 保持合理序列长度）
  3. 词表大小怎么选？对模型有什么影响？
  4. attention_mask 的作用？（避免 padding token 参与计算）
  5. 不同模型的 Chat Template 有什么区别？

下一课：02_attention_mechanism.py - Transformer 与注意力机制
""")
