"""
========================================
 第08课：RAG 系统基础
========================================
用一个最小例子理解 RAG 的完整链路：切块、检索、拼 Prompt、再生成。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/08_rag_basics.py
"""

from collections import Counter

print("=" * 60)
print(" 第08课：RAG 系统基础")
print("=" * 60)

print("\n" + "=" * 60)
print("【1. RAG 在解决什么问题？】")
print("=" * 60)
print(
    """
RAG = Retrieval-Augmented Generation

它主要解决两类问题：
  1. 模型参数里没有最新知识
  2. 模型有知识，但回答时容易幻觉

典型链路：
  文档入库 -> 切块 -> 建索引 -> 用户提问 -> 检索相关块 -> 拼 Prompt -> 生成回答
"""
)


def tokenize(text):
    return [token for token in text.lower().replace("，", " ").replace("。", " ").split() if token]


def overlap_score(query, chunk):
    query_counter = Counter(tokenize(query))
    chunk_counter = Counter(tokenize(chunk))
    return sum((query_counter & chunk_counter).values())


documents = [
    "Transformer 通过自注意力机制建模序列中任意两个位置之间的关系。",
    "LoRA 通过低秩矩阵近似权重更新，从而减少可训练参数。",
    "RAG 通常由切块、向量检索、重排和生成四个主要部分组成。",
    "KV-Cache 通过缓存历史 token 的 key/value 来加速自回归解码。",
]

query = "RAG 系统一般包含哪些部分？"
scored_chunks = sorted(
    ((chunk, overlap_score(query, chunk)) for chunk in documents),
    key=lambda item: item[1],
    reverse=True,
)

print("\n" + "=" * 60)
print("【2. 一个最小检索示例】")
print("=" * 60)
print(f"查询: {query}")
for chunk, score in scored_chunks:
    print(f"- score={score}: {chunk}")

top_chunks = [chunk for chunk, score in scored_chunks[:2] if score > 0]

print("\n" + "=" * 60)
print("【3. Prompt 该怎么拼？】")
print("=" * 60)

context_block = "\n".join(f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(top_chunks))
prompt = f"""你是一个检索增强问答助手。请基于给定资料回答问题。

资料:
{context_block}

问题:
{query}

回答要求:
1. 只基于资料回答
2. 如果资料不足，明确说明
"""

print(prompt)

print("\n" + "=" * 60)
print("【4. 做 RAG 时最常见的坑】")
print("=" * 60)

pitfalls = [
    "切块太大：召回准但冗余高，prefill 变贵",
    "切块太小：召回碎片化，上下文不完整",
    "只做召回不做重排：Top-K 结果常常不够稳",
    "Prompt 里不约束回答边界：模型容易把检索和自有知识混在一起",
]

for item in pitfalls:
    print(f"- {item}")

print("\n" + "=" * 60)
print("【5. 常见面试问法】")
print("=" * 60)
print(
    """
1. RAG 为什么不等于“把文档贴进 prompt”？
   因为真正难点在召回质量、上下文组织、引用约束和在线成本控制。

2. Chunk size 怎么选？
   看任务粒度、文档结构、embedding 模型窗口和下游 prompt 成本，没有固定万能值。

3. 为什么要 rerank？
   向量召回只负责“粗筛”，rerank 负责把真正最相关的内容排到前面。

4. RAG 能完全消除幻觉吗？
   不能，但能显著降低；前提是检索质量和回答约束都做好。
"""
)