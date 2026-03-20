"""
========================================
 第08课：RAG 检索增强生成
========================================
从零实现一个完整的 RAG 管线：文档切分 → 向量化 → 检索 → 生成。
理解 RAG 的核心原理、常见坑和优化方向。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/08_rag_basics.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print(" 第08课：RAG 检索增强生成")
print("=" * 60)

# ============================================================
# 1. 为什么需要 RAG？
# ============================================================
print("\n" + "=" * 60)
print("【1. 为什么需要 RAG？】")
print("=" * 60)
print("""
大模型的知识有两个固有缺陷：
  1. 知识截止日期：训练数据之后发生的事情不知道
  2. 幻觉（Hallucination）：自信地编造事实

RAG (Retrieval-Augmented Generation) 的思路：
  不要光靠模型脑子里的知识，先去数据库/文档里搜索相关信息，
  把搜到的内容塞到 prompt 里，让模型基于检索结果来回答。

  ┌─────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐
  │ 用户提问 │───→│ 检索模块 │───→│ 构建 Prompt │───→│ LLM 生成 │
  └─────────┘    │  搜索知   │    │  问题 +     │    │  基于检索 │
                 │  识库     │    │  检索结果   │    │  结果回答 │
                 └──────────┘    └────────────┘    └──────────┘

RAG vs 微调：
  ┌──────────┬─────────────────────┬─────────────────────┐
  │          │ RAG                 │ 微调 (Fine-tune)    │
  ├──────────┼─────────────────────┼─────────────────────┤
  │ 知识更新 │ 更新文档即可         │ 需要重新训练        │
  │ 事实准确 │ ✅ 有据可查          │ ❌ 可能幻觉         │
  │ 推理成本 │ 检索 + 长 prompt    │ 只需模型本身         │
  │ 适用场景 │ 知识密集型问答       │ 风格/能力迁移       │
  │ 可解释性 │ ✅ 可追溯来源        │ ❌ 黑箱             │
  └──────────┴─────────────────────┴─────────────────────┘
""")

# ============================================================
# 2. RAG 的完整管线
# ============================================================
print("\n" + "=" * 60)
print("【2. RAG 完整管线】")
print("=" * 60)
print("""
离线阶段 (Indexing):
  原始文档 → 切分 (Chunking) → 向量化 (Embedding) → 存入向量库

在线阶段 (Querying):
  用户提问 → 向量化 → 在向量库中检索 Top-K → 构建 Prompt → LLM 生成

  ┌── 离线 ──────────────────────────────────────────────┐
  │                                                      │
  │  [文档1]   [文档2]   [文档3]                         │
  │     ↓         ↓         ↓                            │
  │  [chunk1] [chunk2] [chunk3] [chunk4] ...             │
  │     ↓         ↓         ↓       ↓                   │
  │  [emb1]   [emb2]   [emb3]  [emb4]   → 向量数据库   │
  └──────────────────────────────────────────────────────┘

  ┌── 在线 ──────────────────────────────────────────────┐
  │                                                      │
  │  用户: "什么是 KV-Cache？"                           │
  │     ↓                                                │
  │  query_embedding = embed("什么是 KV-Cache？")        │
  │     ↓                                                │
  │  top_k = vector_db.search(query_embedding, k=3)      │
  │     ↓                                                │
  │  prompt = "基于以下信息回答问题：\\n" + top_k_text    │
  │     ↓                                                │
  │  answer = llm.generate(prompt)                        │
  └──────────────────────────────────────────────────────┘
""")

# ============================================================
# 3. 文档切分 (Chunking)
# ============================================================
print("\n" + "=" * 60)
print("【3. 文档切分 (Chunking) — 最容易被忽视但最重要的环节】")
print("=" * 60)

# 准备知识库文档
knowledge_base = """
Transformer 是 2017 年由 Google 提出的神经网络架构，论文标题为 "Attention Is All You Need"。
它的核心创新是自注意力机制（Self-Attention），允许序列中的每个位置直接关注其他所有位置。
Transformer 完全抛弃了 RNN 和 CNN，仅使用注意力机制，因此可以高度并行化训练。

KV-Cache 是大模型推理中的关键优化技术。在自回归生成时，每个新 token 需要和前面所有 token 做注意力计算。
如果不缓存，每生成一个 token 都要重新计算前面所有 token 的 Key 和 Value。
KV-Cache 把已计算过的 Key 和 Value 矩阵存下来，每步只计算新 token 的 Key 和 Value，
将生成的计算复杂度从 O(N²) 降到 O(N)。

LoRA（Low-Rank Adaptation）是一种参数高效微调方法。它冻结预训练模型的原始参数，
在每个目标层旁边添加两个小矩阵 A 和 B，使得 ΔW = B × A。
由于 A 和 B 的秩 r 远小于原始维度，可训练参数量大幅减少（通常不到 1%）。
LoRA 的优势是训练高效、可即插即用、支持多任务切换。

DPO（Direct Preference Optimization）是一种偏好对齐方法。
传统 RLHF 需要先训练 Reward Model 再用 PPO 优化，流程复杂且不稳定。
DPO 证明了可以直接在偏好数据上优化策略模型，把两步合成一步。
DPO 的 loss 函数本质上是鼓励模型增大 chosen 回答的概率，减小 rejected 回答的概率。

RAG（Retrieval-Augmented Generation）是检索增强生成。它的核心思想是
在 LLM 生成回答之前，先从外部知识库中检索相关信息，然后将检索结果作为上下文
输入给 LLM。这样可以减少幻觉，提供最新的知识，并且支持知识溯源。
"""


def chunk_by_fixed_size(text, chunk_size=150, overlap=30):
    """固定窗口切分，带重叠"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def chunk_by_paragraph(text):
    """按段落切分 — 更语义化"""
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return paragraphs


# 对比两种切分
fixed_chunks = chunk_by_fixed_size(knowledge_base, chunk_size=40, overlap=10)
para_chunks = chunk_by_paragraph(knowledge_base)

print("方法1: 固定窗口切分 (size=40 words, overlap=10)")
for i, chunk in enumerate(fixed_chunks):
    print(f"  Chunk {i}: [{len(chunk)}字] {chunk[:60]}...")

print(f"\n方法2: 按段落切分")
for i, chunk in enumerate(para_chunks):
    print(f"  Chunk {i}: [{len(chunk)}字] {chunk[:60]}...")

print("""
切分策略对比：
  ┌──────────────┬──────────────────┬──────────────────┐
  │ 方法         │ 优点             │ 缺点             │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 固定窗口     │ 简单，长度均匀   │ 可能切断语义     │
  │ 段落/章节    │ 保持语义完整     │ 长度不均匀       │
  │ 语义切分     │ 质量最好         │ 需要额外模型     │
  │ 递归切分     │ 多级 fallback    │ 实现复杂         │
  └──────────────┴──────────────────┴──────────────────┘

实践建议：
  - 通常 chunk_size 在 200~500 tokens 左右
  - overlap 设 10~20%，避免关键信息被切断
  - 如果文档有明确结构（标题、段落），优先用结构切分
""")

# ============================================================
# 4. 向量化 (Embedding) — 用模型自身的隐藏层做嵌入
# ============================================================
print("\n" + "=" * 60)
print("【4. 向量化 (Embedding)】")
print("=" * 60)

model_path = "/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"模型加载完成: {model_path}")

print("""
说明：专业 RAG 通常使用专门的 Embedding 模型（如 bge, e5, gte），
  它们经过对比学习训练，向量质量更高。
  这里为了教学简单，用 LLM 的隐藏层做 Embedding（取最后一层的 mean pooling）。
""")


def get_embedding(text, model, tokenizer):
    """用模型的 hidden states 做 embedding（mean pooling）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # 取最后一层 hidden state 的 mean
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    embedding = last_hidden.mean(dim=1).squeeze()  # (hidden_dim,)
    return F.normalize(embedding, dim=0)  # L2 归一化


# 对知识库的每个段落做向量化
chunks = chunk_by_paragraph(knowledge_base)
print("正在为知识库做向量化...")
chunk_embeddings = []
for i, chunk in enumerate(chunks):
    emb = get_embedding(chunk, model, tokenizer)
    chunk_embeddings.append(emb)
    print(f"  Chunk {i}: dim={emb.shape[0]}, norm={emb.norm().item():.4f}")

chunk_embeddings = torch.stack(chunk_embeddings)  # (num_chunks, hidden_dim)
print(f"\n向量库大小: {chunk_embeddings.shape}")

# ============================================================
# 5. 检索 (Retrieval)
# ============================================================
print("\n\n" + "=" * 60)
print("【5. 向量检索 (Retrieval)】")
print("=" * 60)


def retrieve(query, chunk_embeddings, chunks, model, tokenizer, top_k=3):
    """用余弦相似度检索最相关的 chunks"""
    query_emb = get_embedding(query, model, tokenizer)
    # 余弦相似度（embedding 已经 L2 归一化，点积 = 余弦相似度）
    similarities = torch.matmul(chunk_embeddings, query_emb)
    top_indices = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": idx.item(),
            "score": similarities[idx].item(),
            "text": chunks[idx.item()],
        })
    return results


test_queries = [
    "什么是 KV-Cache？",
    "LoRA 的原理是什么？",
    "DPO 和 RLHF 有什么区别？",
]

for query in test_queries:
    results = retrieve(query, chunk_embeddings, chunks, model, tokenizer, top_k=2)
    print(f"\n查询: {query}")
    for r in results:
        print(f"  [score={r['score']:.4f}] Chunk {r['chunk_id']}: {r['text'][:60]}...")

# ============================================================
# 6. 完整 RAG 管线
# ============================================================
print("\n\n" + "=" * 60)
print("【6. 完整 RAG 管线 — 检索 + 生成】")
print("=" * 60)

RAG_PROMPT_TEMPLATE = """基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请如实说明。

参考资料：
{context}

用户问题：{question}

回答："""


def rag_answer(question, chunk_embeddings, chunks, model, tokenizer, top_k=2):
    """完整的 RAG 问答"""
    # 1. 检索
    results = retrieve(question, chunk_embeddings, chunks, model, tokenizer, top_k)
    context = "\n\n".join([r["text"] for r in results])

    # 2. 构建 prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    # 3. 生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=150, do_sample=False,
        )
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return answer, results


print("--- RAG 问答演示 ---\n")

rag_queries = [
    "KV-Cache 把复杂度从多少降到了多少？",
    "LoRA 微调大约需要训练多少比例的参数？",
]

for q in rag_queries:
    answer, refs = rag_answer(q, chunk_embeddings, chunks, model, tokenizer)
    print(f"问题: {q}")
    print(f"检索到的 chunks: {[r['chunk_id'] for r in refs]}")
    print(f"RAG 回答: {answer[:200]}")
    print()

# ============================================================
# 7. RAG 的常见问题与优化
# ============================================================
print("\n" + "=" * 60)
print("【7. RAG 常见问题与优化】")
print("=" * 60)
print("""
常见问题 及 解决方案：

❌ 问题1: 检索不到相关内容
  原因: Query 和文档的表述方式不同
  方案: Query Rewriting（用 LLM 重写查询）
        HyDE（用 LLM 先生成假设性回答再用它检索）

❌ 问题2: 检索到了但答案在 chunk 边界被切断
  原因: 切分粒度不合适
  方案: 增加 overlap / 使用语义切分 / 父子 chunk 策略

❌ 问题3: 检索到的信息太多，LLM 迷失在长上下文中
  原因: Top-K 太大 或 chunk 太长
  方案: Reranking（用交叉编码器重排序）
        减小 chunk_size / 降低 Top-K

❌ 问题4: LLM 无视检索结果，用自己的知识回答
  原因: Prompt 设计不当
  方案: 强调"仅基于提供的信息回答"
        Few-shot 示例引导

❌ 问题5: 重复或矛盾的检索结果
  原因: 知识库本身有重复/过时内容
  方案: 文档去重 / 时间戳过滤 / 来源质量打分
""")

# ============================================================
# 8. 进阶 RAG 架构
# ============================================================
print("\n" + "=" * 60)
print("【8. 进阶 RAG 架构】")
print("=" * 60)
print("""
基础 RAG → 进阶 RAG 的演进：

  1. Naive RAG（本课实现的）
     Query → 检索 → 生成
     简单直接，但检索质量依赖 embedding 和切分

  2. Advanced RAG
     Query Rewriting → 多路检索 → Reranking → 生成
     ┌────────────────────────────────────────────┐
     │ "KV-Cache 是什么？"                        │
     │        ↓ Query Rewriting                   │
     │ "解释 KV-Cache 原理和作用"                 │
     │        ↓ 多路检索                          │
     │ 向量检索: [chunk2, chunk5, chunk7]          │
     │ BM25:    [chunk2, chunk3, chunk8]           │
     │        ↓ Reranking (Cross-Encoder)          │
     │ 合并去重 → 重排序 → 取 Top-K              │
     │        ↓ 生成                              │
     │ LLM 基于高质量上下文回答                    │
     └────────────────────────────────────────────┘

  3. Agentic RAG
     LLM 自主决定是否需要检索、检索什么、检索几次
     适合复杂多步推理场景（如多跳问答）

  4. GraphRAG
     把文档关系建成知识图谱
     检索时利用实体关系做推理
     适合强结构化知识场景
""")

# ============================================================
# 9. 向量数据库选型
# ============================================================
print("\n" + "=" * 60)
print("【9. 向量数据库选型】")
print("=" * 60)
print("""
┌──────────────┬──────────────────────────┬──────────────────┐
│ 数据库       │ 特点                     │ 适用场景         │
├──────────────┼──────────────────────────┼──────────────────┤
│ FAISS        │ Facebook 开源            │ 研究/原型        │
│              │ 纯库，无服务端           │ 百万级数据       │
├──────────────┼──────────────────────────┼──────────────────┤
│ Milvus       │ 全功能向量数据库         │ 生产环境         │
│              │ 支持分布式、云原生       │ 亿级数据         │
├──────────────┼──────────────────────────┼──────────────────┤
│ Chroma       │ 轻量级，Python 友好     │ 快速原型         │
│              │ 内嵌或 C/S 模式         │ 小规模项目       │
├──────────────┼──────────────────────────┼──────────────────┤
│ Pinecone     │ 全托管 SaaS              │ 不想运维         │
│              │ 开箱即用                 │ 中小规模         │
├──────────────┼──────────────────────────┼──────────────────┤
│ Qdrant       │ Rust 实现，高性能        │ 高性能需求       │
│              │ 支持过滤条件             │ 中大规模         │
└──────────────┴──────────────────────────┴──────────────────┘

本课用的是最简单的方式：torch.matmul 暴力检索。
数据量大时（>10K），必须用 ANN（近似最近邻）算法：
  - IVF（倒排索引）
  - HNSW（图索引）
  - PQ（量化压缩）
""")

# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 60)
print("【10. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. RAG = 检索 + 生成，用外部知识增强 LLM
  2. 离线建库: 文档切分 → 向量化 → 入库
  3. 在线查询: 问题向量化 → 检索 Top-K → 构建 Prompt → 生成
  4. 切分质量直接决定 RAG 效果
  5. 向量检索靠 embedding 的语义相似度

✅ 面试高频题：
  1. RAG 和微调怎么选？（知识密集型用 RAG，能力/风格用微调）
  2. chunk_size 怎么选？（200-500 tokens，太大稀释信噪比，太小丢上下文）
  3. 检索不准怎么办？（Query Rewriting, HyDE, 多路检索, Reranking）
  4. 向量检索用什么算法？（小规模暴力搜索，大规模用 IVF/HNSW/PQ）
  5. 怎么评估 RAG 效果？（检索: Recall@K, MRR; 生成: 忠实度, 相关性）
  6. RAG 和长上下文是什么关系？（长上下文让塞更多检索结果，但检索质量仍然关键）

恭喜完成全部 8 课！🎓
  第1阶段（基础）: 01 Tokenizer → 02 Attention → 03 推理 → 04 LoRA
  第2阶段（进阶）: 05 分布式 → 06 DPO → 07 推理优化 → 08 RAG
""")
