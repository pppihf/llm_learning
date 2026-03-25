# LLM 基础知识学习路线

> 面向大模型/NLP 算法工程师面试准备，基于 fuyao 平台实战演练

## 环境信息
- **平台**: (PPU × 8, 每张 95.6GB)
- **可用 GPU**: GPU 0-7 (8张空闲)
- **PyTorch**: 2.7.0 | **Transformers**: 4.57.6 | **PEFT**: 0.16.0

## 学习大纲

### 第一阶段：基础概念与动手实践

| 编号 | 主题 | 文件 | 核心内容 |
|------|------|------|----------|
| 01 | Tokenizer 基础 | `01_tokenizer_basics.py` | BPE/WordPiece/SentencePiece 原理，分词实践，特殊 token |
| 02 | Transformer 与注意力机制 | `02_attention_mechanism.py` | Self-Attention 手动实现，Multi-Head Attention，位置编码 |
| 03 | 模型加载与推理 | `03_model_inference.py` | 模型结构解析，generate 参数详解，KV-Cache 原理 |
| 04 | LoRA 微调入门 | `04_lora_finetune.py` | LoRA 原理，PEFT 实践，数据准备与训练 |

### 第二阶段（后续扩展）

| 编号 | 主题 | 文件 | 核心内容 |
|------|------|------|----------|
| 05 | 分布式训练基础 | `05_distributed_training.py` | DP/TP/PP/ZeRO/FSDP 的区别，world size 与并行度拆分 |
| 06 | 对齐训练与 DPO | `06_alignment_and_dpo.py` | SFT / RLHF / DPO 关系，偏好数据格式，DPO loss 直觉 |
| 07 | 推理优化与服务化 | `07_inference_optimization.py` | TTFT/TPS、连续批处理、Prefix Cache、Paged Attention |
| 08 | RAG 系统基础 | `08_rag_basics.py` | 切块、检索、Prompt 组装、RAG 常见坑 |
| 09 | LLM Agent 基础 | `09_agent_basics.py` | ReAct、Function Calling、Multi-Agent、MCP |


## 面试高频考点速查
- Transformer 为什么用 scaled dot-product attention？
- BPE vs WordPiece vs Unigram 的区别？
- LoRA 为什么有效？秩 r 怎么选？
- KV-Cache 怎么加速推理？
- FlashAttention 的核心思想？
- RoPE 位置编码的优势？
- ZeRO-3 和 FSDP 有什么异同？
- DPO 为什么不需要 PPO？
- 在线推理为什么要 continuous batching？
- RAG 的 chunk size 应该怎么选？
- Agent 和 RAG 的关系？
- ReAct 范式的流程是什么？
- Multi-Agent 有哪些常见架构？
