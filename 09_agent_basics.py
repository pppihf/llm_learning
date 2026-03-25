"""
========================================
 第09课：LLM Agent 基础
========================================
理解 Agent = LLM + 工具调用 + 规划 + 记忆。
从零手写一个 ReAct Agent 循环，理解 Function Calling、
多步推理、Multi-Agent 等核心概念。

运行方式:
  fuyao shell --job-name=bifrost-2026031617060701-huangzh14
  CUDA_VISIBLE_DEVICES=0 python /workspace/huangzh14@xiaopeng.com/llm_learning/09_agent_basics.py
"""
import torch
import json
import re
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print(" 第09课：LLM Agent 基础")
print("=" * 60)

# ============================================================
# 1. 什么是 Agent？
# ============================================================
print("\n" + "=" * 60)
print("【1. 什么是 Agent？】")
print("=" * 60)
print("""
一句话：Agent = LLM + 工具 + 规划 + 记忆

普通 LLM 聊天：
  用户提问 → LLM 生成回答 → 结束
  LLM 只能靠自己脑子里的知识，不能执行任何动作

Agent：
  用户提问 → LLM 思考需要做什么 → 调用工具获取信息 →
  根据结果继续思考 → 可能再调用工具 → ... → 最终回答

  ┌─────────────────────────────────────────────────┐
  │                    Agent                        │
  │                                                 │
  │  ┌──────────┐   ┌──────────┐   ┌────────────┐  │
  │  │ 规划     │──→│ LLM 大脑 │──→│ 工具调用   │  │
  │  │ Planning │   │ Reasoning│   │ Tool Use   │  │
  │  └──────────┘   └──────────┘   └────────────┘  │
  │        ↑              │              │          │
  │        │         ┌────┴────┐         │          │
  │        └─────────│  记忆   │←────────┘          │
  │                  │ Memory  │                    │
  │                  └─────────┘                    │
  └─────────────────────────────────────────────────┘

Agent 的四大组件：
  1. LLM（大脑）: 负责理解任务、推理、做决策
  2. 工具（手脚）: 搜索、计算、代码执行、API 调用...
  3. 规划（策略）: 把复杂任务拆解为多个步骤
  4. 记忆（经验）: 短期（对话历史）+ 长期（知识库/向量库）

【面试考点】
Q: Agent 和 RAG 的关系？
A: RAG 可以看作最简单的 Agent —— 只有一个"检索"工具。
   Agent 更通用：可以有多种工具，能多步推理，能自主决策。
""")

# ============================================================
# 2. ReAct 范式
# ============================================================
print("\n" + "=" * 60)
print("【2. ReAct 范式：Reasoning + Acting】")
print("=" * 60)
print("""
ReAct (Yao et al., 2022) 是最经典的 Agent 设计模式。
核心思路：让 LLM 交替进行"思考"和"行动"。

  循环流程：
    Thought → Action → Observation → Thought → Action → ... → Final Answer

  示例（查询天气）：
  ┌──────────────────────────────────────────────────────┐
  │ Question: 北京今天比上海热吗？                       │
  │                                                      │
  │ Thought 1: 我需要知道两个城市的温度，先查北京        │
  │ Action 1:  search_weather("北京")                     │
  │ Observation 1: 北京今天 28°C                         │
  │                                                      │
  │ Thought 2: 知道北京了，再查上海                      │
  │ Action 2:  search_weather("上海")                     │
  │ Observation 2: 上海今天 25°C                         │
  │                                                      │
  │ Thought 3: 北京 28°C > 上海 25°C，所以北京更热       │
  │ Final Answer: 是的，北京今天 28°C，比上海 25°C 高。  │
  └──────────────────────────────────────────────────────┘

ReAct 的优势：
  ✅ 推理过程可追溯（有 Thought 链）
  ✅ 能根据中间结果动态调整策略
  ✅ 比纯 CoT 更能处理需要外部信息的任务
""")

# ============================================================
# 3. Function Calling（工具调用）
# ============================================================
print("\n" + "=" * 60)
print("【3. Function Calling（工具调用机制）】")
print("=" * 60)
print("""
Function Calling 是让 LLM 调用外部工具的标准接口。

基本流程：
  1. 开发者定义可用工具的 schema（名字、参数、描述）
  2. 把 schema 放进 system prompt 或特殊 token
  3. LLM 生成结构化的工具调用请求（通常是 JSON）
  4. 系统解析请求 → 执行工具 → 把结果返回给 LLM
  5. LLM 根据结果继续推理

工具定义示例（OpenAI 格式）：
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取指定城市的天气",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string", "description": "城市名"}
        },
        "required": ["city"]
      }
    }
  }

LLM 的输出：
  {"name": "get_weather", "arguments": {"city": "北京"}}

【面试考点】
Q: Function Calling 和 Prompt Engineering 调用工具有什么区别？
A: FC 是模型训练时就学过的结构化输出能力，生成格式更可靠；
   纯 Prompt 让模型输出特定格式容易出错（JSON 格式不对等）。
""")

# ============================================================
# 4. 定义工具集
# ============================================================
print("\n" + "=" * 60)
print("【4. 定义工具集 — 手写 Agent 第一步】")
print("=" * 60)

# 定义几个简单的工具
TOOLS = {
    "calculator": {
        "description": "计算数学表达式，支持加减乘除、乘方、开方等",
        "parameters": {"expression": "要计算的数学表达式（Python 语法）"},
    },
    "string_length": {
        "description": "计算字符串长度",
        "parameters": {"text": "要计算长度的字符串"},
    },
    "knowledge_lookup": {
        "description": "查询知识库中的事实",
        "parameters": {"query": "查询关键词"},
    },
}

# 简单知识库
KNOWLEDGE_BASE = {
    "地球半径": "地球平均半径约为 6371 千米",
    "光速": "光速约为 299792458 米/秒",
    "圆周率": "圆周率 π ≈ 3.14159265358979",
    "transformer": "Transformer 由 Google 在 2017 年提出，论文名 Attention Is All You Need",
    "python": "Python 由 Guido van Rossum 在 1991 年创建",
}


def execute_tool(tool_name, arguments):
    """执行工具并返回结果"""
    if tool_name == "calculator":
        expr = arguments.get("expression", "")
        # 安全地计算数学表达式（只允许数学运算）
        allowed_names = {
            "abs": abs, "round": round,
            "sqrt": math.sqrt, "pow": pow,
            "pi": math.pi, "e": math.e,
            "sin": math.sin, "cos": math.cos, "log": math.log,
        }
        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            return f"计算结果: {expr} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    elif tool_name == "string_length":
        text = arguments.get("text", "")
        return f"字符串 '{text}' 的长度为 {len(text)} 个字符"

    elif tool_name == "knowledge_lookup":
        query = arguments.get("query", "").lower()
        for key, value in KNOWLEDGE_BASE.items():
            if key in query or query in key:
                return f"查询结果: {value}"
        return "未找到相关信息"

    else:
        return f"未知工具: {tool_name}"


print("可用工具：")
for name, info in TOOLS.items():
    print(f"  {name}: {info['description']}")
    print(f"    参数: {info['parameters']}")

print("\n--- 工具调用测试 ---")
test_calls = [
    ("calculator", {"expression": "6371 * 2 * pi"}),
    ("string_length", {"text": "Hello, Agent!"}),
    ("knowledge_lookup", {"query": "光速"}),
]
for tool_name, args in test_calls:
    result = execute_tool(tool_name, args)
    print(f"  {tool_name}({args}) → {result}")

# ============================================================
# 5. 手写 ReAct Agent 循环
# ============================================================
print("\n\n" + "=" * 60)
print("【5. 手写 ReAct Agent 循环】")
print("=" * 60)

REACT_SYSTEM_PROMPT = """你是一个能使用工具的 AI 助手。

可用工具：
1. calculator(expression) - 计算数学表达式
2. string_length(text) - 计算字符串长度
3. knowledge_lookup(query) - 查询知识库

请按以下格式回答：

Thought: 分析需要做什么
Action: tool_name(arguments_json)
（等待系统返回 Observation）

如果已有足够信息，直接给出：
Thought: 已有足够信息
Final Answer: 最终回答
"""


def parse_agent_output(text):
    """解析 LLM 输出，提取 Action 或 Final Answer"""
    # 尝试匹配 Final Answer
    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        return {"type": "final", "content": final_match.group(1).strip()}

    # 尝试匹配 Action
    action_match = re.search(r"Action:\s*(\w+)\((\{.*?\})\)", text, re.DOTALL)
    if action_match:
        tool_name = action_match.group(1)
        try:
            arguments = json.loads(action_match.group(2))
        except json.JSONDecodeError:
            arguments = {}
        return {"type": "action", "tool": tool_name, "arguments": arguments}

    return {"type": "unknown", "content": text}


def run_react_agent(question, model, tokenizer, max_steps=5):
    """运行 ReAct Agent 循环"""
    print(f"\n{'─' * 50}")
    print(f"用户问题: {question}")
    print(f"{'─' * 50}")

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        # LLM 推理
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        print(f"\n[Step {step + 1}] LLM 输出:")
        print(f"  {response[:300]}")

        # 解析 LLM 输出
        parsed = parse_agent_output(response)

        if parsed["type"] == "final":
            print(f"\n✅ Final Answer: {parsed['content']}")
            return parsed["content"]

        elif parsed["type"] == "action":
            # 执行工具
            observation = execute_tool(parsed["tool"], parsed["arguments"])
            print(f"  → 执行工具: {parsed['tool']}({parsed['arguments']})")
            print(f"  → Observation: {observation}")

            # 把 LLM 输出和 observation 加入对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        else:
            print(f"  → 无法解析输出，结束")
            return response

    print(f"\n⚠️ 达到最大步数 {max_steps}，强制结束")
    return "达到最大推理步数"


# 加载模型
model_path = "/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"模型已加载: {model_path}")

print("""
说明：0.5B 模型的 Agent 能力有限，可能无法正确遵循 ReAct 格式。
  实际生产中使用 7B+ 模型（尤其是 Qwen2.5-7B-Instruct 以上）效果会好很多。
  这里主要是演示 Agent 循环的代码结构。
""")

# 运行 Agent
test_questions = [
    "地球的周长大约是多少千米？请先查询地球半径，然后用公式 2*pi*r 计算。",
    "'Transformer Attention Is All You Need' 这个字符串有多长？",
]

for q in test_questions:
    run_react_agent(q, model, tokenizer, max_steps=3)

# ============================================================
# 6. 主流 Agent 框架对比
# ============================================================
print("\n\n" + "=" * 60)
print("【6. 主流 Agent 框架对比】")
print("=" * 60)
print("""
┌──────────────┬──────────────────────────────────────────────┐
│ 框架         │ 特点                                         │
├──────────────┼──────────────────────────────────────────────┤
│ LangChain    │ 最早、生态最全，工具/记忆/链式调用           │
│              │ 缺点：抽象层太重，调试困难                   │
├──────────────┼──────────────────────────────────────────────┤
│ LangGraph    │ LangChain 团队出品，基于状态图               │
│              │ 适合复杂的多步骤、有分支的 Agent 流程        │
├──────────────┼──────────────────────────────────────────────┤
│ AutoGen      │ Microsoft 出品，Multi-Agent 对话框架         │
│ (AG2)        │ Agent 之间互相发消息协作                     │
├──────────────┼──────────────────────────────────────────────┤
│ CrewAI       │ 多角色 Agent 协作，类似"团队"概念            │
│              │ 每个 Agent 有角色、目标、工具                │
├──────────────┼──────────────────────────────────────────────┤
│ OpenAI       │ OpenAI 官方 Agent SDK                        │
│ Agents SDK   │ Handoff 机制、Guardrails、Tracing            │
├──────────────┼──────────────────────────────────────────────┤
│ Dify / Coze  │ 低代码 Agent 平台，拖拽式搭建                │
│              │ 适合快速原型和非技术用户                     │
├──────────────┼──────────────────────────────────────────────┤
│ MCP          │ Anthropic 提出的 Model Context Protocol      │
│              │ 统一工具接口标准，让 Agent 即插即用工具      │
└──────────────┴──────────────────────────────────────────────┘
""")

# ============================================================
# 7. Multi-Agent 系统
# ============================================================
print("\n" + "=" * 60)
print("【7. Multi-Agent 多智能体系统】")
print("=" * 60)
print("""
当任务复杂到一个 Agent 难以胜任时，可以用多个 Agent 协作。

典型架构：

  1. 编排者模式（Orchestrator）
     ┌──────────────────┐
     │   Orchestrator   │ ← 中央调度
     │   (主 Agent)     │
     └──┬────┬────┬─────┘
        ↓    ↓    ↓
     [搜索]  [代码] [分析]  ← 专业子 Agent
     Agent   Agent  Agent

  2. 流水线模式（Pipeline）
     [提取] → [分析] → [撰写] → [审核]
     每个 Agent 处理一个阶段，输出传给下一个

  3. 辩论模式（Debate）
     [Agent A] ←→ [Agent B]
     两个 Agent 互相挑战对方的答案，最终达成共识
     提高复杂推理的准确率

  4. 投票模式（Ensemble）
     [Agent 1] ─┐
     [Agent 2] ─┼→ 投票/聚合 → 最终答案
     [Agent 3] ─┘

实际案例：
  - 代码开发: Planner Agent + Coder Agent + Reviewer Agent
  - 研究报告: Search Agent + Analyst Agent + Writer Agent
  - 客服系统: Router Agent → 售前/售后/技术支持 Agent
""")

# ============================================================
# 8. Agent 的记忆系统
# ============================================================
print("\n" + "=" * 60)
print("【8. Agent 的记忆系统】")
print("=" * 60)
print("""
记忆让 Agent 能利用历史经验，而不是每次从零开始。

三种记忆：
  ┌─────────────┬────────────────────────────────────────┐
  │ 短期记忆    │ 当前对话的上下文窗口                    │
  │ (Working)   │ 实现: 就是 chat history                 │
  │             │ 限制: 受上下文长度限制（4K~128K）       │
  ├─────────────┼────────────────────────────────────────┤
  │ 长期记忆    │ 跨会话持久化的信息                      │
  │ (Long-term) │ 实现: 向量数据库存储关键对话/摘要       │
  │             │ 类似: RAG，但检索的是历史交互           │
  ├─────────────┼────────────────────────────────────────┤
  │ 外部知识    │ 结构化的知识库                          │
  │ (External)  │ 实现: 数据库、API、文件系统             │
  │             │ 特点: 不随对话变化，全局共享             │
  └─────────────┴────────────────────────────────────────┘

短期记忆的常见管理策略：
  1. 滑动窗口: 只保留最近 N 轮对话
  2. 摘要压缩: 用 LLM 把旧对话总结成摘要
  3. 重要性筛选: 保留关键信息，丢弃闲聊
""")

# 演示：摘要压缩的思路
print("--- 演示：短期记忆管理 ---")

chat_history = [
    {"role": "user", "content": "我想学 Python"},
    {"role": "assistant", "content": "好的，你有编程基础吗？"},
    {"role": "user", "content": "我学过 C 语言"},
    {"role": "assistant", "content": "那很好，Python 上手会很快..."},
    {"role": "user", "content": "推荐什么学习资源？"},
    {"role": "assistant", "content": "推荐《Python Crash Course》..."},
    {"role": "user", "content": "我对数据分析感兴趣"},
    {"role": "assistant", "content": "那可以学 pandas 和 matplotlib..."},
]

window_size = 4  # 只保留最近 4 轮
windowed = chat_history[-window_size:]
print(f"  原始历史: {len(chat_history)} 条消息")
print(f"  滑动窗口 (k={window_size}): 保留最近 {len(windowed)} 条")
for msg in windowed:
    print(f"    [{msg['role']}] {msg['content'][:40]}...")

print("""
  更高级的做法是让 LLM 生成摘要：
    "用户是有 C 语言基础的 Python 初学者，对数据分析感兴趣"
  → 把摘要放在 system prompt 里，节省上下文空间
""")

# ============================================================
# 9. Agent 规划策略
# ============================================================
print("\n\n" + "=" * 60)
print("【9. Agent 规划策略】")
print("=" * 60)
print("""
复杂任务需要 Agent 先规划再执行，常见策略：

1. 顺序规划（Plan-then-Execute）
   先生成完整计划 → 逐步执行 → 每步可能微调后续计划
   ┌──────────────────────────┐
   │ "写一篇技术博客"        │
   │ Plan:                    │
   │   1. 确定主题和大纲      │
   │   2. 搜索相关资料        │
   │   3. 写初稿              │
   │   4. 审查和修改          │
   │   5. 格式化发布          │
   └──────────────────────────┘

2. 树搜索（Tree of Thoughts）
   每步生成多个候选 → 评估 → 选最优 → 展开下一步
   适合需要探索多种可能的推理任务

3. 反思/自我修正（Reflexion）
   执行 → 评估结果 → 如果不满意就反思错误 → 重试
   关键: Agent 能自己判断"做得好不好"

4. 递归分解（Divide and Conquer）
   大任务 → 拆成子任务 → 子任务可能再拆 → 递归解决

生产中的建议：
  - 简单任务: ReAct 就够了
  - 中等任务: Plan-then-Execute
  - 复杂任务: 多 Agent + 反思机制
""")

# ============================================================
# 10. Agent 常见坑与最佳实践
# ============================================================
print("\n" + "=" * 60)
print("【10. Agent 常见坑与最佳实践】")
print("=" * 60)
print("""
常见问题：

  ❌ 问题1: Agent 陷入循环（反复调用同一个工具）
     原因: 工具返回结果不明确 或 Prompt 缺少终止条件
     方案: 设 max_steps 上限 + 在 prompt 中明确终止条件

  ❌ 问题2: 工具调用参数格式错误
     原因: 小模型 JSON 生成能力弱
     方案: 用 7B+ 模型 / 约束解码 / 重试机制

  ❌ 问题3: Agent 不知道该用哪个工具
     原因: 工具描述不清晰 / 工具太多选花眼
     方案: 工具描述写清楚 + 给 few-shot 示例 + 控制工具数量(<10)

  ❌ 问题4: 长对话后 Agent 忘记初始目标
     原因: 上下文太长，关键信息被稀释
     方案: 定期在 prompt 中重复任务目标 / 摘要压缩历史

  ❌ 问题5: Agent 自信地编造工具调用结果
     原因: LLM 的幻觉问题
     方案: 严格检查工具返回，LLM 生成的"结果"不可信

最佳实践：
  ✅ 工具描述越清晰越好（想象在给实习生写文档）
  ✅ 必须有 max_steps / timeout 防失控
  ✅ 每步记录日志，方便调试
  ✅ 工具返回要结构化，减少 LLM 解析负担
  ✅ 重要操作加人工确认（Human-in-the-loop）
  ✅ 小模型做简单路由，大模型做复杂推理
""")

# ============================================================
# 11. MCP — 工具生态的统一标准
# ============================================================
print("\n" + "=" * 60)
print("【11. MCP — Model Context Protocol】")
print("=" * 60)
print("""
MCP 是 Anthropic 提出的开放协议，目标是统一 Agent 的工具接口。

痛点: 每个 Agent 框架的工具定义和调用方式都不同。
  - OpenAI Function Calling 用一种 JSON schema
  - LangChain Tool 用另一种定义
  - 每接一个新工具都要写一遍适配代码

MCP 的思路（类似 USB 接口）：
  ┌────────────┐                    ┌──────────────┐
  │ Agent      │                    │ MCP Server   │
  │ (任意框架) │───── MCP 协议 ─────│ (工具提供方) │
  └────────────┘                    └──────────────┘

  MCP Server 暴露标准化的工具接口：
    - tools/list     → 列出可用工具
    - tools/call     → 调用工具
    - resources/read → 读取资源

  好处：
    ✅ 工具开发者写一次 MCP Server，所有 Agent 都能用
    ✅ Agent 开发者接一次 MCP Client，所有工具都能用
    ✅ 类似 LSP（Language Server Protocol）对编辑器的价值

目前支持 MCP 的：
  - Claude Desktop、VS Code Copilot
  - Cursor、Windsurf 等 AI IDE
  - 社区已有大量 MCP Server（GitHub、数据库、搜索等）
""")

# ============================================================
# 12. 总结
# ============================================================
print("\n" + "=" * 60)
print("【12. 本课总结与面试要点】")
print("=" * 60)
print("""
✅ 核心概念：
  1. Agent = LLM + 工具 + 规划 + 记忆
  2. ReAct 是最经典的范式：Thought → Action → Observation 循环
  3. Function Calling 让 LLM 生成结构化的工具调用请求
  4. Multi-Agent 通过角色分工解决复杂任务
  5. 记忆分三层：短期（对话）、长期（向量库）、外部（知识库）
  6. MCP 是统一工具接口的开放协议

✅ 面试高频题：
  1. Agent 和普通 LLM 对话的区别？（能调用工具、多步推理、有记忆）
  2. ReAct 的流程？（Thought-Action-Observation 循环）
  3. Agent 怎么防止死循环？（max_steps + 终止条件 + timeout）
  4. Agent 和 RAG 的关系？（RAG 是只有检索工具的简单 Agent）
  5. Multi-Agent 有哪些常见架构？（编排者、流水线、辩论、投票）
  6. 什么是 MCP？（统一工具接口协议，类似 USB/LSP）
  7. Agent 的记忆怎么管理？（滑动窗口 / 摘要压缩 / 向量存储）
  8. 小模型能做 Agent 吗？（简单路由可以，复杂推理需要 7B+）
  9. 怎么评估 Agent 效果？（任务完成率、步骤效率、工具调用准确率）

全部课程完成！🎓
  第1阶段（基础）: 01 Tokenizer → 02 Attention → 03 推理 → 04 LoRA
  第2阶段（进阶）: 05 分布式 → 06 DPO/GRPO → 07 推理优化 → 08 RAG → 09 Agent
""")
