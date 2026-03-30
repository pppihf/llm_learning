"""
对比测试：基础模型 vs GRPO 微调模型
在相同的 GeoQA 题目上对比两个模型的推理质量和准确率
"""
import re
import json
import random
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

# ============ 配置 ============
BASE_MODEL = "/publicdata/huggingface.co/Qwen/Qwen3-VL-8B-Instruct"
GRPO_CKPT = "/workspace/huangzh14@xiaopeng.com/llm_learning/output/qwen3vl_geoqa_grpo/v4-20260327-101537/checkpoint-2622"
TEST_DATA = "/workspace/huangzh14@xiaopeng.com/llm_learning/data/geoqa_grpo/test.jsonl"
NUM_TEST = 30   # 30题，统计结果更可靠

SYSTEM_PROMPT = (
    "你是一个数学几何专家。请仔细观察图片，分析题目，一步步推理后给出答案。\n"
    "请按以下格式回答：\n"
    "<think>\n在这里写出你的推理过程\n</think>\n"
    "<answer>\n在这里写出答案选项（A/B/C/D）\n</answer>"
)


def extract_answer(text):
    m = re.search(r'<answer>\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'[选答故]([A-D])', text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r'\(([A-D])\)', text)
    if matches:
        return matches[-1].upper()
    return None


def load_test_samples(path, n):
    with open(path) as f:
        lines = f.readlines()
    random.seed(42)
    selected = random.sample(lines, n)
    samples = []
    for line in selected:
        data = json.loads(line)
        samples.append({
            "question": data["messages"][1]["content"],
            "image": data["images"][0],
            "solution": data["solution"],
        })
    return samples


def evaluate(engine, samples, label):
    correct = 0
    has_format = 0
    total_len = 0
    results = []

    for i, s in enumerate(samples):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'<image>\n{s["question"]}'},
        ]
        req = RequestConfig(max_tokens=512, temperature=0.0)
        infer_req = InferRequest(messages=messages, images=[s["image"]])
        resp = engine.infer([infer_req], req)
        output = resp[0].choices[0].message.content

        pred = extract_answer(output)
        is_correct = (pred == s["solution"])
        has_think = bool(re.search(r'<think>.*?</think>', output, re.DOTALL))
        has_ans_tag = bool(re.search(r'<answer>', output))
        fmt_ok = has_think and has_ans_tag

        if is_correct:
            correct += 1
        if fmt_ok:
            has_format += 1
        total_len += len(output)

        results.append({
            "idx": i + 1,
            "question": s["question"],
            "gt": s["solution"],
            "pred": pred,
            "correct": is_correct,
            "fmt_ok": fmt_ok,
            "output": output,
        })

        status = "✓" if is_correct else "✗"
        fmt_icon = "📋" if fmt_ok else "  "
        print(f"  [{i+1:2d}] {status} {fmt_icon} GT={s['solution']} Pred={pred or '?'}")

    n = len(samples)
    acc = correct / n * 100
    fmt_rate = has_format / n * 100
    avg_len = total_len / n
    print(f"\n{'─'*50}")
    print(f"[{label}]")
    print(f"  准确率     : {correct}/{n} = {acc:.1f}%")
    print(f"  格式规范率 : {has_format}/{n} = {fmt_rate:.1f}%")
    print(f"  平均输出长度: {avg_len:.0f} chars")
    print(f"{'─'*50}")
    return acc, fmt_rate, avg_len, results


def show_comparison(base_results, grpo_results):
    """逐题展示两个模型的对比"""
    print("\n" + "=" * 70)
    print("逐题对比（仅展示两模型结论不一致 或 有推理过程的题目）")
    print("=" * 70)

    shown = 0
    for b, g in zip(base_results, grpo_results):
        # 只展示：结论不同 或 GRPO有格式而base没有
        if b["correct"] == g["correct"] and b["fmt_ok"] == g["fmt_ok"]:
            continue
        if shown >= 5:
            break

        gt = b["gt"]
        print(f"\n题 {b['idx']}: {b['question'][:60].strip()}...")
        print(f"正确答案: {gt}")
        print(f"\n  【基础模型】 Pred={b['pred'] or '?'}  {'✓' if b['correct'] else '✗'}  {'📋有格式' if b['fmt_ok'] else '无格式'}")
        print(f"  输出: {b['output'][:300].strip()}")
        print(f"\n  【GRPO模型】 Pred={g['pred'] or '?'}  {'✓' if g['correct'] else '✗'}  {'📋有格式' if g['fmt_ok'] else '无格式'}")
        print(f"  输出: {g['output'][:300].strip()}")
        print("─" * 70)
        shown += 1


def main():
    print("=" * 60)
    print("基础模型 vs GRPO 微调模型 对比测试")
    print(f"测试集: GeoQA3，{NUM_TEST} 道几何选择题")
    print("=" * 60)

    samples = load_test_samples(TEST_DATA, NUM_TEST)

    # ── Step 1: 测试基础模型 ──
    print(f"\n{'='*60}")
    print("Step 1/2: 测试基础模型（无 LoRA）")
    print("=" * 60)
    base_engine = TransformersEngine(BASE_MODEL, torch_dtype="bfloat16")
    base_acc, base_fmt, base_len, base_results = evaluate(base_engine, samples, "基础模型")
    del base_engine  # 释放显存

    # ── Step 2: 测试 GRPO 微调模型 ──
    print(f"\n{'='*60}")
    print("Step 2/2: 测试 GRPO 微调模型（加载 LoRA）")
    print("=" * 60)
    grpo_engine = TransformersEngine(BASE_MODEL, adapters=[GRPO_CKPT], torch_dtype="bfloat16")
    grpo_acc, grpo_fmt, grpo_len, grpo_results = evaluate(grpo_engine, samples, "GRPO 模型")

    # ── 汇总对比 ──
    print("\n" + "=" * 60)
    print("【最终对比汇总】")
    print("=" * 60)
    print(f"{'指标':<16} {'基础模型':>12} {'GRPO模型':>12} {'变化':>10}")
    print("─" * 52)
    print(f"{'准确率':<16} {base_acc:>11.1f}% {grpo_acc:>11.1f}% {grpo_acc-base_acc:>+9.1f}%")
    print(f"{'格式规范率':<14} {base_fmt:>11.1f}% {grpo_fmt:>11.1f}% {grpo_fmt-base_fmt:>+9.1f}%")
    print(f"{'平均输出长度':<13} {base_len:>11.0f}  {grpo_len:>11.0f}  {grpo_len-base_len:>+9.0f}")
    print("─" * 52)

    # 计算两模型同题对比
    both_correct = sum(1 for b, g in zip(base_results, grpo_results) if b["correct"] and g["correct"])
    base_only = sum(1 for b, g in zip(base_results, grpo_results) if b["correct"] and not g["correct"])
    grpo_only = sum(1 for b, g in zip(base_results, grpo_results) if not b["correct"] and g["correct"])
    both_wrong = sum(1 for b, g in zip(base_results, grpo_results) if not b["correct"] and not g["correct"])

    print(f"\n两模型交叉分析（共 {NUM_TEST} 题）:")
    print(f"  两者都对   : {both_correct} 题")
    print(f"  仅基础模型对: {base_only} 题  ← GRPO 退步")
    print(f"  仅GRPO模型对: {grpo_only} 题  ← GRPO 改进")
    print(f"  两者都错   : {both_wrong} 题")

    # 逐题对比
    show_comparison(base_results, grpo_results)


if __name__ == "__main__":
    main()
