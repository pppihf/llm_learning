"""
快速测试微调后的 Qwen3-VL 中文模型
用法: python3 test_qwen3vl_chinese.py
"""
import json
import random
import torch
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

# ===== 配置 =====
MODEL_PATH = "/publicdata/huggingface.co/Qwen/Qwen3-VL-8B-Instruct"
LORA_PATH = "/workspace/huangzh14@xiaopeng.com/llm_learning/output/qwen3vl_allava_chinese_lora/v2-20260325-183048/checkpoint-7600"
VAL_DATA = "/workspace/huangzh14@xiaopeng.com/llm_learning/data/allava_chinese/val.jsonl"
NUM_SAMPLES = 5  # 测试几条

def load_samples(path, n):
    samples = []
    with open(path) as f:
        lines = f.readlines()
    chosen = random.sample(lines, min(n, len(lines)))
    for line in chosen:
        samples.append(json.loads(line))
    return samples

def main():
    print("=" * 60)
    print("加载模型 + LoRA...")
    print(f"  Base: {MODEL_PATH}")
    print(f"  LoRA: {LORA_PATH}")
    print("=" * 60)

    engine = TransformersEngine(
        MODEL_PATH,
        adapters=[LORA_PATH],
        torch_dtype=torch.bfloat16,
    )

    samples = load_samples(VAL_DATA, NUM_SAMPLES)
    print(f"\n随机抽取 {len(samples)} 条验证样本测试:\n")

    for i, sample in enumerate(samples):
        messages = sample.get("messages", [])
        images = sample.get("images", [])

        # 找 user 问题和 ground truth
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        gt_answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        # 构造推理请求（只传 user 消息）
        infer_messages = [{"role": "user", "content": user_msg}]
        request = InferRequest(messages=infer_messages, images=images)
        config = RequestConfig(max_tokens=256, temperature=0.0)

        print(f"{'='*60}")
        print(f"[样本 {i+1}]")
        if images:
            print(f"  图片: {images[0].split('/')[-1]}")
        # 显示问题（去掉 <image> tag）
        question = user_msg.replace("<image>", "").strip()[:150]
        print(f"  问题: {question}")
        print(f"  参考答案: {gt_answer[:200]}")

        resp = engine.infer([request], request_config=config)
        pred = resp[0].choices[0].message.content
        print(f"  模型回答: {pred[:200]}")
        print()

    print("=" * 60)
    print("测试完成！")

if __name__ == "__main__":
    main()
