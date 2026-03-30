"""
Qwen3-VL TextVQA 评估脚本
加载训练好的 LoRA 模型，在验证集上测试 VQA 准确率
"""
import json
import os
import re
import torch
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def load_val_data(val_jsonl_path, max_samples=200):
    """加载验证集（取前 max_samples 条用于快速评估）"""
    data = []
    with open(val_jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= max_samples:
                break
    return data


def extract_answer(text):
    """从模型输出中提取答案（简单清洗）"""
    # 去掉常见的前缀
    text = text.strip()
    for prefix in ["The answer is", "Answer:", "答案是", "答案："]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    # 去掉句号等
    text = text.rstrip(".,;!。，；！")
    return text.strip()


def normalize_answer(s):
    """标准化答案用于比较"""
    s = s.lower().strip()
    # 去掉冠词
    s = re.sub(r'\b(a|an|the)\b', '', s)
    # 去掉标点
    s = re.sub(r'[^\w\s]', '', s)
    # 合并多余空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def evaluate(model_path, lora_path, val_jsonl_path, max_samples=200):
    """评估 VQA 准确率"""
    print("=" * 50)
    print("Qwen3-VL TextVQA 评估")
    print("=" * 50)

    # 加载模型
    print(f"\n加载基座模型: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path and os.path.exists(lora_path):
        print(f"加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()

    # 加载数据
    val_data = load_val_data(val_jsonl_path, max_samples)
    print(f"评估样本数: {len(val_data)}")

    correct = 0
    total = 0
    results = []

    for i, sample in enumerate(val_data):
        messages = sample["messages"]
        images = sample["images"]
        gt_answer = messages[1]["content"]  # assistant 的回答是 ground truth

        # 构造输入（只用 user 的问题）
        user_msg = messages[0]["content"]

        try:
            # 用 processor 处理
            from qwen_vl_utils import process_vision_info
            chat_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{images[0]}"},
                        {"type": "text", "text": user_msg.replace("<image>\n", "")},
                    ],
                }
            ]
            text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(chat_messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            generated = processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0]

            pred = extract_answer(generated)
            gt = gt_answer

            # 匹配判断（标准化后包含即算对）
            is_correct = (
                normalize_answer(pred) == normalize_answer(gt)
                or normalize_answer(gt) in normalize_answer(pred)
                or normalize_answer(pred) in normalize_answer(gt)
            )

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "question": user_msg[:80],
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_correct,
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(val_data)}] 当前准确率: {correct}/{total} = {correct / total:.1%}")

        except Exception as e:
            print(f"  样本 {i} 出错: {e}")
            continue

    # 输出结果
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"评估结果:")
    print(f"  总样本: {total}")
    print(f"  正确: {correct}")
    print(f"  准确率: {accuracy:.1%}")
    print(f"{'=' * 50}")

    # 保存详细结果
    output_file = os.path.join(os.path.dirname(val_jsonl_path), "eval_results.json")
    with open(output_file, "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": total, "details": results[:50]}, f, indent=2, ensure_ascii=False)
    print(f"详细结果已保存: {output_file}")

    # 打印几个样例
    print("\n--- 部分样例 ---")
    for r in results[:5]:
        mark = "✅" if r["correct"] else "❌"
        print(f"  {mark} Q: {r['question'][:60]}...")
        print(f"     GT: {r['ground_truth']}  |  Pred: {r['prediction']}")


if __name__ == "__main__":
    WORKSPACE = "/workspace/huangzh14@xiaopeng.com/llm_learning"
    MODEL_PATH = "/publicdata/huggingface.co/Qwen/Qwen3-VL-8B-Instruct"

    # 查找最新的 checkpoint
    output_dir = f"{WORKSPACE}/output/qwen3vl_textvqa_lora"
    lora_path = None
    if os.path.exists(output_dir):
        checkpoints = sorted([
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
        ], key=lambda x: int(x.split("-")[1]))
        if checkpoints:
            lora_path = os.path.join(output_dir, checkpoints[-1])
            print(f"找到最新 checkpoint: {lora_path}")

    if lora_path is None:
        print("未找到训练好的模型，使用基座模型评估")

    evaluate(
        model_path=MODEL_PATH,
        lora_path=lora_path,
        val_jsonl_path=f"{WORKSPACE}/data/textvqa/val.jsonl",
        max_samples=200,
    )
