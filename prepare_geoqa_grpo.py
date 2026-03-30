"""
GeoQA3 几何选择题 → ms-swift GRPO JSONL 格式
将 GeoQA3 pickle 数据转为 GRPO 可用的格式：
- messages: [user + 带图的几何题]（不包含 assistant 回答）
- images: [图片路径]
- solution: 正确答案选项（如 "B"）

GRPO 训练时模型自行生成回答，reward 函数判断对错。
"""
import json
import os
import pickle
import random
import numpy as np
from PIL import Image

GEOQA_DIR = "/dataset-pretrain/VLM-datasets/GeoQA/GeoQA3"
OUTPUT_DIR = "/workspace/huangzh14@xiaopeng.com/llm_learning/data/geoqa_grpo"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(IMAGE_DIR, exist_ok=True)

OPTION_LABELS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "你是一个数学几何专家。请仔细观察图片，分析题目，一步步推理后给出答案。\n"
    "请按以下格式回答：\n"
    "<think>\n在这里写出你的推理过程\n</think>\n"
    "<answer>\n在这里写出答案选项（A/B/C/D）\n</answer>"
)

def process_split(data, split_name):
    """处理一个 split (train/test)"""
    samples = []
    skipped = 0

    for idx, item in enumerate(data):
        subject = item.get("subject", "")
        choices = item.get("choices", [])
        label = item.get("label", 0)  # 0-based index
        image_array = item.get("image", None)

        if not choices or label >= len(choices):
            skipped += 1
            continue

        # 保存图片
        img_filename = f"{split_name}_{idx:05d}.png"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        if image_array is not None and not os.path.exists(img_path):
            if len(image_array.shape) == 2:
                img = Image.fromarray(image_array.astype(np.uint8), mode='L')
            else:
                img = Image.fromarray(image_array.astype(np.uint8))
            img.save(img_path)

        # 构造选项文本
        options_text = "\n".join(
            f"({OPTION_LABELS[i]}) {c}" for i, c in enumerate(choices)
        )

        # 正确答案选项
        correct_option = OPTION_LABELS[label]

        # 构造用户消息
        user_content = f"<image>\n{subject}\n\n{options_text}"

        # GRPO 格式: messages 只有 user 消息（模型自己生成回答）
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "images": [img_path],
            "solution": correct_option,
        }
        samples.append(sample)

    return samples, skipped


def main():
    print("=" * 60)
    print("GeoQA3 → GRPO 数据准备")
    print("=" * 60)

    # 加载数据
    with open(os.path.join(GEOQA_DIR, "train.pk"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(GEOQA_DIR, "test.pk"), "rb") as f:
        test_data = pickle.load(f)

    print(f"原始数据: train={len(train_data)}, test={len(test_data)}")

    # 处理
    train_samples, train_skip = process_split(train_data, "train")
    test_samples, test_skip = process_split(test_data, "test")

    print(f"转换完成: train={len(train_samples)} (跳过{train_skip}), "
          f"test={len(test_samples)} (跳过{test_skip})")

    # 写入 JSONL
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test.jsonl")

    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(test_path, "w") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n输出文件:")
    print(f"  {train_path} ({len(train_samples)} 条)")
    print(f"  {test_path} ({len(test_samples)} 条)")
    print(f"  图片目录: {IMAGE_DIR}")

    # 显示样例
    print(f"\n样例数据:")
    s = train_samples[0]
    print(f"  question: {s['messages'][1]['content'][:100]}")
    print(f"  solution: {s['solution']}")
    print(f"  image: {s['images'][0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
