"""
ALLaVA-Chinese 数据准备
1. 修复图片路径（/dataset/ → /dataset-pretrain/）
2. 过滤掉图片不存在的样本
3. 从训练数据中切出验证集
"""
import json
import os
import random

random.seed(42)

BASE = "/dataset-pretrain/VLM-datasets/ALLaVA-Chinese"
OUT_DIR = "/workspace/huangzh14@xiaopeng.com/llm_learning/data/allava_chinese"
VAL_RATIO = 0.02  # 2% 作为验证集


def fix_and_filter(input_path, output_train, output_val):
    """修复路径 + 过滤 + 切分"""
    print(f"\n处理: {os.path.basename(input_path)}")

    valid = []
    skipped = 0

    with open(input_path) as f:
        for line in f:
            sample = json.loads(line)
            images = sample.get("images", [])

            # 修复路径: /dataset/VLM-datasets/ → /dataset-pretrain/VLM-datasets/
            fixed_images = []
            all_exist = True
            for img in images:
                fixed = img.replace("/dataset/VLM-datasets/", "/dataset-pretrain/VLM-datasets/")
                if not os.path.exists(fixed):
                    all_exist = False
                    break
                fixed_images.append(fixed)

            if not all_exist:
                skipped += 1
                continue

            sample["images"] = fixed_images
            valid.append(sample)

    # 随机打乱
    random.shuffle(valid)

    # 切分
    val_size = int(len(valid) * VAL_RATIO)
    val_data = valid[:val_size]
    train_data = valid[val_size:]

    # 写入
    with open(output_train, "a") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(output_val, "a") as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  有效: {len(valid)} | 跳过: {skipped}")
    print(f"  训练: {len(train_data)} | 验证: {len(val_data)}")
    return len(train_data), len(val_data)


if __name__ == "__main__":
    print("=" * 50)
    print("ALLaVA-Chinese 数据准备")
    print("=" * 50)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, "train.jsonl")
    val_path = os.path.join(OUT_DIR, "val.jsonl")

    # 清空已有文件
    for p in [train_path, val_path]:
        if os.path.exists(p):
            os.remove(p)

    total_train = 0
    total_val = 0

    # 处理两个 JSONL 文件
    sources = [
        "allava_instruct_laion_zh_new.jsonl",   # LAION 图片对话
        "allva_instruct_vflan_zh_new.jsonl",     # 多任务 VQA
    ]

    for fname in sources:
        fp = os.path.join(BASE, fname)
        if os.path.exists(fp):
            n_train, n_val = fix_and_filter(fp, train_path, val_path)
            total_train += n_train
            total_val += n_val
        else:
            print(f"  文件不存在: {fp}")

    print(f"\n{'=' * 50}")
    print(f"完成!")
    print(f"  训练集: {total_train} 条 → {train_path}")
    print(f"  验证集: {total_val} 条 → {val_path}")
    print(f"{'=' * 50}")
