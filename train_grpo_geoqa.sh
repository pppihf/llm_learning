#!/bin/bash
# ============================================================
# Qwen3-VL-8B GRPO 训练脚本 (几何推理强化学习)
# 数据集: GeoQA3 中文几何选择题
# 方法: GRPO + LoRA + DeepSpeed ZeRO-2
# ============================================================
#
# GRPO 核心思想:
#   1. 对每个 prompt 生成 G 个回答
#   2. 用 reward 函数对每个回答打分
#   3. 组内相对比较（好的回答 vs 差的回答）
#   4. 策略梯度更新，让模型倾向于产出高分回答
#
# 使用方法:
#   fuyao shell --job-name=bifrost-2026032318040000-huangzh14
#   cd /workspace/huangzh14@xiaopeng.com/llm_learning
#   bash train_grpo_geoqa.sh
#
# GPU 说明:
#   总共 16 张卡全部可用, 使用 0-15 共 16 张
# ============================================================

set -e

# ============ 路径配置 ============
WORKSPACE="/workspace/huangzh14@xiaopeng.com/llm_learning"
MODEL_PATH="/publicdata/huggingface.co/Qwen/Qwen3-VL-8B-Instruct"
TRAIN_DATA="${WORKSPACE}/data/geoqa_grpo/train.jsonl"
VAL_DATA="${WORKSPACE}/data/geoqa_grpo/test.jsonl"
OUTPUT_DIR="${WORKSPACE}/output/qwen3vl_geoqa_grpo"
REWARD_FILE="${WORKSPACE}/geoqa_reward.py"

# ============ 训练超参 ============
NUM_GPUS=16
CUDA_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

# GRPO 特有参数
NUM_GENERATIONS=8         # 每个 prompt 生成几个回答 (G)
                          # 越大→方差越小→但显存越大
                          # 16 张卡显存充足, 用 8 效果更好

MAX_NEW_TOKENS=512        # 最大生成长度
LEARNING_RATE=5e-6        # GRPO 学习率比 SFT 小一个量级
EPOCHS=3                  # GeoQA 只有 3499 条, 多跑几轮
BATCH_SIZE=1              # 每卡 batch size
GRAD_ACCUM=2              # 等效 BS = 1×8GPU×2 = 16 prompts
                          # GRPO 每个 prompt 有 G 个 completion

# ============ Step 1: 数据准备 ============
echo "========================================"
echo "Step 1: 准备 GeoQA3 GRPO 数据"
echo "========================================"

if [ -f "${TRAIN_DATA}" ]; then
    echo "数据已存在，跳过转换"
else
    python3 "${WORKSPACE}/prepare_geoqa_grpo.py"
fi

TRAIN_COUNT=$(wc -l < "${TRAIN_DATA}")
VAL_COUNT=$(wc -l < "${VAL_DATA}")

echo "  训练集: ${TRAIN_COUNT} 条"
echo "  验证集: ${VAL_COUNT} 条"

# ============ Step 2: 开始 GRPO 训练 ============
echo "========================================"
echo "Step 2: 开始 GRPO 训练"
echo "========================================"
echo "  模型: ${MODEL_PATH}"
echo "  数据: GeoQA3 几何选择题"
echo "  GRPO G=${NUM_GENERATIONS} (每 prompt 生成${NUM_GENERATIONS}个回答)"
echo "  GPU: ${CUDA_DEVICES} (${NUM_GPUS} 张)"
echo "  LoRA rank: 16, alpha: 32"
echo "  等效 prompt batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "  输出: ${OUTPUT_DIR}"
echo "========================================"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
NPROC_PER_NODE=${NUM_GPUS} \
swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --external_plugins "${REWARD_FILE}" \
    --reward_funcs geoqa_accuracy geoqa_format \
    --reward_weights 1.0 0.3 \
    --dataset "${TRAIN_DATA}" \
    --num_generations ${NUM_GENERATIONS} \
    --max_completion_length ${MAX_NEW_TOKENS} \
    --torch_dtype bfloat16 \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --loss_type grpo \
    --advantage_estimator grpo \
    --split_dataset_ratio 0 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "========================================"
echo "GRPO 训练完成！"
echo "========================================"
