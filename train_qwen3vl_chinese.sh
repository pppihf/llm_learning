#!/bin/bash
# ============================================================
# Qwen3-VL-8B 中文多模态微调训练脚本
# 数据集: ALLaVA-Chinese (中文图文对话 + 中文 VQA)
# 方法: LoRA + DeepSpeed ZeRO-2
# ============================================================
#
# 使用方法:
#   fuyao shell --job-name=bifrost-2026032318040000-huangzh14
#   cd /workspace/huangzh14@xiaopeng.com/llm_learning
#   bash train_qwen3vl_chinese.sh
#
# GPU 说明:
#   总共 16 张卡, gpu_run.py 占了 8-15, 训练可用 0-7
# ============================================================

set -e

# ============ 路径配置 ============
WORKSPACE="/workspace/huangzh14@xiaopeng.com/llm_learning"
MODEL_PATH="/publicdata/huggingface.co/Qwen/Qwen3-VL-8B-Instruct"
TRAIN_DATA="${WORKSPACE}/data/allava_chinese/train.jsonl"
VAL_DATA="${WORKSPACE}/data/allava_chinese/val.jsonl"
OUTPUT_DIR="${WORKSPACE}/output/qwen3vl_allava_chinese_lora"

# ============ 训练超参 ============
NUM_GPUS=8
CUDA_DEVICES="0,1,2,3,4,5,6,7"
EPOCHS=1
BATCH_SIZE=1
GRAD_ACCUM=8          # 等效 batch = 1 * 8 * 8 = 64
LR=1e-4
LORA_RANK=16
LORA_ALPHA=32
MAX_LENGTH=2048
MAX_PIXELS=1003520    # 1280 * 28 * 28

# ============ Step 1: 数据准备 ============
echo "========================================"
echo "Step 1: 准备 ALLaVA-Chinese 数据"
echo "========================================"

if [ ! -f "$TRAIN_DATA" ]; then
    python3 "${WORKSPACE}/prepare_allava_chinese.py"
else
    echo "数据已存在，跳过转换"
    wc -l "$TRAIN_DATA" "$VAL_DATA"
fi

# ============ Step 2: 训练 ============
echo ""
echo "========================================"
echo "Step 2: 开始训练"
echo "========================================"
echo "  模型: ${MODEL_PATH}"
echo "  训练集: $(wc -l < ${TRAIN_DATA}) 条"
echo "  验证集: $(wc -l < ${VAL_DATA}) 条"
echo "  GPU: ${CUDA_DEVICES} (${NUM_GPUS} 张)"
echo "  LoRA rank: ${LORA_RANK}, alpha: ${LORA_ALPHA}"
echo "  等效 batch size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  输出: ${OUTPUT_DIR}"
echo "========================================"

export DISABLE_MLFLOW_INTEGRATION=true
export WANDB_DISABLED=true

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
MAX_PIXELS=${MAX_PIXELS} \
NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model "${MODEL_PATH}" \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --torch_dtype bfloat16 \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --tuner_type lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length ${MAX_LENGTH} \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --split_dataset_ratio 0 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "========================================"
echo "训练完成！"
echo "模型输出: ${OUTPUT_DIR}"
echo "========================================"
