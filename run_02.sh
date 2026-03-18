#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
cd /workspace/huangzh14@xiaopeng.com/llm_learning
python 02_attention_mechanism.py > /workspace/huangzh14@xiaopeng.com/llm_learning/run_02_output.txt 2>&1
echo "EXIT_CODE=$?"
