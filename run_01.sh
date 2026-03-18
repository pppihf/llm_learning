#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
cd /workspace/huangzh14@xiaopeng.com/llm_learning
python 01_tokenizer_basics.py 2>&1
