import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print("  GPU %d: %s, Memory: %.1f GB" % (i, props.name, mem_gb))

# Check key packages
# 检查本地模型路径
import os
model_path = '/publicdata/huggingface.co/Qwen/Qwen2.5-0.5B-Instruct'
print('\nLocal model path exists:', os.path.isdir(model_path))
if os.path.isdir(model_path):
    print('Model files:', os.listdir(model_path)[:10])

packages = ['transformers', 'peft', 'accelerate', 'datasets', 'safetensors', 'sentencepiece', 'tiktoken']
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f"{pkg}: {ver}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
