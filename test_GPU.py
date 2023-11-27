import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()

    # 获取当前 GPU 的名称
    current_gpu_name = torch.cuda.get_device_name(0)

    print(f"CUDA is available. Found {num_gpus} GPU(s).")
    print(f"Current GPU: {current_gpu_name}")
else:
    print("CUDA is not available. Using CPU.")
