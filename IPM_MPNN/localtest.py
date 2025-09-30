import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Current memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
