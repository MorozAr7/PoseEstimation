import torch

CUDA_DEVICE = 5
DEVICE = "mps" if getattr(torch, 'has_mps', False) else CUDA_DEVICE if torch.cuda.is_available() else "cpu"
