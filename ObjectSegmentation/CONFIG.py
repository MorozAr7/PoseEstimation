import torch
import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = CURR_DIR_PATH.split("/")[0] + "/" + CURR_DIR_PATH.split("/")[1] + "/" + CURR_DIR_PATH.split("/")[2] + "/" + CURR_DIR_PATH.split("/")[3]

CUDA_DEVICE = 1
DEVICE = "mps" if getattr(torch, 'has_mps', False) else CUDA_DEVICE if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.00025
BATCH_SIZE = 256

