import torch
import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = ""
for i in CURR_DIR_PATH.split("/")[:-1]:
    MAIN_DIR_PATH += i + "/"

sys.path.insert(0, MAIN_DIR_PATH)
sys.path.insert(1, CURR_DIR_PATH)

CUDA_DEVICE = 7
DEVICE = "mps" if getattr(torch, 'has_mps', False) else CUDA_DEVICE if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0025
NUM_EPOCHS = 500
BATCH_SIZE = 64
SAVE_MODEL = True
SUBSET_NUM_DATA = {"Training": 256, "Validation": 1}

