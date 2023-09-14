import torch
import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = ""
for i in CURR_DIR_PATH.split("/")[:-1]:
    MAIN_DIR_PATH += i + "/"

sys.path.insert(0, MAIN_DIR_PATH)
sys.path.insert(1, CURR_DIR_PATH)

CUDA_DEVICE = 5
DEVICE = "mps" if getattr(torch, 'has_mps', False) else CUDA_DEVICE if torch.cuda.is_available() else "cpu"
LR = 0.00005
BATCH_SIZE = 256
NUM_DATA = {"Training": 10000, "Validation": 2000}
NUM_EPOCHS = 1000
USE_AUGMENTATION = True
PROJECTION_TYPE_LOSS = "2D"
DISENTANGLED_LOSS = True
MODEL_NAME = ""

LR_DECAY_EPOCHS = list(range(100, NUM_EPOCHS, 100))
LR_DECAY_FACTOR = 2