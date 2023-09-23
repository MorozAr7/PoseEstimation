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
LR = 0.0001
BATCH_SIZE = 512

NUM_DATA = {"Training": 25000, "Validation": 5000}
NUM_EPOCHS = 1000
USE_AUGMENTATION = True
PROJECTION_TYPE_LOSS = "2D"
LOSS_TYPE = 0 # 0 for disentangled, 1 for entangled, 2 for combined
LOSS_WEIGHTS = {"R": 1, "XY": 1, "Z" : 1, "T": 1}
MODEL_NAME = ""

LR_DECAY_EPOCHS = list(range(100, NUM_EPOCHS, 100))
LR_DECAY_FACTOR = 2