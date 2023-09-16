import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = ""
for i in CURR_DIR_PATH.split("/")[:-1]:
    MAIN_DIR_PATH += i + "/"

sys.path.insert(0, MAIN_DIR_PATH)
sys.path.insert(1, CURR_DIR_PATH)

DATASET_PATH = MAIN_DIR_PATH + "Dataset/"
OBJECT_TYPE = "Chassis"
CAM_DATA_FILE = "camera_data_1.json"
IMG_SIZE = 224
UVW_RANGE = 256
DATA_AMOUNT = {"Training": 256, "Validation": 1}
