import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = CURR_DIR_PATH.split("/")[0] + "/" + CURR_DIR_PATH.split("/")[1] + "/" + CURR_DIR_PATH.split("/")[2] + "/" + CURR_DIR_PATH.split("/")[3]

DATASET_PATH = MAIN_DIR_PATH + "/" + "Dataset/"
OBJECT_TYPE = "Chassis"
CAM_DATA_FILE = "camera_data_1.json"
IMG_SIZE = 224
UVW_RANGE = 256
LAST_IMAGE = 3001
DATA_AMOUNT = {"Training": 100000, "Validation": 10000}
