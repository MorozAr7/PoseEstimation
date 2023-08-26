import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = CURR_DIR_PATH.split("/")[0] + "/" + CURR_DIR_PATH.split("/")[1] + "/" + CURR_DIR_PATH.split("/")[2]

DATASET_PATH = MAIN_DIR_PATH + "/" + "Dataset/"
OBJECT_TYPE = "Chassis"
CAM_DATA_FILE = "camera_data_1.json"
IMG_SIZE = 224
UVW_RANGE = 256

DATA_AMOUNT = {"Training": 10, "Validation": 10}
