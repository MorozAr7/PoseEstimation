import sys

ROOT_PATH = sys.path[1]
DATASET_PATH = ROOT_PATH + "/" + "PoseEstimationDataset/"
OBJECT_TYPE = "Chassis"
CAM_DATA_FILE = "camera_data_1.json"
IMG_SIZE = 224
UVW_RANGE = 256

DATA_AMOUNT = {"Training": 100, "Validation": 10}
