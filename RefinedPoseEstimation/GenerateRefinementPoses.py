from re import sub
from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import os
import numpy as np
import random
import cv2
from DatasetRenderer.Renderer import DatasetRenderer
from Utils.MathUtils import Transformations
from Utils.IOUtils import IOUtils
dataset_renderer = DatasetRenderer()
transformations = Transformations()
io = IOUtils()

NUM_RENDERED_POSES_PER_IMAGE = 10

def distort_target_pose(pose):
    distorted_pose = {"RotX": None, "RotY": None, "RotZ": None, "TransX": None, "TransY": None, "TransZ": None}
    for param in pose.keys():
        if param == "RotX":
            distorted_pose[param] = pose[param] + random.randint(-15, 15)
        elif param == "RotY":
            distorted_pose[param] = pose[param] + random.randint(-15, 15)
        elif param == "RotZ":
            distorted_pose[param] = pose[param] + random.randint(-15, 15)
        elif param == "TransX":
            distorted_pose[param] = pose[param] + random.randint(-25, 25)
        elif param == "TransY":
            distorted_pose[param] = pose[param] + random.randint(-25, 25)
        elif param == "TransZ":
            distorted_pose[param] = pose[param] + random.randint(-65, 65)

    return distorted_pose


def crop_and_resize(array, bbox_corner):
    image_size = 224
    x_min, y_min, x_max, y_max = bbox_corner

    cropped = array[y_min:y_max, x_min:x_max]
    try:
        resized = cv2.resize(cropped, (image_size, image_size))
        return resized
    except Exception as e:
        print(e)
        print(x_min, x_max, y_min, y_max)
        exit()

def get_centered_bbox(trans_matrix, bbox_rendered, bbox_image):
    projected_center = dataset_renderer.project_point_cloud(np.array([[0, 0, 0]]), trans_matrix)
    x_c_projected, y_c_projected = projected_center[0][0], projected_center[1][0]
    x1_min, y1_min, x1_max, y1_max = bbox_image
    x2_min, y2_min, x2_max, y2_max = bbox_rendered

    size = max(x1_max - x1_min, y1_max - y1_min, x2_max - x2_min, y2_max - y2_min) * 1.2
    x_min = x_c_projected - int(size//2)
    y_min = y_c_projected - int(size//2)
    x_max = x_c_projected + int(size//2)
    y_max = y_c_projected + int(size//2)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def generate_refinement_image(subset, index):
    path = MAIN_DIR_PATH + "Dataset/" + subset + "/"
    json_data = io.load_json_file(path + "Pose/" + "data_{}.json".format(index))
    
    pose_real = json_data["Pose"]
    bbox_real = json_data["Box"]
    
    path_datapoint = path + "ImageRefinement/" + "Data_{}/".format(index)
    os.mkdir(path_datapoint)
    os.mkdir(path_datapoint + "Image")
    os.mkdir(path_datapoint + "Pose")
    print("Real Image num {} has pose {}".format(index, pose_real))
    for i in range(NUM_RENDERED_POSES_PER_IMAGE):
        pose_rendered = distort_target_pose(pose_real)
        trans_matrix = transformations.get_transformation_matrix_from_pose(pose_rendered)
        
        rendered_image_dict = dataset_renderer.get_image(trans_matrix, pose_rendered, image_black=True, image_background=False, UVW=False, constant_light=True)
        
        rendered_image = rendered_image_dict["ImageBlack"] * np.expand_dims(rendered_image_dict["Mask"], axis=-1)
        
        bbox_rendered = rendered_image_dict["Box"]
        
        bbox_crop = get_centered_bbox(trans_matrix, bbox_rendered, bbox_real)
        
        rendered_image_cropped = crop_and_resize(rendered_image, bbox_crop)
        
        data_json = {"Pose": pose_rendered, "Box": bbox_crop}
        
        io.save_numpy_file(path_datapoint + "Image/" + "data_{}.npy".format(i), rendered_image_cropped)
        io.save_json_file(path_datapoint + "Pose/" + "data_{}.json".format(i), data_json)
        print("Rendered image number {} with pose {} was genedered".format(i, pose_rendered))
        

def generate_refinements_dataset():
    for index in range(0, 256):
        generate_refinement_image(subset='Training', index=index)
    """or index in range(0, 5000):
        generate_refinement_image(subset='Validation', index=index)"""


if __name__ == "__main__":
    generate_refinements_dataset()
        
        
        
        