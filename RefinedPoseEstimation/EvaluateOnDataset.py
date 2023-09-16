from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import torch
import random
import sys
import numpy as np
from CnnModel import PoseRefinementNetwork
from Utils.DataAugmentationUtils import PoseEstimationAugmentation, NormalizeToTensor
from Utils.IOUtils import IOUtils
from Utils.ConvUtils import init_weights, change_learning_rate
from Utils.MathUtils import Transformations
from RefinedPoseEstimation.LossFunction import ProjectionLoss
import torch.nn as nn
import warnings
import time
import cv2
from DatasetRenderer.Renderer import DatasetRenderer
warnings.filterwarnings("ignore")


class EvaluateOnDataset:
    def __init__(self):
        self.pose_ref_model = PoseRefinementNetwork()
        self.init_cnn()
        self.subset = "Training"
        self.transformations = Transformations()
        self.renderer = DatasetRenderer()
        self.io = IOUtils()
        self.image_size = 224
    
    def init_cnn(self):
        self.pose_ref_model.load_state_dict(torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/PoseEstimation/RefinedPoseEstimation/TrainedModels/RefinedPoseEstimationModelProjection2DfasterLRreduction.pt", map_location="cpu"))
        
        self.pose_ref_model.to("mps")
        self.pose_ref_model.eval()
    
    def crop_and_resize(self, array, bbox_corner):
        x_min, y_min, x_max, y_max = bbox_corner

        cropped = array[y_min:y_max, x_min:x_max]
        try:
            resized = cv2.resize(cropped, (self.image_size, self.image_size))
            return resized
        except Exception as e:
            print(e)
            print(x_min, x_max, y_min, y_max)
            exit()
    
    def load_image(self, index):
        path = MAIN_DIR_PATH + "/Dataset/" + self.subset + "/"
        self.index = index
        
        real_image = self.io.load_numpy_file(path + "ImageBackground/" + "data_{}.np".format(index))

        json_data = self.io.load_json_file(path + "Pose/" + "data_{}.json".format(index))
        
        real_pose = json_data["Pose"]

        refinement_pose_number = random.randint(0, 9)
        path_datapoint = path + "ImageRefinement/" + "Data_{}/".format(index)
        refinement_image_path = path_datapoint + "Image/" + "data_{}.npy".format(refinement_pose_number)
        refinement_data_path = path_datapoint + "Pose/" + "data_{}.json".format(refinement_pose_number)

        rendered_image_cropped = self.io.load_numpy_file(refinement_image_path)
        rendered_data = self.io.load_json_file(refinement_data_path)
        rendered_pose = rendered_data["Pose"]
        rendered_bbox = rendered_data["Box"]
        trans_matrix_real = self.transformations.get_transformation_matrix_from_pose(real_pose)
        trans_matrix_rendered = self.transformations.get_transformation_matrix_from_pose(rendered_pose)
        
        real_image_cropped = self.crop_and_resize(real_image, rendered_bbox)
        
        return real_image, real_image_cropped, rendered_image_cropped, trans_matrix_real, trans_matrix_rendered, rendered_bbox

    def process_cnn(self, real_image_torch, rendered_image_torch):
        
        trans_pred, rot_pred = self.pose_ref_model(real_image_torch, rendered_image_torch)
        
        trans_pred_numpy = trans_pred.detach().cpu().numpy()[0]
        rot_pred_numpy = rot_pred.detach().cpu().numpy()[0]
        
        return trans_pred_numpy, rot_pred_numpy
    
    
    def update_pose_prediction(self, coarse_T, predicted_R, predicted_t):
        print(predicted_t)
        print(predicted_R)
        print(coarse_T)
        coarse_xy = coarse_T[:2, -1]
        coarse_R = coarse_T[:3, 0:3]
        coarse_z = coarse_T[2:3, -1]
        updated_z = predicted_t[2:3] * coarse_z
        updated_xy = (predicted_t[0:2] + coarse_xy/coarse_z) * updated_z
        T_predicted = coarse_T.copy()
        T_predicted[0:3, 0:3] = predicted_R @ coarse_R
        T_predicted[:2, -1] = updated_xy
        T_predicted[2:3, -1] = updated_z 
        print(T_predicted)
        return T_predicted
        
        
    def evaluate_dataset(self):
        
        for index in range(256):
            real_image, real_image_cropped, rendered_image_cropped, trans_matrix_real, trans_matrix_rendered, rendered_bbox = self.load_image(index)
            #cv2.imshow("img", np.hstack([real_image_cropped, rendered_image_cropped]))
            #cv2.waitKey(0)
            real_image_torch = NormalizeToTensor(image=real_image_cropped)["image"].unsqueeze(0).to("mps")
            rendered_image_torch = NormalizeToTensor(image=rendered_image_cropped)["image"].unsqueeze(0).to("mps")
            
            trans_pred, rot_pred = self.process_cnn(real_image_torch, rendered_image_torch)
            
            T_predicted = self.update_pose_prediction(trans_matrix_rendered, rot_pred, trans_pred)
            rendered_image_dict = self.renderer.get_image(trans_matrix_rendered, None, image_black=True, image_background=False, UVW=False, constant_light=True)
            mask = np.expand_dims(rendered_image_dict["Mask"], axis=-1)
            rendered_image_coarse = rendered_image_dict["ImageBlack"] * mask
            visualize_coarse = real_image * (1 - mask) + rendered_image_coarse
            visualize_coarse = self.crop_and_resize(visualize_coarse.astype(np.uint8), rendered_bbox)
            rendered_image_dict = self.renderer.get_image(T_predicted, None, image_black=True, image_background=False, UVW=False, constant_light=True)
            mask = np.expand_dims(rendered_image_dict["Mask"], axis=-1)
            rendered_image_refined = rendered_image_dict["ImageBlack"] * mask
            
            visualize_refined = real_image * (1 - mask) + rendered_image_refined
            visualize_refined = self.crop_and_resize(visualize_refined.astype(np.uint8), rendered_bbox)
            cv2.imshow("video", np.vstack([visualize_refined, visualize_coarse])/255)
            cv2.waitKey(0)


if __name__ == "__main__":
    evaluation = EvaluateOnDataset()
    evaluation.evaluate_dataset()
            
            
            
            
        
        
        