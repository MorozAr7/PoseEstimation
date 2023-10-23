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
        self.device = "cpu"
        self.init_cnn()
        self.subset = "Training"
        self.transformations = Transformations()
        self.renderer = DatasetRenderer()
        self.io = IOUtils()
        self.image_size = 224
    
    def init_cnn(self):
        self.pose_ref_model.load_state_dict(torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/PoseEstimation/RefinedPoseEstimation/TrainedModels/RefinedPoseEstimationBigModel.pt", map_location="cpu"))

        self.pose_ref_model.to(self.device)
        self.pose_ref_model.eval()
    
    def crop_and_resize(self, array, bbox_corner, resize=True):
        x_min, y_min, x_max, y_max = bbox_corner
        print(x_min, y_min, x_max, y_max)
        print(array.shape)
        cropped = array[y_min:y_max, x_min:x_max, :]
        print(cropped.shape)
        if resize:
            try:
                resized = cv2.resize(cropped, (self.image_size, self.image_size))
                return resized
            except Exception as e:
                print(e)
                print(x_min, x_max, y_min, y_max)
                exit()
        else:
            return cropped


    
    def load_image(self, index):
        path = MAIN_DIR_PATH + "/Dataset/" + self.subset + "/"
        self.index = index
        
        real_image = cv2.imread(path + "ImageBackground/" + "img_{}.png".format(index))
        json_data = self.io.load_json_file(path + "Label/" + "data_{}.json".format(index))
        
        real_pose = json_data["Pose"]
        bbox_real_image_enlarged = json_data["EnlargedBox"]

        refinement_pose_number = random.randint(0, 9)
        path_datapoint = path + "ImageRefinement/" + "Data_{}/".format(index)
        refinement_image_path = path_datapoint + "Image/" + "data_{}.png".format(refinement_pose_number)
        refinement_data_path = path_datapoint + "Pose/" + "data_{}.json".format(refinement_pose_number)

        rendered_image_cropped = cv2.imread(refinement_image_path)
        rendered_data = self.io.load_json_file(refinement_data_path)
        rendered_pose = rendered_data["Pose"]
        bbox = rendered_data["Box"]

        bbox_crop = [bbox[0] - bbox_real_image_enlarged[0], bbox[1] - bbox_real_image_enlarged[1], bbox[2] - bbox_real_image_enlarged[0], bbox[3] - bbox_real_image_enlarged[1]]
        trans_matrix_real = self.transformations.get_transformation_matrix_from_pose(real_pose)
        trans_matrix_rendered = self.transformations.get_transformation_matrix_from_pose(rendered_pose)

        real_image_cropped = self.crop_and_resize(real_image, bbox_crop)
        real_image = self.crop_and_resize(real_image, bbox_crop, False)

        return real_image, real_image_cropped, rendered_image_cropped, trans_matrix_real, trans_matrix_rendered, bbox_crop, bbox

    def process_cnn(self, real_image_torch, rendered_image_torch):
        
        trans_pred, rot_pred = self.pose_ref_model(real_image_torch, rendered_image_torch)
        
        trans_pred_numpy = trans_pred.detach().cpu().numpy()[0]
        rot_pred_numpy = rot_pred.detach().cpu().numpy()[0]
        
        return trans_pred_numpy, rot_pred_numpy

    def update_pose_prediction(self, coarse_T, predicted_R, predicted_t):

        coarse_xy = coarse_T[:2, -1]
        coarse_R = coarse_T[:3, :3]
        coarse_z = coarse_T[2:3, -1]
        updated_z = predicted_t[2:3] * coarse_z
        updated_xy = (predicted_t[0:2] + coarse_xy/coarse_z) * updated_z
        T_predicted = coarse_T.copy()
        T_predicted[0:3, 0:3] = predicted_R @ coarse_R
        T_predicted[:2, -1] = updated_xy
        T_predicted[2:3, -1] = updated_z
        return T_predicted

    def evaluate_dataset(self):
        
        for index in range(32):
            real_image, real_image_cropped, rendered_image_cropped, trans_matrix_real, trans_matrix_rendered, rendered_bbox, bbox = self.load_image(index)

            real_image_torch = NormalizeToTensor(image=real_image_cropped)["image"].unsqueeze(0).to(self.device)
            rendered_image_torch = NormalizeToTensor(image=rendered_image_cropped)["image"].unsqueeze(0).to(self.device)

            trans_pred, rot_pred = self.process_cnn(real_image_torch, rendered_image_torch)
            
            T_predicted = self.update_pose_prediction(trans_matrix_rendered, rot_pred, trans_pred)
            rendered_image_dict = self.renderer.render_image("Chassis", T_predicted, None, False, constant_light=True)
            image_black = rendered_image_dict["ImageBlack"]
            mask = rendered_image_dict["Mask"]
            cropped_black = self.crop_and_resize(image_black, bbox, True)
            cropped_mask = self.crop_and_resize(np.expand_dims(mask, -1).astype(float), bbox, True)
            cropped_mask = np.expand_dims(cropped_mask, -1)
            visualize_coarse = real_image_cropped * (1 - cropped_mask) + rendered_image_cropped
            visualize_refined = real_image_cropped * (1 - cropped_mask) + cropped_black
            cv2.imshow("coarse and refined", np.concatenate([visualize_coarse, visualize_refined], axis=1)/255)
            cv2.waitKey(0)

            print(cropped_black.shape, cropped_mask.shape)
            print(real_image.shape, real_image_cropped.shape, rendered_image_cropped.shape)
            #exit()


if __name__ == "__main__":
    evaluation = EvaluateOnDataset()
    evaluation.evaluate_dataset()
            
            
            
            
        
        
        