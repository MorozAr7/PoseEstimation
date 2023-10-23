from CONFIG import *

sys.path.insert(0, MAIN_DIR_PATH)
import torch
import random
import sys
import numpy as np
from CoarsePoseEstimation.Evaluation import CoarsePoseEvaluation
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
        self.coarse_pose_estimator = CoarsePoseEvaluation("cpu")
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

    @staticmethod
    def get_square_bbox(bbox):
        centroid_x, centroid_y, width, height = bbox
        size_side = max(width, height)

        return centroid_x, centroid_y, size_side, size_side

    @staticmethod
    def convert_corner_to_centroid(bbox):
        x_min, y_min, x_max, y_max = bbox
        centroid_x = (x_max + x_min) // 2
        centroid_y = (y_max + y_min) // 2
        width = x_max - x_min
        height = y_max - y_min

        return centroid_x, centroid_y, width, height

    @staticmethod
    def convert_centroid_to_corner(bbox):
        centroid_x, centroid_y, width, height = bbox

        x_min = int(centroid_x - width // 2)
        x_max = int(centroid_x + width // 2)
        y_min = int(centroid_y - height // 2)
        y_max = int(centroid_y + height // 2)

        return x_min, y_min, x_max, y_max

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

        bbox_tight = json_data["TightBox"]
        bbox_enlarged = json_data["EnlargedBox"]
        x_min_, y_min_, x_max_, y_max_ = bbox_enlarged
        print(bbox_tight, bbox_enlarged)
        bbox_corner = [bbox_tight[0] - x_min_, bbox_tight[1] - y_min_, bbox_tight[2] - x_min_, bbox_tight[3] - y_min_]

        bbox_centroid = self.convert_corner_to_centroid(bbox_corner)
        box_square = self.get_square_bbox(bbox_centroid)
        bbox_corner = self.convert_centroid_to_corner(box_square)
        trans_matrix_real = self.transformations.get_transformation_matrix_from_pose(real_pose)


        real_image_cropped = self.crop_and_resize(real_image, bbox_corner)

        return real_image, real_image_cropped, trans_matrix_real, bbox_corner

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
        updated_xy = (predicted_t[0:2] + coarse_xy / coarse_z) * updated_z
        T_predicted = coarse_T.copy()
        T_predicted[0:3, 0:3] = predicted_R @ coarse_R
        T_predicted[:2, -1] = updated_xy
        T_predicted[2:3, -1] = updated_z
        return T_predicted

    def evaluate_dataset(self):

        for index in range(32):
            real_image, real_image_cropped, trans_matrix_real, box_square = self.load_image(index)

            T_coarse = self.coarse_pose_estimator.get_coarse_pose_estimate(np.expand_dims(real_image_cropped, 0), None, np.expand_dims(np.array(box_square), 0))[0]

            coarse_estimate_dict = self.renderer.render_image("Chassis", T_coarse, None, False, constant_light=True)
            coarse_image_estimate = coarse_estimate_dict["ImageBlack"]
            mask = coarse_estimate_dict["Mask"]

            cropped_coarse = self.crop_and_resize(coarse_image_estimate, box_square, True)
            cv2.imshow("coarse estimate", np.concatenate([real_image_cropped, cropped_coarse], axis=1))
            cv2.waitKey(0)
            cropped_mask = self.crop_and_resize(np.expand_dims(mask, -1).astype(float), bbox, True)
            cropped_mask = np.expand_dims(cropped_mask, -1)
            visualize_coarse = real_image_cropped * (1 - cropped_mask) + rendered_image_cropped
            visualize_refined = real_image_cropped * (1 - cropped_mask) + cropped_black
            cv2.imshow("coarse and refined", np.concatenate([visualize_coarse, visualize_refined], axis=1) / 255)
            cv2.waitKey(0)
            real_image_torch = NormalizeToTensor(image=real_image_cropped)["image"].unsqueeze(0).to(self.device)
            rendered_image_torch = NormalizeToTensor(image=rendered_image_cropped)["image"].unsqueeze(0).to(self.device)

            trans_pred, rot_pred = self.process_cnn(real_image_torch, rendered_image_torch)
            T_predicted = self.update_pose_prediction(trans_matrix_rendered, rot_pred, trans_pred)
            print(cropped_black.shape, cropped_mask.shape)
            print(real_image.shape, real_image_cropped.shape, rendered_image_cropped.shape)
            # exit()


if __name__ == "__main__":
    evaluation = EvaluateOnDataset()
    evaluation.evaluate_dataset()
