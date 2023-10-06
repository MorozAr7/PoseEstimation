from CONFIG import *

sys.path.insert(0, MAIN_DIR_PATH)
import random
import numpy as np
import cv2
import torch
from Utils.IOUtils import IOUtils
from Utils.DataAugmentationUtils import NormalizeToTensor
from Utils.MathUtils import Transformations


class Dataset(torch.utils.data.Dataset):
    def __init__(self, subset, num_images, dataset_renderer, data_augmentation=None) -> None:
        super(Dataset, self).__init__()
        self.data_augmentation = data_augmentation
        self.subset = subset
        self.dataset_len = num_images
        self.image_size = 224
        self.full_scale_w = 1865
        self.full_scale_h = 1039
        self.io = IOUtils()
        self.dataset_renderer = dataset_renderer
        self.index = None
        self.transformations = Transformations()

    def __len__(self):
        return self.dataset_len

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

    def random_shift(self, square_bbox, shift_limits):
        pass

    def is_inside_image(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        results_dict = {"x_min": x_min,
                        "y_min": y_min,
                        "x_max": self.full_scale_w + 1 - x_max,
                        "y_max": self.full_scale_h + 1 - y_max}
        return results_dict

    @staticmethod
    def shift_bbox(square_bbox, shift_limits):
        shift_x = random.randint(-shift_limits["left"], shift_limits["right"])
        shift_y = random.randint(-shift_limits["up"], shift_limits["down"])
        print(shift_x, shift_y)
        x_min, y_min, x_max, y_max = square_bbox
        return x_min + shift_x, y_min + shift_y, x_max + shift_x, y_max + shift_y

    @staticmethod
    def get_shift_limits(bbox, check_results):
        size = bbox[2] - bbox[0]
        size_20_percent = 0  # size // 5
        limits = {"left": None, "right": None, "up": None, "down": None}
        for key, val in check_results.items():
            if key == "x_min":
                limits["left"] = min(val, size_20_percent)
            elif key == "x_max":
                limits["right"] = min(val, size_20_percent)
            elif key == "y_min":
                limits["up"] = min(val, size_20_percent)
            elif key == "y_max":
                limits["down"] = min(val, size_20_percent)
        # print(bbox, check_results, limits)
        return limits

    @staticmethod
    def get_square_bbox(bbox):
        x_min, y_min, x_max, y_max = bbox
        size_x = x_max - x_min
        size_y = y_max - y_min
        difference = abs(size_x - size_y)
        if size_x > size_y:
            y_min = y_min - difference // 2
            y_max = y_max + difference // 2
        else:
            x_min = x_min - difference // 2
            x_max = x_max + difference // 2
        return x_min, y_min, x_max, y_max

    def get_bbox(self, bbox):
        square_bbox = self.get_square_bbox(bbox)
        check_results = self.is_inside_image(square_bbox)
        limits = self.get_shift_limits(square_bbox, check_results)
        shifted_bbox = self.shift_bbox(square_bbox, limits)
        return shifted_bbox

    def get_bbox_from_mask(self, mask: np.array) -> tuple:
        mask_positive_pixels = np.where(mask == 1)
        x_min = np.min(mask_positive_pixels[1])
        x_max = np.max(mask_positive_pixels[1])
        y_min = np.min(mask_positive_pixels[0])
        y_max = np.max(mask_positive_pixels[0])

        return x_min, y_min, x_max, y_max

    def get_centered_bbox(self, trans_matrix, bbox_rendered, bbox_image):
        projected_center = self.dataset_renderer.project_point_cloud(np.array([[0, 0, 0]]), trans_matrix)
        x_c_projected, y_c_projected = projected_center[0][0], projected_center[1][0]
        x1_min, y1_min, x1_max, y1_max = bbox_image
        x2_min, y2_min, x2_max, y2_max = bbox_rendered

        size = max(x1_max - x1_min, y1_max - y1_min, x2_max - x2_min, y2_max - y2_min) * 1.2
        x_min = x_c_projected - int(size // 2)
        y_min = y_c_projected - int(size // 2)
        x_max = x_c_projected + int(size // 2)
        y_max = y_c_projected + int(size // 2)
        return x_min, y_min, x_max, y_max

    def __getitem__(self, index):
        try:
            path = MAIN_DIR_PATH + "Dataset/" + self.subset + "/"
            self.index = index
            real_image = self.io.load_numpy_file(path + "ImageBackground/" + "data_{}.np".format(index))
            mask = self.io.load_numpy_file(path + "Mask/" + "data_{}.np".format(index))
            json_data = self.io.load_json_file(path + "Pose/" + "data_{}.json".format(index))

            real_pose = json_data["Pose"]

            refinement_pose_number = random.randint(0, 9)
            if self.subset == "Validation":
                refinement_pose_number = 0
            path_datapoint = path + "ImageRefinement/" + "Data_{}/".format(index)
            refinement_image_path = path_datapoint + "Image/" + "data_{}.npy".format(refinement_pose_number)
            refinement_data_path = path_datapoint + "Pose/" + "data_{}.json".format(refinement_pose_number)

            rendered_image = self.io.load_numpy_file(refinement_image_path)
            rendered_data = self.io.load_json_file(refinement_data_path)
            rendered_pose = rendered_data["Pose"]
            rendered_bbox = rendered_data["Box"]
            trans_matrix_real = self.transformations.get_transformation_matrix_from_pose(real_pose)
            trans_matrix_rendered = self.transformations.get_transformation_matrix_from_pose(rendered_pose)
            optical_flow_map = self.dataset_renderer.get_optical_flow(trans_matrix_real, trans_matrix_rendered).reshape(1039, 1865, 3)

            real_image = self.crop_and_resize(real_image, rendered_bbox)
            reshaped_flow = self.crop_and_resize(optical_flow_map, rendered_bbox)
            #p#rint(mask.dtype)
            mask = self.crop_and_resize(mask.astype(float), rendered_bbox)
            if self.data_augmentation:
                real_image = self.data_augmentation(image=real_image)["image"]
            #cv2.imshow("mask", mask)
            #cv2.imshow("flow", np.concatenate([reshaped_flow[..., 0:3], real_image/255, rendered_image/255], axis=0))

            #cv2.waitKey(0)
            image_tensor = NormalizeToTensor(image=real_image)["image"]
            rendered_image_tensor = NormalizeToTensor(image=rendered_image)["image"]

            return image_tensor, rendered_image_tensor, \
                   torch.tensor(reshaped_flow, dtype=torch.float32).permute(2, 0, 1), \
                    torch.tensor(mask, dtype=torch.float32)
        except Exception as e:
            print(e)
            print("IMAGE ERROR IS: ", index)
            self.__getitem__(index)
