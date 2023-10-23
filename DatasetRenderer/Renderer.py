import os
import sys

CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = ""
for i in CURR_DIR_PATH.split("/")[:-1]:
    MAIN_DIR_PATH += i + "/"

sys.path.insert(0, MAIN_DIR_PATH)
sys.path.insert(1, CURR_DIR_PATH)

DATASET_PATH = MAIN_DIR_PATH + "Dataset/"

import open3d as o3d
from DatasetRenderer.RendererConfig import *
import numpy as np
import random
import cv2
import sys
import json

from Utils.MathUtils import Transformations
from Utils.IOUtils import IOUtils

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))


class DatasetRenderer:
    def __init__(self):
        self.transformations = Transformations()
        self.io = IOUtils()
        self.render_config = self.io.load_json_file(MAIN_DIR_PATH + "DatasetRenderer/ConfigFile.json")

        self.camera_file = "camera_data_1.json"
        self.models_data_path = "DatasetRenderer/Models3D/MeshesReconstructed/"
        self.object_types = self.render_config["ObjectTypes"]

        self.camera_file = self.io.load_json_file(MAIN_DIR_PATH + "/CameraData/" + self.camera_file)
        self.camera_intrinsic = np.array(self.camera_file["K"])

        self.image_h, self.image_w = self.camera_file["resolution_undistorted"]
        self.data_dict = self.load_data()

        self.pose_ranges = self.render_config["PoseRanges"]
        self.noise_threshold = 2

        self.backgrounds_path = MAIN_DIR_PATH + "/DatasetRenderer/Backgrounds/"
        self.backgrounds_list = os.listdir(self.backgrounds_path)

    def sample_point_clouds(self, num_points, object_type):
        file = MAIN_DIR_PATH + self.models_data_path + object_type + "/MeshEdited.obj"
        mesh = o3d.io.read_triangle_mesh(file)
        pcd = mesh.sample_points_poisson_disk(num_points)#mesh.sample_points_uniformly(num_points)
        pcd = np.array(pcd.points)

        if num_points == self.render_config["DensePointCloudSize"]:
            self.io.save_numpy_file(MAIN_DIR_PATH + self.models_data_path + f"{object_type}/DensePointCloud.npy", pcd)
        elif num_points == 10000:
            self.io.save_numpy_file(MAIN_DIR_PATH + self.models_data_path + f"{object_type}/SparsePointCloud.npy", pcd)

    def load_data(self):
        data_dict = {}
        for obj_type in self.object_types:
            triangle_model = o3d.io.read_triangle_model(MAIN_DIR_PATH + self.models_data_path + f"{obj_type}/MeshEdited.obj")
            point_cloud = self.io.load_numpy_file(MAIN_DIR_PATH + self.models_data_path + f"{obj_type}/DensePointCloud.npy")
            mapping = self.create_uvw_mapping(obj_type, point_cloud)#self.io.load_numpy_file(MAIN_DIR_PATH + self.models_data_path + f"{obj_type}/DenseMapping.npy")
            data_dict[obj_type] = {"model": triangle_model, "point_cloud": point_cloud, "mapping": mapping}
        return data_dict

    @staticmethod
    def get_homogenous_coords(cartesian_coords: np.array) -> np.array:
        ones = np.ones(shape=(cartesian_coords.shape[0], 1))
        return np.concatenate([cartesian_coords, ones], axis=-1)

    def sample_pose(self) -> dict:
        pose_params = {}
        for key, value in self.pose_ranges.items():
            pose_params[key] = random.randint(value[0], value[1])

        return pose_params

    def get_object_mask(self, rendered_image: np.array) -> np.array:
        mask = rendered_image > self.noise_threshold
        return np.array((mask[..., 0] * mask[..., 1] * mask[..., 2]), dtype=bool)

    @staticmethod
    def get_bbox_from_mask(mask: np.array) -> tuple:
        mask_positive_pixels = np.where(mask == 1)
        x_min = np.min(mask_positive_pixels[1])
        x_max = np.max(mask_positive_pixels[1])
        y_min = np.min(mask_positive_pixels[0])
        y_max = np.max(mask_positive_pixels[0])

        return int(x_min), int(y_min), int(x_max), int(y_max)

    @staticmethod
    def get_min_max_scaling(coords: np.array) -> np.array:
        min_val = np.min(coords)
        max_val = np.max(coords)

        scaled_coords = (coords - min_val) / (max_val - min_val)

        return scaled_coords

    def create_uvw_mapping(self, object_type, coords3d: np.array) -> tuple:
        centroid = np.mean(coords3d, axis=0)
        x_centered = coords3d[..., 0] - centroid[0]
        y_centered = coords3d[..., 1] - centroid[1]
        z_centered = coords3d[..., 2] - centroid[2]

        x_scaled = self.get_min_max_scaling(x_centered)
        y_scaled = self.get_min_max_scaling(y_centered)
        z_scaled = self.get_min_max_scaling(z_centered)
        #print(np.max(z_scaled), np.min(z_scaled))
        u = np.floor(x_scaled * (self.render_config["MappingRange"] - 1) - (self.render_config["MappingRange"] - 1)//2)
        v = np.floor(y_scaled * (self.render_config["MappingRange"] - 1) - (self.render_config["MappingRange"] - 1)//2)
        w = np.floor(z_scaled * (self.render_config["MappingRange"] - 1) - (self.render_config["MappingRange"] - 1)//2)
        #print(np.max(u), np.min(u), np.max(v), np.min(v),np.max(w), np.min(w))
        """uvw_mapping_dict = dict()
        file = open(MAIN_DIR_PATH + self.models_data_path + f"{object_type}/Mapping.json", "w")

        for index in range(coords3d.shape[0]):
            uvw_mapping_dict[str((int(u[index]), int(v[index]), int(w[index])))] = coords3d[index, :].reshape(-1).tolist()
        json.dump(uvw_mapping_dict, file)"""
        return u, v, w

    def project_point_cloud(self, coords3d: np.array, transformation_matrix: np.array) -> tuple:
        homogenous_coords_3d = self.get_homogenous_coords(coords3d)

        homogenous_coords_2d = self.camera_intrinsic @ (transformation_matrix @ homogenous_coords_3d.T)
        homogenous_coords_2d[2, :][np.where(homogenous_coords_2d[2, :] == 0)] = 1

        coords_2d = homogenous_coords_2d[:2, ...] / homogenous_coords_2d[2, ...]
        pixel_coords = (np.floor(coords_2d)).T.astype(int)

        x_2d = np.clip(pixel_coords[:, 0], 0, self.image_w - 1)
        y_2d = np.clip(pixel_coords[:, 1], 0, self.image_h - 1)

        return x_2d, y_2d

    def get_rendered_uvw_maps(self, x_2d: np.array, y_2d: np.array, uvw_maps: tuple) -> np.array:
        u_map = np.zeros((self.image_h, self.image_w, 1))
        v_map = np.zeros((self.image_h, self.image_w, 1))
        w_map = np.zeros((self.image_h, self.image_w, 1))

        u_map[y_2d, x_2d] = uvw_maps[0].reshape(-1, 1)
        v_map[y_2d, x_2d] = uvw_maps[1].reshape(-1, 1)
        w_map[y_2d, x_2d] = uvw_maps[2].reshape(-1, 1)

        return np.concatenate([u_map.astype(int), v_map.astype(int), w_map.astype(int)], axis=-1)

    @staticmethod
    def setup_renderer_scene(renderer, direction: list, color: list, intensity: int):
        renderer.scene.scene.set_sun_light(direction, color, intensity)
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.enable_light_shadow("directional", True)

        return renderer

    def load_random_background(self) -> o3d.geometry.Image:
        index = random.randint(0, len(self.backgrounds_list) - 1)
        background = cv2.imread(self.backgrounds_path + self.backgrounds_list[index])
        background = o3d.geometry.Image(background)

        return background

    def get_enlarged_bbox(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        bbox_size_amplification = self.render_config["BboxSizeAmplification"]
        bbox_max_size = max(y_max - y_min, x_max - x_min)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        bbox_max_size_amplified = (1 + bbox_size_amplification) * bbox_max_size

        x_min_ = int(center_x - bbox_max_size_amplified / 2)
        x_max_ = int(center_x + bbox_max_size_amplified / 2)
        y_min_ = int(center_y - bbox_max_size_amplified / 2)
        y_max_ = int(center_y + bbox_max_size_amplified / 2)

        return [x_min_, y_min_, x_max_, y_max_]
    @staticmethod
    def randomize_light_conditions(constant=False):
        if constant:
            direction = [0, 0, 0]
            intensity = 1000
            color = [255, 255, 255]
        else:
            direction = [0, 0, 0]
            intensity = random.randint(500, 2500)
            r = random.randint(100, 205)
            g = r + random.randint(-50, 50)
            b = r + random.randint(-50, 50)
            color = [r, g, b]

        return direction, color, intensity

    def crop_images(self, image, image_black, mask, uvw_map, bbox):
        x_min, y_min, x_max, y_max = bbox

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        cropped_uvw_map = uvw_map[y_min:y_max, x_min:x_max]
        cropped_black = image_black[y_min:y_max, x_min:x_max]
        return cropped_image, cropped_black, cropped_mask, cropped_uvw_map

    def render_image(self, object_type, T_matrix, pose6d, crop=True, constant_light=False) -> dict:

        model = self.data_dict[object_type]["model"]
        point_cloud = self.data_dict[object_type]["point_cloud"]
        mapping = self.data_dict[object_type]["mapping"]
        print("Before renderer init")
        renderer = o3d.visualization.rendering.OffscreenRenderer(self.image_w, self.image_h)
        renderer.scene.add_model(object_type, model)
        print("After renderer init")
        direction, color, intensity = self.randomize_light_conditions(constant_light)

        renderer = self.setup_renderer_scene(renderer, direction, color, intensity)
        renderer.setup_camera(self.camera_intrinsic[0:3, 0:3], T_matrix, self.image_w, self.image_h)

        print("Before load background")
        background_image = self.load_random_background()
        renderer.scene.set_background(np.array([0, 0, 0, 1]), image=background_image)
        image_background = np.array(renderer.render_to_image())
        print("After load background")
        renderer.scene.set_background(np.array([0, 0, 0, 1]))
        image_black = np.array(renderer.render_to_image())
        x_2d, y_2d = self.project_point_cloud(point_cloud, T_matrix)
        print("After black image is rendered")
        uvw_map = self.get_rendered_uvw_maps(x_2d, y_2d, mapping)
        mask = self.get_object_mask(image_black)
        tight_bbox = self.get_bbox_from_mask(mask)
        enlarged_bbox = self.get_enlarged_bbox(tight_bbox)
        print("After box is computed")
        if crop:
            image_background, image_black, mask, uvw_map = self.crop_images(image_background, image_black, mask, uvw_map, enlarged_bbox)

        rendered_image_dict = {"ImageBackground": image_background,
                               "ImageBlack": image_black,
                               "Mask": mask,
                               "UVWmap": uvw_map * np.expand_dims(mask, axis=-1),
                               "TightBox": tight_bbox,
                               "EnlargedBox": enlarged_bbox,
                               "Pose": pose6d,
                               "Class": object_type
                               }
        # del renderer
        print("After renderer is deleted")
        return rendered_image_dict

    @staticmethod
    def crop_and_resize(full_scale, bbox):
        x_min, y_min, x_max, y_max = bbox
        return cv2.resize(full_scale[y_min: y_max, x_min: x_max], (IMG_SIZE, IMG_SIZE))

    def save_data(self, img_object, index, subset):
        json_data = {}
        for key, data in img_object.items():
            if key in ["TightBox", "EnlargedBox", "Pose", "Class"]:
                json_data[key] = data
            elif key == "ImageBackground":
                path = DATASET_PATH + subset + "/" + key + "/img_{}.png".format(index)
                cv2.imwrite(path, data)
            elif key == "ImageBlack":
                pass
            else:
                path = DATASET_PATH + subset + "/" + key + "/data_{}.np".format(index)
                self.io.save_numpy_file(path, data)
        path = DATASET_PATH + subset + "/Label" + "/data_{}.json".format(index)
        self.io.save_json_file(path, json_data)

    def render_dataset(self):
        for subset in ["Training", "Validation"]:
            for data_index in range(0, self.render_config["DataAmount"][subset]):
                print("Ready to generate object")
                integer_object_type = random.randint(0, len(self.object_types) - 1)
                object_type = self.object_types[integer_object_type]
                pose6d = self.sample_pose()
                T_matrix = self.transformations.get_transformation_matrix_from_pose(pose6d)
                rendered_image_dict = self.render_image(object_type, T_matrix=T_matrix, pose6d=pose6d)
                self.save_data(rendered_image_dict, data_index, subset)
                print("{} image number {}/{} has been rendered".format(subset, data_index, self.render_config["DataAmount"][subset]))


if __name__ == "__main__":
    dataset_renderer = DatasetRenderer()

    dataset_renderer.render_dataset()
