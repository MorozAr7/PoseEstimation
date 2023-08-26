import os
import open3d as o3d
import json
from RendererConfig import *
import numpy as np
import random
import cv2
import sys
sys.path.insert(1, ROOT_PATH + '/PoseEstimation/CameraData')
sys.path.insert(1, ROOT_PATH + '/PoseEstimation/Utils')
from Utils.MathUtils import Transformations
from Utils.IOUtils import IOUtils
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))


class RenderedImage:
	def __init__(self,
	             full_scale_black=None,
	             full_scale_background=None,
	             mask=None,
	             u_map=None,
	             v_map=None,
	             w_map=None,
	             bbox=None,
	             pose_6d=None):

		self.full_scale_black = full_scale_black
		self.full_scale_background = full_scale_background
		self.mask = mask
		self.u_map = u_map
		self.v_map = v_map
		self.w_map = w_map
		self.bbox = bbox
		self.pose_6d = pose_6d


class DatasetRenderer:
	def __init__(self):
		self.transformations = Transformations()
		self.io = IOUtils()

		self.camera_data = self.io.load_json_file(ROOT_PATH + "/CameraData/" + CAM_DATA_FILE)
		self.camera_intrinsic = np.array(self.camera_data["K"])

		self.image_h, self.image_w = self.camera_data["res_undist"]
		self.point_cloud = self.io.load_numpy_file("./Models3D/" + OBJECT_TYPE + "/PointCloud.npy")
		self.model3d = o3d.io.read_triangle_model("./Models3D/" + OBJECT_TYPE + "/Mesh.obj")

		self.uvw_mapping = self.io.load_json_file("./Models3D/" + OBJECT_TYPE + "/UVWmapping.json")

		self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.image_w, self.image_h)
		self.renderer.scene.add_model(OBJECT_TYPE + "Model", self.model3d)

		self.uvw_coords = self.create_uvw_mapping(self.point_cloud)

		self.pose_ranges = {"RotX": (-180, 180),
		                    "RotY": (-180, 180),
		                    "RotZ": (-180, 180),
		                    "TransX": (-270, 270),
		                    "TransY": (-120, 120),
		                    "TransZ": (500, 1000)}

		self.noise_threshold = 2

		self.backgrounds_path = "./Backgrounds/"
		self.backgrounds_list = os.listdir(self.backgrounds_path)

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

	def create_uvw_mapping(self, coords3d: np.array) -> tuple:
		centroid = np.mean(coords3d, axis=0)
		x_centered = coords3d[..., 0] - centroid[0]
		y_centered = coords3d[..., 1] - centroid[1]
		z_centered = coords3d[..., 2] - centroid[2]

		x_scaled = self.get_min_max_scaling(x_centered)
		y_scaled = self.get_min_max_scaling(y_centered)
		z_scaled = self.get_min_max_scaling(z_centered)

		u = np.floor(x_scaled * (UVW_RANGE - 1))
		v = np.floor(y_scaled * (UVW_RANGE - 1))
		w = np.floor(z_scaled * (UVW_RANGE - 1))

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

	def get_rendered_uvw_maps(self, x_2d: np.array, y_2d: np.array, uvw_maps: tuple) -> tuple:
		u_map = np.zeros((self.image_h, self.image_w, 1))
		v_map = np.zeros((self.image_h, self.image_w, 1))
		w_map = np.zeros((self.image_h, self.image_w, 1))

		u_map[y_2d, x_2d] = uvw_maps[0].reshape(-1, 1)
		v_map[y_2d, x_2d] = uvw_maps[1].reshape(-1, 1)
		w_map[y_2d, x_2d] = uvw_maps[2].reshape(-1, 1)

		return u_map.astype(np.uint8), v_map.astype(np.uint8), w_map.astype(np.uint8)

	def setup_renderer_scene(self, direction: list, color: list, intensity: int) -> None:
		self.renderer.scene.scene.set_sun_light(direction, color, intensity)
		self.renderer.scene.scene.enable_sun_light(True)
		self.renderer.scene.scene.enable_light_shadow("directional", True)

	def load_random_background(self) -> o3d.geometry.Image:
		index = random.randint(0, len(self.backgrounds_list) - 1)
		background = cv2.imread(self.backgrounds_path + self.backgrounds_list[index])
		background = o3d.geometry.Image(background)

		return background

	def render_to_image(self, transformation_matrix: np.array, use_constant_light_cond: bool = False, image_black: bool = True, image_background: bool = True) -> dict:
		if use_constant_light_cond:
			direction = [0, 0, 0]
			intensity = 1000
			color = [255, 255, 255]
		else:
			direction = [0, 0, 0]
			intensity = random.randint(500, 2500)
			r = random.randint(100, 230)
			g = r + random.randint(-25, 25)
			b = r + random.randint(-25, 25)
			color = [r, g, b]

		self.setup_renderer_scene(direction, color, intensity)
		self.renderer.setup_camera(self.camera_intrinsic[0:3, 0:3], transformation_matrix, self.image_w, self.image_h)

		images_dict = {"black": None, "background": None}

		if image_black:
			self.renderer.scene.set_background(np.array([0, 0, 0, 1]))
			image_no_background = np.array(self.renderer.render_to_image())
			images_dict["black"] = image_no_background
		if image_background:
			background_image = self.load_random_background()
			self.renderer.scene.set_background(np.array([0, 0, 0, 1]), image=background_image)
			image_background = np.array(self.renderer.render_to_image())
			images_dict["background"] = image_background

		return images_dict

	@staticmethod
	def crop_and_resize(full_scale, bbox):
		x_min, y_min, x_max, y_max = bbox
		return cv2.resize(full_scale[y_min: y_max, x_min: x_max], (IMG_SIZE, IMG_SIZE))

	def get_image(self, transformation_matrix: np.array = None, bbox: list = None, image_black: bool = True, image_background: bool = True):
		if transformation_matrix is None:
			object_pose = self.sample_pose()
			transformation_matrix = self.transformations.get_transformation_matrix_from_pose(object_pose)

		images_dict = self.render_to_image(transformation_matrix, False, image_black, image_background)
		image_black = images_dict["black"]
		image_background = images_dict["background"]

		mask = self.get_object_mask(image_black)
		if bbox is None:
			bbox = self.get_bbox_from_mask(mask)

		x_2d, y_2d = self.project_point_cloud(self.point_cloud, transformation_matrix)

		u_map, v_map, w_map = self.get_rendered_uvw_maps(x_2d, y_2d, self.uvw_coords)

		rendered_image_dict = {"ImageBackground": image_background,
		                       "ImageBlack": image_black,
		                       "Mask": mask,
		                       "Umap": u_map,
		                       "Vmap": v_map,
		                       "Wmap": w_map,
		                       "Box": bbox,
		                       "Pose": object_pose
		                       }

		return rendered_image_dict

	def save_data(self, img_object, index, subset):
		json_data = {"Pose": None, "Box": None}
		for key, data in img_object.items():
			if key in ["Box", "Pose"]:
				json_data[key] = data
			else:
				path = DATASET_PATH + subset + "/" + key + "/data_{}.np".format(index)
				self.io.save_numpy_file(path, data)
		path = DATASET_PATH + subset + "/" + "Pose" + "/data_{}.json".format(index)
		self.io.save_json_file(path, json_data)

	def render_dataset(self):
		for subset in ["Training", "Validation"]:
			for data_index in range(DATA_AMOUNT[subset]):
				rendered_image_dict = self.get_image()
				self.save_data(rendered_image_dict, data_index, subset)
				print("{} image number {}/{} has been rendered".format(subset, data_index, DATA_AMOUNT[subset]))


if __name__ == "__main__":
	dataset_renderer = DatasetRenderer()
	dataset_renderer.render_dataset()
