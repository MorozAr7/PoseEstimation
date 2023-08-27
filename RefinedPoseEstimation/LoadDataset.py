from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import random
import numpy as np
import cv2
import torch
from Utils.IOUtils import IOUtils
from Utils.DataAugmentationUtils import NormalizeToTensor


class Dataset(torch.utils.data.Dataset):
	def __init__(self, subset, num_images, dataset_renderer, data_augmentation=None):
		super(Dataset, self).__init__()
		self.data_augmentation = data_augmentation
		self.subset = subset
		self.dataset_len = num_images
		self.image_size = 224
		self.full_scale_w = 1865
		self.full_scale_h = 1039
		self.io = IOUtils()
		self.dataset_renderer = dataset_renderer

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
		x_min, y_min, x_max, y_max = square_bbox
		return x_min + shift_x, y_min + shift_y, x_max + shift_x, y_max + shift_y

	@staticmethod
	def get_shift_limits(bbox, check_results):
		size = bbox[2] - bbox[0]
		size_20_percent = size // 5
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
		#print(bbox, check_results, limits)
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

	@staticmethod
	def distort_target_pose(pose):
		distorted_pose = {"RotX": None, "RotY": None, "RotZ": None, "TransX": None, "TransY": None, "TransZ": None}
		for param in pose.keys():
			if "Rot" in param:
				distorted_pose[param] = pose[param] + random.randint(-15, 15)
			elif param == "TransX" or param == "TransY":
				distorted_pose[param] = pose[param] + random.randint(-20, 20)
			elif param == "TransZ":
				distorted_pose[param] = pose[param] + random.randint(-50, 50)

		return distorted_pose

	def __getitem__(self, index):
		path = MAIN_DIR_PATH + "/Dataset/" + self.subset + "/"

		image = self.io.load_numpy_file(path + "ImageBackground/" + "data_{}.np".format(index))
		json_data = self.io.load_json_file(path + "Pose/" + "data_{}.json".format(index))

		bbox = json_data["Box"]
		target_pose = json_data["Pose"]
		bbox_crop = self.get_bbox(bbox)

		image = self.crop_and_resize(image, bbox_crop)
		angles_target = np.array([target_pose["RotX"], target_pose["RotY"], target_pose["RotZ"]]) / 180
		t_target = np.array([target_pose["TransX"], target_pose["TransY"], target_pose["TransZ"]])

		coarse_pose = self.distort_target_pose(target_pose)
		rendered_image_dict = self.dataset_renderer.get_image(coarse_pose, bbox_crop, image_black=True, image_background=False, UVW=False)
		refinement_image = rendered_image_dict["ImageBlack"]
		refinement_image = self.crop_and_resize(refinement_image, bbox_crop)

		angles_coarse = np.array([coarse_pose["RotX"], coarse_pose["RotY"], coarse_pose["RotZ"]]) / 180
		t_coarse = np.array([coarse_pose["TransX"], coarse_pose["TransY"], coarse_pose["TransZ"]])

		if self.data_augmentation:
			image = self.data_augmentation(image=image)["image"]
		image_tensor = NormalizeToTensor(image=image)["image"]
		refinement_image_tensor = NormalizeToTensor(image=refinement_image)["image"]

		return image_tensor, \
		       refinement_image_tensor, \
		       torch.tensor(angles_target, dtype=torch.float32), \
		       torch.tensor(t_target, dtype=torch.float32), \
		       torch.tensor(angles_coarse, dtype=torch.float32), \
		       torch.tensor(t_coarse, dtype=torch.float32)

