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
		self.z_displacements = [[random.randint(-75, 75) for j in range(2)] for i in range(self.dataset_len)]
		self.x_displacements = [[random.randint(-25, 25) for j in range(2)] for i in range(self.dataset_len)]
		self.y_displacements = [[random.randint(-25, 25) for j in range(2)] for i in range(self.dataset_len)]
		self.A_displacements = [[random.randint(-15, 15) for j in range(2)] for i in range(self.dataset_len)]
		self.B_displacements = [[random.randint(-15, 15) for j in range(2)] for i in range(self.dataset_len)]
		self.C_displacements = [[random.randint(-15, 15) for j in range(2)] for i in range(self.dataset_len)]
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
		size_20_percent = 0#size // 5
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

	def distort_target_pose(self, pose):
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

	def distort_ref_poses(self, pose):
		distorted_pose = {"RotX": None, "RotY": None, "RotZ": None, "TransX": None, "TransY": None, "TransZ": None}
		for param in pose.keys():
			if "Rot" in param:
				distorted_pose[param] = pose[param] + random.randint(-180, 180)
			elif param == "TransX":
				distorted_pose[param] = pose[param]
			elif param == "TransY":
				distorted_pose[param] = pose[param]
			elif param == "TransZ":
				distorted_pose[param] = pose[param]#random.randint(-50, 50) #* self.index

		return distorted_pose

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
		x_min = x_c_projected - int(size//2)
		y_min = y_c_projected - int(size//2)
		x_max = x_c_projected + int(size//2)
		y_max = y_c_projected + int(size//2)
		return x_min, y_min, x_max, y_max

	def __getitem__(self, index):
		try:
			path = MAIN_DIR_PATH + "/Dataset/" + self.subset + "/"
			self.index = index
			real_image = self.io.load_numpy_file(path + "ImageBackground/" + "data_{}.np".format(index))

			json_data = self.io.load_json_file(path + "Pose/" + "data_{}.json".format(index))
			
			real_pose = json_data["Pose"]
   
			refinement_pose_number = random.randint(0, 9)
			path_datapoint = path + "ImageRefinement/" + "Data_{}/".format(index)
			refinement_image_path = path_datapoint + "Image/" + "data_{}.npy".format(refinement_pose_number)
			refinement_data_path = path_datapoint + "Pose/" + "data_{}.json".format(refinement_pose_number)
   
			rendered_image = self.io.load_numpy_file(refinement_image_path)
			rendered_data = self.io.load_json_file(refinement_data_path)
			rendered_pose = rendered_data["Pose"]
			rendered_bbox = rendered_data["Box"]
			trans_matrix_real = self.transformations.get_transformation_matrix_from_pose(real_pose)
			trans_matrix_rendered = self.transformations.get_transformation_matrix_from_pose(rendered_pose)
			
			real_image = self.crop_and_resize(real_image, rendered_bbox)
			#cv2.imshow("images", np.concatenate([image_real, rendered_image]))
			#cv2.waitKey(0)
			"""print(real_pose["TransX"] - rendered_pose["TransX"], real_pose["TransY"] - rendered_pose["TransY"], real_pose["TransZ"] - rendered_pose["TransZ"])
			"""
			"""trans_matrix_target = self.transformations.get_transformation_matrix_from_pose(target_pose)

			coarse_pose1 = self.distort_target_pose(target_pose)
			trans_matrix_coarse1 = self.transformations.get_transformation_matrix_from_pose(coarse_pose1) 

			rendered_image_dict1 = self.dataset_renderer.get_image(trans_matrix_coarse1, coarse_pose1, image_black=True, image_background=False, UVW=False, constant_light=True)

			refinement_image1 = rendered_image_dict1["ImageBlack"] * np.expand_dims(rendered_image_dict1["Mask"], axis=-1)
		
			bbox_image = json_data["Box"]
			bbox_rendered = rendered_image_dict1["Box"]
			bbox_crop = self.get_centered_bbox(trans_matrix_coarse1, bbox_rendered=bbox_rendered, bbox_image=bbox_image)
   
			real_image = self.crop_and_resize(real_image, bbox_crop)
			
			refinement_image1 = self.crop_and_resize(refinement_image1, bbox_crop)"""
				
			if self.data_augmentation:
				real_image = self.data_augmentation(image=real_image)["image"]
			image_tensor = NormalizeToTensor(image=real_image)["image"]
			rendered_image_tensor = NormalizeToTensor(image=rendered_image)["image"]

			return image_tensor, rendered_image_tensor, \
				torch.tensor(trans_matrix_real, dtype=torch.float32), \
				torch.tensor(trans_matrix_rendered, dtype=torch.float32)
		except Exception as e:
			print(e)
			print("IMAGE ERROR IS: ", index)
			self.__getitem__(index)
