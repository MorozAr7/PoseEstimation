import random
import numpy as np
from CONFIG import *

sys.path.insert(0, MAIN_DIR_PATH)
import cv2
import torch
from Utils.IOUtils import IOUtils
from Utils.DataAugmentationUtils import NormalizeToTensor, NormalizeToTensor


class Dataset(torch.utils.data.Dataset):
	def __init__(self, subset, num_images, data_augmentation=None):
		super(Dataset, self).__init__()
		self.data_augmentation = data_augmentation
		self.subset = subset
		self.dataset_len = num_images
		self.image_size = 224
		self.io = IOUtils()

	def __len__(self):
		return self.dataset_len

	def crop_and_resize(self, array, bbox_corner):
		x_min, y_min, x_max, y_max = bbox_corner

		cropped = array[y_min:y_max, x_min:x_max]

		try:
			resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

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
	def get_shift_limits(bbox):
		size = bbox[2] - bbox[0]
		size_20_percent = size // 5
		limits = {"left": None, "right": None, "up": None, "down": None}
		for key, val in limits.items():
			limits[key] = size_20_percent

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
		p = random.random()
		if p > 0.75 or self.subset == "Validation":
			return square_bbox
		else:
			limits = self.get_shift_limits(square_bbox)
			shifted_bbox = self.shift_bbox(square_bbox, limits)
			return square_bbox

	def __getitem__(self, index):
		path = MAIN_DIR_PATH + "Dataset/" + self.subset + "/"

		image = cv2.imread(path + "ImageBackground/" + "img_{}.png".format(index))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = self.io.load_numpy_file(path + "Mask/" + "data_{}.np".format(index))
		uvw_map = self.io.load_numpy_file(path + "UVWmap/" + "data_{}.np".format(index))
		u_map = uvw_map[..., 0] * mask # self.io.load_numpy_file(path + "Umap/" + "data_{}.np".format(index))
		v_map = uvw_map[..., 1] * mask # self.io.load_numpy_file(path + "Vmap/" + "data_{}.np".format(index))
		w_map = uvw_map[..., 2] * mask # self.io.load_numpy_file(path + "Wmap/" + "data_{}.np".format(index))
		json_data = self.io.load_json_file(path + "Label/" + "data_{}.json".format(index))

		bbox = json_data["Box"]
		# img_width, img_height = image.shape[1], image.shape[0]
		x_min, y_min, x_max, y_max = bbox
		bbox_max_size = max(y_max - y_min, x_max - x_min)
		center_x = (x_min + x_max) // 2
		center_y = (y_min + y_max) // 2

		bbox_max_size_amplified = (1 + 0.5) * bbox_max_size
		x_min_ = int(center_x - bbox_max_size_amplified / 2)
		y_min_ = int(center_y - bbox_max_size_amplified / 2)
		bbox = [bbox[0] - x_min_, bbox[1] - y_min_, bbox[2] - x_min_, bbox[3] - y_min_]
		print(bbox)

		bbox_crop = self.get_bbox(bbox)
		image = self.crop_and_resize(image, bbox_crop)
		mask = self.crop_and_resize(mask.astype(float), bbox_crop)
		u_map = self.crop_and_resize(u_map/250, bbox_crop)
		v_map = self.crop_and_resize(v_map/250, bbox_crop)
		w_map = self.crop_and_resize(w_map/250, bbox_crop)
		print(np.max(u_map), np.min(u_map), np.max(v_map), np.min(v_map),np.max(w_map), np.min(w_map))
		cv2.imshow("img", image)
		cv2.waitKey(0)
		cv2.imshow("img", mask)
		cv2.waitKey(0)
		cv2.imshow("img", u_map)
		cv2.waitKey(0)
		cv2.imshow("img", v_map)
		cv2.waitKey(0)
		cv2.imshow("img", w_map)
		cv2.waitKey(0)

		if self.data_augmentation:
			image = self.data_augmentation(image=image)["image"]
		image_tensor = NormalizeToTensor(image=image)["image"]

		return image_tensor, \
			torch.tensor(np.expand_dims(mask, axis=2), dtype=torch.float32).permute(2, 0, 1), \
			torch.tensor(u_map, dtype=torch.float32), torch.tensor(v_map, dtype=torch.float32), torch.tensor(w_map, dtype=torch.float32),
