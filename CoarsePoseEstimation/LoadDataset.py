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

	def crop_and_resize(self, array, bbox_corner, resize=True):
		x_min, y_min, x_max, y_max = bbox_corner

		cropped = array[y_min:y_max, x_min:x_max]

		if resize:
			try:
				resized = cv2.resize(cropped, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

				return resized
			except Exception as e:
				print(e)
				print(x_min, x_max, y_min, y_max)
				exit()
		else:
			return cropped

	@staticmethod
	def augment_bbox(bbox):
		centroid_x, centroid_y, width, height = bbox
		size_20_percent = int(0.2 * width)

		shift_x = random.randint(-size_20_percent, size_20_percent)
		shift_y = random.randint(-size_20_percent, size_20_percent)

		enlargement = random.randint(0, size_20_percent)

		return centroid_x + shift_x, centroid_y + shift_y, width + enlargement, height + enlargement

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

	def get_crop_bbox(self, bbox):
		square_bbox = self.get_square_bbox(bbox)
		p = random.random()
		if p > 0.8 or self.subset == "Validation":
			return square_bbox
		else:
			bbox = self.augment_bbox(square_bbox)
			return bbox

	def __getitem__(self, index):
		path = MAIN_DIR_PATH + "Dataset/" + self.subset + "/"
		#print("Image Index", index)
		image = cv2.imread(path + "ImageBackground/" + "img_{}.png".format(index))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		mask = self.io.load_numpy_file(path + "Mask/" + "data_{}.np".format(index))

		uvw_map = self.io.load_numpy_file(path + "UVWmap/" + "data_{}.np".format(index))
		u_map = uvw_map[..., 0] * mask
		v_map = uvw_map[..., 1] * mask
		w_map = uvw_map[..., 2] * mask

		json_data = self.io.load_json_file(path + "Label/" + "data_{}.json".format(index))

		bbox_tight = json_data["TightBox"]
		bbox_enlarged = json_data["EnlargedBox"]
		x_min_, y_min_, x_max_, y_max_ = bbox_enlarged
		bbox_corner = [bbox_tight[0] - x_min_, bbox_tight[1] - y_min_, bbox_tight[2] - x_min_, bbox_tight[3] - y_min_]
		bbox_centroid = self.convert_corner_to_centroid(bbox_corner)
		bbox_crop = self.get_crop_bbox(bbox_centroid)
		bbox_crop = self.convert_centroid_to_corner(bbox_crop)

		image = self.crop_and_resize(image, bbox_crop)
		mask = self.crop_and_resize(mask.astype(float), bbox_crop)
		u_map = self.crop_and_resize(u_map/250, bbox_crop)
		v_map = self.crop_and_resize(v_map/250, bbox_crop)
		w_map = self.crop_and_resize(w_map/250, bbox_crop)
		#cv2.imshow("img", image)
		#cv2.waitKey(0)
		"""cv2.imshow("img", image)
		cv2.waitKey(0)
		cv2.imshow("img", mask)
		cv2.waitKey(0)
		cv2.imshow("img", u_map)
		cv2.waitKey(0)
		cv2.imshow("img", v_map)
		cv2.waitKey(0)
		cv2.imshow("img", w_map)
		cv2.waitKey(0)"""

		if self.data_augmentation:
			image = self.data_augmentation(image=image)["image"]
		image_tensor = NormalizeToTensor(image=image)["image"]

		return image_tensor, \
			torch.tensor(np.expand_dims(mask, axis=2), dtype=torch.float32).permute(2, 0, 1), \
			torch.tensor(u_map, dtype=torch.float32), torch.tensor(v_map, dtype=torch.float32), torch.tensor(w_map, dtype=torch.float32),
