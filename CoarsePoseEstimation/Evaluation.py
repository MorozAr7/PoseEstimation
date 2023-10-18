from CONFIG import *
from CoarsePoseEstimation.CnnModel import AutoencoderPoseEstimationModel
from Utils.DataAugmentationUtils import NormalizeToTensor
from Utils.IOUtils import IOUtils
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np


class CoarsePoseEvaluation:
	def __init__(self, device):
		self.pose_estimation_model = AutoencoderPoseEstimationModel()
		self.io = IOUtils()
		self.correspondence_uvw_mapping = self.io.load_json_file(MAIN_DIR_PATH + "/DatasetRenderer/Models3D/Chassis/ChassisUVWmapping.json")
		self.camera_intrinsic = np.array(self.io.load_json_file(MAIN_DIR_PATH + "/CameraData/camera_data_1.json")["K"])[0:3, 0:3]
		self.input_size = 224
		self.distortion_coefficients = np.array(self.io.load_json_file(MAIN_DIR_PATH + "/CameraData/camera_data_1.json")["dist_coef"])
		self.device = device
		self.init_pose_estimation_model()

	def init_pose_estimation_model(self):
		self.pose_estimation_model.load_state_dict(torch.load(MAIN_DIR_PATH + "/CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorNegativeRange.pt",
		                                                      map_location="cpu"))
		self.pose_estimation_model.eval()
		self.pose_estimation_model.to(self.device)

	def convert_images_to_tensors(self, images):
		batch_tensor = torch.tensor([])
		for index in range(images.shape[0]):
			image = images[index, ...]
			image_tensor = self.normalize_convert_to_tensor(image)
			batch_tensor = torch.cat([batch_tensor, image_tensor], dim=0)
		return batch_tensor.unsqueeze(0)

	@staticmethod
	def normalize_convert_to_tensor(image):
		return NormalizeToTensor(image=image)["image"]

	def get_uwv_predictions(self, image):
		image = cv2.resize(image, (self.input_size, self.input_size))
		object_image_tensor = self.normalize_convert_to_tensor(image).unsqueeze(0)
		return self.pose_estimation_model(object_image_tensor)

	def solve_pnp_ransac(self, points3d, points2d):
		predicted_transformation_matrix = np.zeros(shape=(4, 4))
		try:
			_, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(points3d, dtype=float),
			                                              np.array(points2d, dtype=float), self.camera_intrinsic,
			                                              iterationsCount=100, reprojectionError=1, flags=cv2.SOLVEPNP_P3P,distCoeffs=self.distortion_coefficients)

			rot, _ = cv2.Rodrigues(rvecs, jacobian=None)

			predicted_transformation_matrix[0:3, 0:3] = np.array(rot)
			predicted_transformation_matrix[0:3, 3:4] = np.array(tvecs)
			predicted_transformation_matrix[3:4, 3:4] = 1
		except Exception as e:
			print(e)
			predicted_transformation_matrix[0, 0] = 1
			predicted_transformation_matrix[1, 1] = 1
			predicted_transformation_matrix[2, 2] = 1
			predicted_transformation_matrix[3, 3] = 1

		return predicted_transformation_matrix

	def get_coarse_pose_estimate(self, images: np.array, masks: np.array, bboxes):
		with torch.no_grad():
			batch_tensor = self.convert_images_to_tensors(images).to(self.device)
			print(batch_tensor.shape)
			uvw_predicted = self.pose_estimation_model(batch_tensor)
		              
		coarse_pose_predictions = np.array([])
		for index in range(images.shape[0]):
			mask = np.array(uvw_predicted[3][index, ...].detach().cpu().numpy() > 0.99, dtype=bool).reshape(self.input_size, self.input_size)#np.array(masks[index, ...], dtype=bool).reshape(self.input_size, self.input_size)
			print(mask.shape)
			bbox = bboxes[index, ...]

			u_predicted = np.floor(255 * uvw_predicted[0].permute(0, 2, 3, 1).detach().cpu().numpy()[index])
			v_predicted = np.floor(255 * uvw_predicted[1].permute(0, 2, 3, 1).detach().cpu().numpy()[index])
			w_predicted = np.floor(255 * uvw_predicted[2].permute(0, 2, 3, 1).detach().cpu().numpy()[index])
			u_predicted = np.array(u_predicted, dtype=int)
			v_predicted = np.array(v_predicted, dtype=int)
			w_predicted = np.array(w_predicted, dtype=int)
			#u_predicted = torch.argmax(uvw_predicted[0], dim=1, keepdim=True).permute(0, 2, 3, 1).detach().cpu().numpy()[index]
			#v_predicted = torch.argmax(uvw_predicted[1], dim=1, keepdim=True).permute(0, 2, 3, 1).detach().cpu().numpy()[index]
			#w_predicted = torch.argmax(uvw_predicted[2], dim=1, keepdim=True).permute(0, 2, 3, 1).detach().cpu().numpy()[index]

			visualize = np.concatenate([u_predicted * masks[index, ...], v_predicted* masks[index, ...], w_predicted* masks[index, ...]], axis=0)/255

			#cv2.imshow("image", visualize)
			#cv2.waitKey(0)
			u_masked = np.array(u_predicted)[mask].reshape(-1, 1)
			v_masked = np.array(v_predicted)[mask].reshape(-1, 1)
			w_masked = np.array(w_predicted)[mask].reshape(-1, 1)
			
			uvw_array = np.concatenate([u_masked, v_masked, w_masked], axis=-1)

			coords_2d_x = np.arange(self.input_size).reshape(1, -1).repeat(repeats=self.input_size, axis=0)
			coords_2d_y = np.arange(self.input_size).reshape(-1, 1).repeat(repeats=self.input_size, axis=1)

			coords_x_masked = coords_2d_x[mask]
			coords_y_masked = coords_2d_y[mask]

			points_2d = []
			points_3d = []

			scale_coefficient = max((bbox[3] - bbox[1]), (bbox[2] - bbox[0])) / self.input_size

			for i in range(uvw_array.shape[0]):
				try:
					points_3d.append(self.correspondence_uvw_mapping[str(tuple(uvw_array[i, :]))])
					points_2d.append([coords_x_masked[i] * scale_coefficient + bbox[0], coords_y_masked[i] * scale_coefficient + bbox[1]])
				except KeyError:
					pass

			predicted_transformation_matrix = self.solve_pnp_ransac(points_3d, points_2d)

			predicted_transformation_matrix = np.expand_dims(predicted_transformation_matrix, axis=0)
			if index == 0:
				coarse_pose_predictions = predicted_transformation_matrix
			else:
				coarse_pose_predictions = np.concatenate([coarse_pose_predictions, predicted_transformation_matrix], axis=0)
		return coarse_pose_predictions
	

if __name__ == "__main__":
	coarse_pose_estimation = CoarsePoseEvaluation("mps")
