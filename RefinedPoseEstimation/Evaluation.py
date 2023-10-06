from CONFIG import *
from RefinedPoseEstimation.CnnModel import PoseRefinementNetwork
from ObjectSegmentation.Evaluation import ObjectSegmentationEvaluation
import torch
import cv2
import numpy as np
from Utils.DataAugmentationUtils import NormalizeToTensor, PoseEstimationAugmentation
from DatasetRenderer.Renderer import DatasetRenderer

class RefinedPoseEstimation:
	def __init__(self, device):
		self.device = device
		self.pose_refinement_model = PoseRefinementNetwork()
		self.dataset_renderer = DatasetRenderer()
		self.object_segmentation = ObjectSegmentationEvaluation(self.device)
		self.input_size = 224
		self.load_model_weights()

	def load_model_weights(self):
		self.pose_refinement_model.load_state_dict(torch.load("./RefinedPoseEstimation/TrainedModels/RefinedPoseEstimationModelProjection2DGrayScale.pt",
		                                                      map_location="cpu"))
		self.pose_refinement_model.eval()
		self.pose_refinement_model.to(self.device)

	@staticmethod
	def normalize_convert_to_tensor(image):
		
		return NormalizeToTensor(image=image)["image"]

	def convert_images_to_tensors(self, images, bbox):
		batch_tensor = torch.tensor([])
		for index in range(images.shape[0]):
			image = images[index, ...]
	
			image = self.crop_and_resize(image, bbox)
			#resized_image = image.reshape(1, 224, 224, 1)
			#resized_image = np.repeat(resized_image, axis=-1, repeats=3)
			#segmentation = self.object_segmentation.segment_image(resized_image)
			#print(image.shape, segmentation.shape)
			#image = image# * segmentation.reshape(224, 224)
			if image is None:
				return None
			image_torch = self.normalize_convert_to_tensor(image)
			batch_tensor = torch.cat([batch_tensor, image_torch], dim=0)
		return batch_tensor.unsqueeze(0)

	def crop_and_resize(self, frame, bbox):
		x_min, y_min, x_max, y_max = bbox

		cropped = frame[y_min:y_max, x_min:x_max]
		try:
			return cv2.resize(cropped, (self.input_size, self.input_size))
		except Exception as e:
			return None

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
		return int(x_min), int(y_min), int(x_max), int(y_max)
 
	def render_refinement_images(self, poses, bboxes):
		batch_tensor = torch.tensor([])
		for index in range(poses.shape[0]):
			img_dict = self.dataset_renderer.get_image(transformation_matrix=poses[index, ...], image_black=True, image_background=False, constant_light=True)
			image = img_dict["ImageBlack"]
			bbox_rendered = img_dict["Box"]
			bbox = self.get_centered_bbox(poses[index, ...], bbox_rendered=bbox_rendered, bbox_image=bboxes[index])

			cropped_image = self.crop_and_resize(image, bbox)
			cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
			if cropped_image is None:
				return None, None
			image_torch = self.normalize_convert_to_tensor(cropped_image)
		batch_tensor = torch.cat([batch_tensor, image_torch], dim=0)

		return batch_tensor.unsqueeze(0), bbox

	def get_refined_pose(self, real_images, coarse_poses, bboxes, index):
		rendered_images_batch, box = self.render_refinement_images(coarse_poses, bboxes)
		if rendered_images_batch is None:
			return None
		rendered_images_batch = rendered_images_batch.to(self.device)
		real_images_batch = self.convert_images_to_tensors(np.expand_dims(real_images, axis=0), box).to(self.device)
		if real_images_batch is None:
			return None
		#print(real_images_batch.shape, rendered_images_batch.shape)
		vis = torch.cat([real_images_batch, rendered_images_batch], dim=-1).permute(0, 2, 3, 1)[0].detach().cpu().numpy()
		cv2.imwrite("/Users/artemmoroz/Desktop/CIIRC_projects/PoseEstimation/RefinedPoseEstimation/SavedImagesReal/image_{}.png".format(index), vis * 255)
		cv2.imshow("input", vis)
		cv2.waitKey(1)
		print(rendered_images_batch.shape, real_images_batch.shape)
		with torch.no_grad():
			prediction_t, prediction_R = self.pose_refinement_model(real_images_batch, rendered_images_batch)
			coarse_poses = torch.tensor(coarse_poses, dtype=torch.float32).to(self.device)
			refined_T = coarse_poses.clone()

			t_coarse = coarse_poses[..., :3, -1]
			R_coarse = coarse_poses[..., :3, :3] 

			updated_z = prediction_t[..., 2:] * t_coarse[..., 2:3]
			updated_xy = (prediction_t[..., :2] + t_coarse[..., :2] / t_coarse[..., 2:3]) * updated_z
			updated_R = prediction_R @ R_coarse#R_coarse#torch.bmm(prediction_R, R_coarse)

			refined_T[..., :3, :3] = updated_R

			refined_T[..., 2:3, -1] = updated_z
			refined_T[..., :2, -1] = updated_xy

		return refined_T.detach().cpu().numpy()[0, ...]
	

if __name__ == "__main__":
	obj_segm = RefinedPoseEstimation("mps")
