from CONFIG import *
from RefinedPoseEstimation.CnnModel import PoseRefinementNetwork
import torch
import cv2
from Utils.DataAugmentationUtils import NormalizeToTensor
from DatasetRenderer.Renderer import DatasetRenderer

class RefinedPoseEstimation:
	def __init__(self, device):
		self.pose_refinement_model = PoseRefinementNetwork()
		self.dataset_renderer = DatasetRenderer()
		self.input_size = 224
		self.device = device
		self.load_model_weights()

	def load_model_weights(self):
		self.pose_refinement_model.load_state_dict(torch.load("./RefinedPoseEstimation/TrainedModels/RefinedPoseEstimationModel.pt",
		                                                      map_location="cpu"))
		self.pose_refinement_model.eval()
		self.pose_refinement_model.to(self.device)

	@staticmethod
	def normalize_convert_to_tensor(image):
		return NormalizeToTensor(image=image)["image"]

	def convert_images_to_tensors(self, images):
		batch_tensor = torch.tensor([])
		for index in range(images.shape[0]):
			image = images[index, ...]
			image_torch = self.normalize_convert_to_tensor(image)
			batch_tensor = torch.cat([batch_tensor, image_torch], dim=0)
		return batch_tensor.unsqueeze(0)

	def crop_and_resize(self, frame, bbox):
		x_min, y_min, x_max, y_max = bbox
		width = x_max - x_min
		height = y_max - y_min
		max_size = max(width, height)
		
		return cv2.resize(frame[y_min:y_min + max_size, x_min:x_min + max_size], (self.input_size, self.input_size))

	def render_refinement_images(self, poses, bboxes):
		batch_tensor = torch.tensor([])
		for index in range(poses.shape[0]):
			img_dict = self.dataset_renderer.get_image(transformation_matrix=poses[index, ...], image_black=True, image_background=False, constant_light=True)
			image = img_dict["ImageBlack"]
			cropped_image = self.crop_and_resize(image, bboxes[index])
			'''cv2.imshow("img", cropped_image)
			cv2.waitKey(0)'''
			image_torch = self.normalize_convert_to_tensor(cropped_image)
		batch_tensor = torch.cat([batch_tensor, image_torch], dim=0)
		return batch_tensor.unsqueeze(0)

	def get_refined_pose(self, real_images, coarse_poses, bboxes):
		real_images_batch = self.convert_images_to_tensors(real_images).to(self.device)
		rendered_images_batch = self.render_refinement_images(coarse_poses, bboxes).to(self.device)
		#print(real_images_batch.shape, rendered_images_batch.shape)
		visualize = torch.cat([real_images_batch, rendered_images_batch], dim=-1)[0].permute(1, 2, 0).detach().cpu().numpy()
		#cv2.imshow('vis', visualize)
		#cv2.waitKey(0)
		with torch.no_grad():
			pose_input = torch.cat([torch.tensor(coarse_poses[..., 0:3, -1]).float(), torch.tensor(coarse_poses[..., 0:3, -1]).float()], dim=-1).to(self.device)

			predictions = self.pose_refinement_model(real_images_batch, rendered_images_batch, pose_input)

		return predictions.detach().cpu().numpy()
	

if __name__ == "__main__":
	obj_segm = RefinedPoseEstimation("mps")
