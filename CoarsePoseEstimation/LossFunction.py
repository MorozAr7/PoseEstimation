import torch
from torch import nn
from torch.nn.functional import conv2d, avg_pool2d
import torchvision
import cv2


class DotProductLoss(nn.Module):
	def __init__(self, DEVICE):
		super(DotProductLoss, self).__init__()
		self.DEVICE = DEVICE
		self.L1Loss = nn.L1Loss(reduction="sum")

	def forward(self, prediction, target):
		loss = self.L1Loss(prediction, target)

		return loss


class MultiscaleSsimLossFunction(nn.Module):
	def __init__(self, DEVICE):
		super(MultiscaleSsimLossFunction, self).__init__()
		self.K1 = 0.01
		self.K2 = 0.03
		self.C1 = (self.K1 * 1) ** 2
		self.C2 = (self.K2 * 1) ** 2
		self.DEVICE = DEVICE
		self.num_scales = 3

	def compute_patches_mean(self, image_tensor: torch.tensor) -> torch.tensor:
		convolved = conv2d(input=image_tensor, weight=self.ones_filter / (self.window_size ** 2), stride=1)
		return convolved

	def compute_patches_std(self, image_tensor, mean_map):
		convolved = conv2d(image_tensor ** 2, weight=self.ones_filter / (self.window_size ** 2 - 1), stride=1) - mean_map ** 2
		return convolved

	def compute_joint_std(self, in_painted_image_tensor, real_image_tensor, prediction_mean_map, ground_truth_mean_map):
		convolved = conv2d(input=in_painted_image_tensor * real_image_tensor, weight=self.ones_filter / (self.window_size ** 2 - 1), stride=1) - prediction_mean_map * ground_truth_mean_map
		return convolved

	def get_multiscale_representation(self, images_tensor: torch.tensor) -> list:
		multiscale_images = []
		for scale in range(self.num_scales):
			multiscale_images.append(images_tensor)
			images_tensor = avg_pool2d(images_tensor, kernel_size=2, stride=2)

		return multiscale_images

	@staticmethod
	def resize_tensor(image_tensor, image_size):
		resize_transform = torch.nn.Sequential(torchvision.transforms.Resize((image_size, image_size)))
		return resize_transform(image_tensor)

	def get_ssim_maps(self, predicted_image_tensor, target_image_tensor, window_size=5):
		self.window_size = window_size
		self.ones_filter = torch.ones(size=(1, 1, self.window_size, self.window_size), dtype=torch.float32).to(self.DEVICE)
		prediction_mean_map = self.compute_patches_mean(predicted_image_tensor)
		ground_truth_mean_map = self.compute_patches_mean(target_image_tensor)

		prediction_std_map = self.compute_patches_std(predicted_image_tensor, prediction_mean_map)
		ground_truth_std_map = self.compute_patches_std(target_image_tensor, ground_truth_mean_map)

		joint_std_map = self.compute_joint_std(predicted_image_tensor, target_image_tensor, prediction_mean_map, ground_truth_mean_map)

		brightness_sim = (2 * prediction_mean_map * ground_truth_mean_map + self.C2) / (prediction_mean_map ** 2 + ground_truth_mean_map ** 2 + self.C2)
		contrast_sim = (2 * torch.sqrt(prediction_std_map + 1e-8) * torch.sqrt(ground_truth_std_map + 1e-8) + self.C1) / (prediction_std_map + ground_truth_std_map + self.C1)
		content_sim = (joint_std_map + self.C1) / (torch.sqrt(prediction_std_map + 1e-8) * torch.sqrt(ground_truth_std_map + 1e-8) + self.C1)
		ssim_map = content_sim * brightness_sim * contrast_sim

		return 1 - torch.mean(ssim_map)

	@staticmethod
	def convolve(image, kernel):
		
		convolved = nn.functional.conv2d(image, weight=kernel.unsqueeze(0).unsqueeze(0), stride=1, padding=1)
		return convolved

	def compute_gradient_map(self, tensor):
		filter_1 = torch.tensor([[0, 0, 0, 0, 0], [1, 3, 8, 3, 1], [0, 0, 0, 0, 0], [-1, -3, -8, -3, -1], [0, 0, 0, 0, 0]]).to(self.DEVICE) / 1
		filter_2 = torch.tensor([[0, 1, 0, -1, 0], [0, 3, 0, -3, 0], [0, 8, 0, -8, 0], [0, 3, 0, -3, 0], [0, 1, 0, -1, 0]]).to(self.DEVICE) / 1
		filter_3 = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 3, 8, 0], [-1, -3, 0, 3, 1], [0, -8, -3, 0, 0], [0, 0, -1, 0, 0]]).to(self.DEVICE) / 1
		filter_4 = torch.tensor([[0, 0, 1, 0, 0], [0, 8, 3, 0, 0], [1, 3, 0, -3, -1], [0, 0, -3, -8, 0], [0, 0, -1, 0, 0]]).to(self.DEVICE) / 1

		image_1_gradient = torch.abs(self.convolve(tensor, filter_1).unsqueeze(0))
		image_2_gradient = torch.abs(self.convolve(tensor, filter_2).unsqueeze(0))
		image_3_gradient = torch.abs(self.convolve(tensor, filter_3).unsqueeze(0))
		image_4_gradient = torch.abs(self.convolve(tensor, filter_4).unsqueeze(0))
		concatenated = torch.cat([image_1_gradient, image_2_gradient, image_3_gradient, image_4_gradient], dim=0)
		image_grad_map = torch.max(concatenated, dim=0, keepdim=False)[0]

		return image_grad_map

	def get_gradient_ssim_loss(self, predicted_image_tensor, target_image_tensor, window_size=7):
		predicted_image_gradients = self.compute_gradient_map(predicted_image_tensor)
		target_image_gradients = self.compute_gradient_map(target_image_tensor)

		gradient_similarity_loss = self.get_ssim_maps(predicted_image_gradients, target_image_gradients, window_size=window_size)

		return gradient_similarity_loss

	def get_multiscale_structural_sim_loss(self, predicted_image_tensor, target_image_tensor, window_size=7):
		predicted_multiscale = self.get_multiscale_representation(predicted_image_tensor)
		target_multiscale = self.get_multiscale_representation(target_image_tensor)
		structural_similarity_loss = 0
		for i in range(self.num_scales):
			
			structural_similarity_loss += self.get_ssim_maps(predicted_multiscale[i], target_multiscale[i], window_size)

		return structural_similarity_loss


if __name__ == "__main__":
	ssim_api = MultiscaleSsimLossFunction("cpu")
	predicted_tensor = torch.cat([-2*torch.ones(size=(1, 1, 32, 32)), torch.ones(size=(1, 1, 32, 32)), -torch.ones(size=(1, 1, 32, 32)), 0 * torch.ones(size=(1, 1, 32, 32))], dim=0)
	target_tensor = torch.cat([-2 * torch.ones(size=(1, 1, 32, 32)), torch.ones(size=(1, 1, 32, 32)), -torch.ones(size=(1, 1, 32, 32)), 0 * torch.ones(size=(1, 1, 32, 32))], dim=0)

	loss = ssim_api.get_multiscale_structural_sim_loss(predicted_tensor, target_tensor)
	print(loss)

