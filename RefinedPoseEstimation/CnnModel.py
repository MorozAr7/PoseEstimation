import sys
from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import cv2
from Utils.ConvUtils import *
import numpy as np
from CONFIG import DEVICE


class EncoderModel(nn.Module):
	def __init__(self):
		super(EncoderModel, self).__init__()
		self.layer_channels = [2, 64, 96, 128, 256, 512, 512 * 3]

		self.Conv0 = ConvBnActiv(in_channels=self.layer_channels[0], out_channels=self.layer_channels[1], kernel_size=7, padding=3)

		self.Conv1 = ConvBnActiv(in_channels=self.layer_channels[1], out_channels=self.layer_channels[2], stride=2, kernel_size=5, padding=2)
		self.ResLayer1 = ResidualBlock(in_channels=self.layer_channels[2], out_channels=self.layer_channels[2])
		self.ResLayer2 = ResidualBlock(in_channels=self.layer_channels[2], out_channels=self.layer_channels[2])

		self.Conv2 = ConvBnActiv(in_channels=self.layer_channels[2], out_channels=self.layer_channels[3], stride=2)
		self.ResLayer3 = ResidualBlock(in_channels=self.layer_channels[3], out_channels=self.layer_channels[3])
		self.ResLayer4 = ResidualBlock(in_channels=self.layer_channels[3], out_channels=self.layer_channels[3])

		self.Conv3 = ConvBnActiv(in_channels=self.layer_channels[3], out_channels=self.layer_channels[4], stride=2)
		self.ResLayer5 = ResidualBlock(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])
		self.ResLayer6 = ResidualBlock(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])

		self.Conv4 = ConvBnActiv(in_channels=self.layer_channels[4], out_channels=self.layer_channels[5], stride=2)
		self.Conv5 = ConvBnActiv(in_channels=self.layer_channels[5], out_channels=self.layer_channels[6], kernel_size=1, stride=1, apply_bn=False, apply_activation=True, padding=0, apply_bias=True)
		
	def forward(self, x):
		x = self.Conv0(x)
		print(x.shape)
		x = self.Conv1(x)
		print(x.shape)
		x = self.ResLayer1(x)
		x = self.ResLayer2(x)
		
		x = self.Conv2(x)

		x = self.ResLayer3(x)
		x = self.ResLayer4(x)

		x = self.Conv3(x)
  
		x = self.ResLayer5(x)
		x = self.ResLayer6(x)
  
		x = self.Conv4(x)
		x = self.Conv5(x)

		return x


class PoseRefinementNetwork(nn.Module):
	def __init__(self):
		super(PoseRefinementNetwork, self).__init__()
		self.EncoderReal = EncoderModel()
		self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

		self.xy_linear_1 = nn.Linear(512, 256)
		self.xy_linear_2 = nn.Linear(256, 2)

		self.z_linear_1 = nn.Linear(512, 256)
		self.z_linear_2 = nn.Linear(256, 1)
		
		self.rotation_linear_1 = nn.Linear(512, 256)
		self.rotation_linear_2 = nn.Linear(256, 6)

		self.ReLU = nn.ReLU()
  
	def orthonormalization(self, rotation_output):
		v1 = rotation_output[..., :3]
		v2 = rotation_output[..., 3:]

		v1 = v1 / torch.norm(v1, p=2, dim=-1, keepdim=True)
		v3 = torch.cross(v1, v2, dim=-1) 
		v3 = v3 / torch.norm(v3, p=2, dim=-1, keepdim=True)
		v2 = torch.cross(v3, v1, dim=-1)
  
		v1 = v1.reshape(-1, 3, 1)
		v2 = v2.reshape(-1, 3, 1)
		v3 = v3.reshape(-1, 3, 1)

		R = torch.cat([v1, v2, v3], dim=2)

		return R

	def forward_rotation_linear(self, feature_vector):

		x = self.rotation_linear_1(feature_vector)
		x = self.ReLU(x)
		x = self.rotation_linear_2(x)
		x = self.orthonormalization(x)
  
		return x

	def forward_xy_linear(self, feature_vector):
		x = self.xy_linear_1(feature_vector)
		x = self.ReLU(x)
		x = self.xy_linear_2(x)
  
		return x

	def forward_z_linear(self, feature_vector):
		x = self.z_linear_1(feature_vector)
		x = self.ReLU(x)
		x = self.z_linear_2(x)
		x = torch.exp(x)
  
		return x

	def forward_cnn(self, x):
		features_real = self.EncoderReal(x)
		features_real = self.AvgPool(features_real).reshape(-1, 512 * 3)

		return features_real

	def forward(self, images_real, images_rendered):
		x = torch.cat([images_real, images_rendered], dim=1)
		feature_vector = self.forward_cnn(x)

		z_output = self.forward_z_linear(feature_vector[..., :512])
		xy_output = self.forward_xy_linear(feature_vector[..., 512:1024])
		rotation_output = self.forward_rotation_linear(feature_vector[..., 1024:])
  
		translation_output = torch.cat([xy_output, z_output], dim=-1)
  
		return translation_output, rotation_output


if __name__ == "__main__":
	pose_ref_cnn = PoseRefinementNetwork().to(DEVICE)
	model_parameters = filter(lambda p: p.requires_grad, pose_ref_cnn.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("NUMBER PARAMS GENERATOR:", params)
	input_tensor = torch.ones(size=(8, 1, 224, 224)).to(DEVICE)
	u = pose_ref_cnn(input_tensor, input_tensor)
	print(u[0].shape)