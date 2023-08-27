import sys

sys.path.insert(0, sys.path[1] + "/Utils")
print(sys.path[1])
from Utils.ConvUtils import *
import numpy as np
from CONFIG import DEVICE


class EncoderModel(nn.Module):
	def __init__(self):
		super(EncoderModel, self).__init__()
		self.layer_channels = [3, 32, 64, 128, 192, 256, 512]

		self.Conv0 = ConvBnReLU(in_channels=self.layer_channels[0], out_channels=self.layer_channels[1])

		self.Conv1 = ConvBnReLU(in_channels=self.layer_channels[1], out_channels=self.layer_channels[2], stride=2)
		self.ResLayer1 = DepthWiseConvResidualBlock(in_channels=self.layer_channels[2], out_channels=self.layer_channels[2])
		self.ResLayer2 = DepthWiseConvResidualBlock(in_channels=self.layer_channels[2], out_channels=self.layer_channels[2])

		self.Conv2 = ConvBnReLU(in_channels=self.layer_channels[2], out_channels=self.layer_channels[3], stride=2)
		self.ResLayer3 = DepthWiseConvResidualBlock(in_channels=self.layer_channels[3], out_channels=self.layer_channels[3])
		self.ResLayer4 = DepthWiseConvResidualBlock(in_channels=self.layer_channels[3], out_channels=self.layer_channels[3])

		self.Conv3 = ConvBnReLU(in_channels=self.layer_channels[3], out_channels=self.layer_channels[4], stride=2)
		self.ResLayer5 = DepthWiseConvResidualBlock(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])
		self.ResLayer6 = DepthWiseConvResidualBlock(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])

		self.Conv4 = ConvBnReLU(in_channels=self.layer_channels[4], out_channels=self.layer_channels[5], stride=2)
		self.Conv5 = ConvBnReLU(in_channels=self.layer_channels[5], out_channels=self.layer_channels[6], kernel_size=1, stride=1, apply_bn=False, apply_relu=False, padding=0, apply_bias=False)

	def forward(self, x):
		x = self.Conv0(x)
		x = self.Conv1(x)

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
		self.EncoderRendered = EncoderModel()
		self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

		self.z_linear_1 = nn.Linear(512, 256)
		self.z_linear_2 = nn.Linear(256, 1)

		self.xy_linear_1 = nn.Linear(512, 256)
		self.xy_linear_2 = nn.Linear(256, 2)

		self.rotation_linear_1 = nn.Linear(512, 256)
		self.rotation_linear_2 = nn.Linear(256, 3)

		self.ReLU = nn.ReLU(inplace=True)
		self.Tanh = nn.Tanh()

	def forward_rotation_linear(self, feature_vector):

		x = self.rotation_linear_1(feature_vector)
		x = self.ReLU(x)
		x = self.rotation_linear_2(x)
		x = self.Tanh(x)
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

		return x

	def forward_cnn(self, x_real, x_rendered):
		features_real = self.EncoderReal(x_real)
		features_rendered = self.EncoderRendered(x_rendered)
		diff = features_real - features_rendered

		feature_vector = self.AvgPool(diff).reshape(-1, 512)

		return feature_vector

	def forward(self, x_real, x_rendered, prediction):

		feature_vector = self.forward_cnn(x_real, x_rendered)

		prediction_xy = prediction[..., 0:2]
		prediction_z = prediction[..., 2:3]
		prediction_rot = prediction[..., 3:]

		refined_z = torch.exp(self.forward_z_linear(feature_vector)) * prediction_z
		refined_xy = (self.forward_xy_linear(feature_vector) + prediction_xy / prediction_z) * refined_z
		refined_rotation = self.forward_rotation_linear(feature_vector) + prediction_rot

		return torch.cat([refined_xy, refined_z, refined_rotation], dim=1)


if __name__ == "__main__":
	pose_ref_cnn = PoseRefinementNetwork().to(DEVICE)
	model_parameters = filter(lambda p: p.requires_grad, pose_ref_cnn.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("NUMBER PARAMS GENERATOR:", params)
	input_tensor = torch.ones(size=(8, 3, 224, 224)).to(DEVICE)
	u = pose_ref_cnn(input_tensor, input_tensor)
	print(u[0].shape)