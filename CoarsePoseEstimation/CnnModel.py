from Utils.ConvUtils import *


class EncoderModel(nn.Module):
	def __init__(self):
		super(EncoderModel, self).__init__()
		self.layer_channels = [3, 32, 64, 128, 192, 256]

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
		self.DilatedConv1 = ConvBnReLU(in_channels=self.layer_channels[5], out_channels=self.layer_channels[5], dilation_rate=2, padding=2)
		self.DilatedConv2 = ConvBnReLU(in_channels=self.layer_channels[5], out_channels=self.layer_channels[5], dilation_rate=4, padding=4)

	def forward(self, x):
		skip_connections = []
		x = self.Conv0(x)
		skip_connections.append(x)
		x = self.Conv1(x)
		skip_connections.append(x)
		x = self.ResLayer1(x)
		x = self.ResLayer2(x)
		x = self.Conv2(x)
		skip_connections.append(x)
		x = self.ResLayer3(x)
		x = self.ResLayer4(x)
		x = self.Conv3(x)
		skip_connections.append(x)
		x = self.ResLayer5(x)
		x = self.ResLayer6(x)
		x = self.Conv4(x)
		x = self.DilatedConv1(x)
		x = self.DilatedConv2(x)
		return x, skip_connections


class DecoderModel(nn.Module):
	def __init__(self):
		super(DecoderModel, self).__init__()
		self.channel = [256, 192, 128, 64, 32, 256]

		self.TransConv1 = TransposeConvBnReLU(in_channels=self.channel[0], out_channels=self.channel[1])
		self.Conv1 = ConvBnReLU(in_channels=self.channel[1], out_channels=self.channel[1])
		self.Conv2 = ConvBnReLU(in_channels=self.channel[1], out_channels=self.channel[1])
		self.TransConv2 = TransposeConvBnReLU(in_channels=self.channel[1], out_channels=self.channel[2])
		self.Conv3 = ConvBnReLU(in_channels=self.channel[2], out_channels=self.channel[2])
		self.Conv4 = ConvBnReLU(in_channels=self.channel[2], out_channels=self.channel[2])
		self.TransConv3 = TransposeConvBnReLU(in_channels=self.channel[2], out_channels=self.channel[3])
		self.Conv5 = ConvBnReLU(in_channels=self.channel[3], out_channels=self.channel[3])
		self.Conv6 = ConvBnReLU(in_channels=self.channel[3], out_channels=self.channel[3])
		self.TransConv4 = TransposeConvBnReLU(in_channels=self.channel[3], out_channels=self.channel[4])
		self.Conv7 = ConvBnReLU(in_channels=self.channel[4], out_channels=self.channel[4])
		self.Conv8 = ConvBnReLU(in_channels=self.channel[4], out_channels=self.channel[5], apply_bn=False, apply_relu=False, apply_bias=False)

		self.Sigmoid = nn.Sigmoid()

	def forward(self, x, skip_connections):
		x = self.TransConv1(x)
		x = self.Conv1(x)
		x = self.Conv2(x)
		x = self.TransConv2(x + skip_connections[3])
		x = self.Conv3(x)
		x = self.Conv4(x)
		x = self.TransConv3(x + skip_connections[2])
		x = self.Conv5(x)
		x = self.Conv6(x)
		x = self.TransConv4(x + skip_connections[1])
		x = self.Conv7(x + skip_connections[0])
		x = self.Conv8(x)

		return x


class AutoencoderPoseEstimationModel(nn.Module):
	def __init__(self):
		super(AutoencoderPoseEstimationModel, self).__init__()
		self.Encoder = EncoderModel()
		self.DecoderU = DecoderModel()
		self.DecoderV = DecoderModel()
		self.DecoderW = DecoderModel()

	def forward(self, x):
		feature_vector, skip_connections = self.Encoder(x)
		u_prediction = self.DecoderU(feature_vector, skip_connections)
		v_prediction = self.DecoderV(feature_vector, skip_connections)
		w_prediction = self.DecoderW(feature_vector, skip_connections)

		return u_prediction, v_prediction, w_prediction


if __name__ == "__main__":
	device = "mps" if getattr(torch, 'has_mps', False) else 5 if torch.cuda.is_available() else "cpu"
	print(device)
	ae_model = AutoencoderPoseEstimationModel().to(device)
	model_parameters = filter(lambda p: p.requires_grad, ae_model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("NUMBER PARAMS GENERATOR:", params)
	input_tensor = torch.ones(size=(8, 3, 224, 224)).to(device)
	u = ae_model(input_tensor)
	#print(pose_ref_cnn(input_tensor).shape)
	print(u[0].shape)