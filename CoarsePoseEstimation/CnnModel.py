from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)

from Utils.ConvUtils import *


class EncoderModel(nn.Module):
	def __init__(self):
		super(EncoderModel, self).__init__()
		self.layer_channels = [1, 64, 96, 128, 192, 256]

		self.Conv0 = ConvBnActiv(in_channels=self.layer_channels[0], out_channels=self.layer_channels[1], kernel_size=7, padding=3)

		self.Conv1 = ConvBnActiv(in_channels=self.layer_channels[1], out_channels=self.layer_channels[2], stride=2)
		self.ResLayer1 = ResidualBlock(in_channels=self.layer_channels[2], out_channels=self.layer_channels[2])
		self.ResLayer2 = ResidualBlock(in_channels=self.layer_channels[2], out_channels=self.layer_channels[2])

		self.Conv2 = ConvBnActiv(in_channels=self.layer_channels[2], out_channels=self.layer_channels[3], stride=2)
		self.ResLayer3 = ResidualBlock(in_channels=self.layer_channels[3], out_channels=self.layer_channels[3])
		self.ResLayer4 = ResidualBlock(in_channels=self.layer_channels[3], out_channels=self.layer_channels[3])

		self.Conv3 = ConvBnActiv(in_channels=self.layer_channels[3], out_channels=self.layer_channels[4], stride=2)
		self.ResLayer5 = ResidualBlock(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])
		self.ResLayer6 = ResidualBlock(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])

		self.Conv4 = ConvBnActiv(in_channels=self.layer_channels[4], out_channels=self.layer_channels[5], stride=2)
		self.DilatedConv1 = ConvBnActiv(in_channels=self.layer_channels[5], out_channels=self.layer_channels[5], dilation_rate=2, padding=2)
		self.DilatedConv2 = ConvBnActiv(in_channels=self.layer_channels[5], out_channels=self.layer_channels[5], dilation_rate=4, padding=4)

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
		self.channel = [256, 192, 128, 96, 64, 4]

		self.TransConv1 = TransposeConvBnActiv(in_channels=self.channel[0], out_channels=self.channel[1])
		self.ResLayer1 = ResidualBlock(in_channels=self.channel[1], out_channels=self.channel[1])
		self.ResLayer2 = ResidualBlock(in_channels=self.channel[1], out_channels=self.channel[1])
		self.TransConv2 = TransposeConvBnActiv(in_channels=self.channel[1], out_channels=self.channel[2])
		self.ResLayer3 = ResidualBlock(in_channels=self.channel[2], out_channels=self.channel[2])
		self.ResLayer4 = ResidualBlock(in_channels=self.channel[2], out_channels=self.channel[2])
		self.TransConv3 = TransposeConvBnActiv(in_channels=self.channel[2], out_channels=self.channel[3])
		self.ResLayer5 = ResidualBlock(in_channels=self.channel[3], out_channels=self.channel[3])
		self.ResLayer6 = ResidualBlock(in_channels=self.channel[3], out_channels=self.channel[3])
		self.TransConv4 = TransposeConvBnActiv(in_channels=self.channel[3], out_channels=self.channel[4])
		self.Conv7 = ConvBnActiv(in_channels=self.channel[4], out_channels=self.channel[4])
		self.Conv8 = ConvBnActiv(in_channels=self.channel[4], out_channels=self.channel[4])
		self.Conv9 = ConvBnActiv(in_channels=self.channel[4], out_channels=self.channel[5], kernel_size=1, stride=1, padding=0, apply_activation=False, apply_bias=False, apply_bn=False)

		self.Sigmoid = nn.Sigmoid()

	def forward(self, x, skip_connections):
		x = self.TransConv1(x)
		x = self.ResLayer1(x)
		x = self.ResLayer2(x)
		x = self.TransConv2(x + skip_connections[3])
		x = self.ResLayer3(x)
		x = self.ResLayer4(x)
		x = self.TransConv3(x + skip_connections[2])
		x = self.ResLayer5(x)
		x = self.ResLayer6(x)
		x = self.TransConv4(x + skip_connections[1])
		x = self.Conv7(x + skip_connections[0])
		x = self.Conv8(x)
		x = self.Conv9(x)
		x = self.Sigmoid(x)
		return x


class AutoencoderPoseEstimationModel(nn.Module):
	def __init__(self):
		super(AutoencoderPoseEstimationModel, self).__init__()
		self.Encoder = EncoderModel()
		self.Decoder = DecoderModel()

	def forward(self, x):
		feature_vector, skip_connections = self.Encoder(x)
		prediction = self.Decoder(feature_vector, skip_connections)
		#v_prediction = self.DecoderV(feature_vector, skip_connections)
		#w_prediction = self.DecoderW(feature_vector, skip_connections)
		u_prediction = prediction[:, 0:1, ...]
		v_prediction = prediction[:, 1:2, ...]
		w_prediction = prediction[:, 2:3, ...]
		binary_mask = prediction[:, 3:4, ...]
		return u_prediction, v_prediction, w_prediction, binary_mask


if __name__ == "__main__":
	device = "mps" if getattr(torch, 'has_mps', False) else 5 if torch.cuda.is_available() else "cpu"
	print(device)
	ae_model = AutoencoderPoseEstimationModel().to(device)
	model_parameters = filter(lambda p: p.requires_grad, ae_model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("NUMBER PARAMS GENERATOR:", params)
	input_tensor = torch.ones(size=(8, 1, 224, 224)).to(device)
	u = ae_model(input_tensor)
	#print(pose_ref_cnn(input_tensor).shape)
	print(u[0].shape)