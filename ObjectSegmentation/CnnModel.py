from CONFIG import *
#sys.path.insert(0, MAIN_DIR_PATH)
from Utils.ConvUtils import *


class UnetSegmentation(nn.Module):
	def __init__(self):
		super(UnetSegmentation, self).__init__()

		channels = [3, 32, 64, 128, 192, 256, 1]
		self.Conv0 = ConvBnActiv(in_channels=channels[0], out_channels=channels[1])
		self.Conv1 = ConvBnActiv(in_channels=channels[1], out_channels=channels[2], stride=2)
		self.ResLayer1 = ConvBnActiv(in_channels=channels[2], out_channels=channels[2])
		self.ResLayer2 = ConvBnActiv(in_channels=channels[2], out_channels=channels[2])
		self.Conv2 = ConvBnActiv(in_channels=channels[2], out_channels=channels[3], stride=2)
		self.ResLayer3 = ConvBnActiv(in_channels=channels[3], out_channels=channels[3])
		self.ResLayer4 = ConvBnActiv(in_channels=channels[3], out_channels=channels[3])
		self.Conv3 = ConvBnActiv(in_channels=channels[3], out_channels=channels[4], stride=2)
		self.ResLayer5 = ConvBnActiv(in_channels=channels[4], out_channels=channels[4])
		self.ResLayer6 = ConvBnActiv(in_channels=channels[4], out_channels=channels[4])
		self.Conv4 = ConvBnActiv(in_channels=channels[4], out_channels=channels[5], stride=2)
		self.ResLayer7 = ConvBnActiv(in_channels=channels[5], out_channels=channels[5], dilation_rate=2, padding=2)
		self.ResLayer8 = ConvBnActiv(in_channels=channels[5], out_channels=channels[5], dilation_rate=4, padding=4)

		self.TransConv1 = TransposeConvBnActiv(in_channels=channels[5], out_channels=channels[4])
		self.ResLayer9 = ConvBnActiv(in_channels=channels[4], out_channels=channels[4])
		self.ResLayer10 = ConvBnActiv(in_channels=channels[4], out_channels=channels[4])
		self.TransConv2 = TransposeConvBnActiv(in_channels=channels[4], out_channels=channels[3])
		self.ResLayer11 = ConvBnActiv(in_channels=channels[3], out_channels=channels[3])
		self.ResLayer12 = ConvBnActiv(in_channels=channels[3], out_channels=channels[3])
		self.TransConv3 = TransposeConvBnActiv(in_channels=channels[3], out_channels=channels[2])
		self.ResLayer13 = ConvBnActiv(in_channels=channels[2], out_channels=channels[2])
		self.ResLayer14 = ConvBnActiv(in_channels=channels[2], out_channels=channels[2])
		self.TransConv4 = TransposeConvBnActiv(in_channels=channels[2], out_channels=channels[1])
		self.Conv5 = ConvBnActiv(in_channels=channels[1], out_channels=channels[1])
		self.ConvOut = ConvBnActiv(in_channels=channels[1], out_channels=channels[-1], kernel_size=1, padding=0, apply_bn=False, apply_activation=False, apply_bias=False)

		self.Sigmoid = nn.Sigmoid()

	def forward(self, x):
		skip0 = self.Conv0(x)
		skip1 = self.Conv1(skip0)

		x = self.ResLayer1(skip1)
		x = self.ResLayer2(x)
		skip2 = self.Conv2(x)

		x = self.ResLayer3(skip2)
		x = self.ResLayer4(x)
		skip3 = self.Conv3(x)

		x = self.ResLayer5(skip3)
		x = self.ResLayer6(x)
		x = self.Conv4(x)

		x = self.ResLayer7(x)

		x = self.ResLayer8(x)

		x = self.TransConv1(x)
		x = self.ResLayer9(x)
		x = self.ResLayer10(x)

		x = self.TransConv2(x + skip3)
		x = self.ResLayer11(x)
		x = self.ResLayer12(x)
		x = self.TransConv3(x + skip2)
		x = self.ResLayer13(x)
		x = self.ResLayer14(x)
		x = self.TransConv4(x + skip1)

		x = self.Conv5(x + skip0)
		x = self.ConvOut(x)

		x = self.Sigmoid(x)

		return x


if __name__ == "__main__":
	x = torch.zeros(size=(2, 3, 224, 224)).to(DEVICE)

	model = UnetSegmentation().to(DEVICE)

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print("NUMBER PARAMS GENERATOR:", params)
	print(DEVICE)
	out = model(x)
