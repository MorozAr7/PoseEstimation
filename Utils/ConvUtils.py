import torch
from torch import nn
import numpy as np


class TransposeConvBnActiv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel=3, stride=2, in_padding=(1, 1), out_padding=(1, 1), apply_bn=True, apply_activation=True):
		super(TransposeConvBnActiv, self).__init__()
		self.apply_relu = apply_activation
		self.apply_bn = apply_bn
		self.TransposeConv = nn.ConvTranspose2d(in_channels=in_channels,
		                                        out_channels=out_channels,
		                                        kernel_size=kernel,
		                                        stride=stride,
		                                        padding=in_padding,
		                                        output_padding=out_padding,
		                                        bias=not apply_bn)
		if self.apply_bn:
			self.BN = nn.BatchNorm2d(num_features=out_channels)
		if self.apply_relu:
			self.Activation = nn.SiLU(inplace=True)

	def forward(self, x):

		if self.apply_bn and self.apply_relu:
			return self.Activation(self.BN(self.TransposeConv(x)))
		elif self.apply_relu and not self.apply_bn:
			return self.Activation(self.TransposeConv(x))
		elif self.apply_bn and not self.apply_relu:
			return self.BN(self.TransposeConv(x))
		else:
			return self.TransposeConv(x)


class ConvBnActiv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation_rate=1, apply_bn=True, apply_activation=True, apply_bias=True, activ_type="silu"):
		super(ConvBnActiv, self).__init__()

		self.apply_relu = apply_activation
		self.apply_bn = apply_bn
  
		self.Conv = nn.Conv2d(in_channels=in_channels,
		                      out_channels=out_channels,
		                      kernel_size=kernel_size,
		                      stride=stride,
		                      padding=padding,#kernel_size // 2,
		                      groups=groups,
		                      bias=not apply_bn and apply_bias,
		                      dilation=dilation_rate)
  
		if apply_bn:
			self.BN = nn.BatchNorm2d(num_features=out_channels)
		if apply_activation:
			self.Activation = nn.SiLU(inplace=True) if activ_type == "silu" else nn.PReLU(num_parameters=out_channels)

	def forward(self, x):

		if self.apply_bn and self.apply_relu:
			return self.Activation(self.BN(self.Conv(x)))
		elif not self.apply_bn and self.apply_relu:
			return self.Activation(self.Conv(x))
		elif self.apply_bn and not self.apply_relu:
			return self.BN(self.Conv(x))
		elif not self.apply_bn and not self.apply_relu:
			return self.Conv(x)


class DepthWiseConvResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1, dilation_rate=1, apply_activ=False, use_skip=True):
		super(DepthWiseConvResidualBlock, self).__init__()
		self.apply_activ = apply_activ
		self.use_skip = use_skip
		self.expansion_convolution = ConvBnActiv(in_channels=in_channels,
		                                        out_channels=in_channels * expansion_factor,
		                                        kernel_size=1,
		                                        padding=0)
		self.projection_convolution = ConvBnActiv(in_channels=in_channels * expansion_factor,
		                                         out_channels=out_channels,
		                                         kernel_size=1,
		                                         padding=0,
		                                         apply_activation=False)
		self.depth_wise_convolution = ConvBnActiv(in_channels=in_channels * expansion_factor,
		                                         out_channels=in_channels * expansion_factor,
		                                         stride=stride,
		                                         groups=in_channels * expansion_factor,
		                                         dilation_rate=dilation_rate,
		                                         padding=dilation_rate
		                                         )
		self.apply_1x1_conv = stride != 1 or in_channels != out_channels
		if self.apply_1x1_conv:
			self.residual_convolution = ConvBnActiv(in_channels=in_channels,
			                                       out_channels=out_channels,
			                                       kernel_size=1,
			                                       padding=0,
			                                       stride=stride)
		if apply_activ:
			self.Activation = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x
		x = self.expansion_convolution(x)
		x = self.depth_wise_convolution(x)
		x = self.projection_convolution(x)
  
		if self.apply_activ:
			return self.Activation(x + residual)
		else:
			return x + residual


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, apply_activ=True):
		super(ResidualBlock, self).__init__()
		self.apply_activ = apply_activ
  
		self.Conv1 = ConvBnActiv(in_channels=in_channels, out_channels=out_channels)
		self.Conv2 = ConvBnActiv(in_channels=in_channels, out_channels=out_channels, apply_activation=False)
  
		if self.apply_activ:
			self.Activation = nn.SiLU(inplace=True)

	def forward(self, x):
		residual = x
		x = self.Conv1(x)
		x = self.Conv2(x)
		if self.apply_activ:
			return self.Activation(x + residual)
		else:
			return x + residual


def init_weights(m) -> None:
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
		torch.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.BatchNorm2d]:
		torch.nn.init.normal_(m.weight.data, 1.0, 2.0)
		torch.nn.init.constant_(m.bias.data, 0)


def change_learning_rate(optimizer, epoch, epochs_list, decay_factor) -> None:
	if epoch in epochs_list:
		optimizer.param_groups[0]["lr"] /= decay_factor
