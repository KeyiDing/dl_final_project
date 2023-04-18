import numpy as np
import torch

def conv(in_size, out_size, kernel_size=3, padding=0):
	return torch.nn.Sequential(
		torch.nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=2, padding=padding),
		torch.nn.LeakyReLU(inplace=False),
		torch.nn.BatchNorm2d(num_features=out_size)
	)

def up_conv(in_size, out_size, kernel_size=3, padding=0, output_padding=0):
	return torch.nn.Sequential(
		torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding),
		torch.nn.LeakyReLU(inplace=False),
		torch.nn.BatchNorm2d(num_features=out_size)
	)

class Encoder(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv1 = conv(in_size=3, out_size=32, padding=1)
		self.conv2 = conv(in_size=32, out_size=64, padding=1)
		self.conv3 = conv(in_size=64, out_size=128, padding=1)
		self.conv4 = conv(in_size=128, out_size=256, padding=1)
		self.conv5 = conv(in_size=256, out_size=512, padding=1)
		self.conv6 = conv(in_size=512, out_size=1024, padding=1)

		
	def forward(self, x):
		"""
		Input: Image: (B, 3, H, W)
		Output: Feature Maps: (B, 512, H//32, W//32)
		"""
		x = self.conv1(x)
		print(x.shape)
		x = self.conv2(x)
		print(x.shape)
		x = self.conv3(x)
		print(x.shape)
		x = self.conv4(x)
		print(x.shape)
		x = self.conv5(x)
		print(x.shape)
		x = self.conv6(x)
		print(x.shape)

		return x
	
class Decoder(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv1 = up_conv(in_size=1024, out_size=524, padding=1, output_padding=(1,0))
		self.conv2 = up_conv(in_size=524, out_size=256, padding=1, output_padding=(1,1))
		self.conv3 = up_conv(in_size=256, out_size=128, padding=1, output_padding=(0,1))
		self.conv4 = up_conv(in_size=128, out_size=64, padding=1, output_padding=(1,0))
		self.conv5 = up_conv(in_size=64, out_size=32, padding=1, output_padding=(1,0))
		self.conv6 = up_conv(in_size=32, out_size=1, padding=1, output_padding=(0,1))
		self.out = torch.nn.Sigmoid()

	def forward(self, x):
		"""
		Input: Z feature: (B, 1024, H, W)
		Output: Feature Maps: (B, 1, H, W)
		"""
		x = self.conv1(x)
		print(x.shape)
		x = self.conv2(x)
		print(x.shape)
		x = self.conv3(x)
		print(x.shape)
		x = self.conv4(x)
		print(x.shape)
		x = self.conv5(x)
		print(x.shape)
		x = self.conv6(x)
		print(x.shape)
		x = self.out(x)
		print(x.shape)

		return x
	
class DepthNet(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = Encoder()
		self.decoder = Decoder()
		self.out = torch.nn.Sigmoid()
	
	def forward(self, x):
		x = self.encoder(x)
		print("encoder output shape:", x.shape)
		x = self.decoder(x)
		print("decoder output shape:", x.shape)

		return x
	
class PoseNet(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = Encoder()
		

