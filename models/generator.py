import numpy as np
import torch
from models.depth_decoder import DepthDecoder
from models.resnet_encoder import ResnetEncoder

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
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)

		return x
	
class Decoder(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.conv1 = up_conv(in_size=1024, out_size=524, padding=1, output_padding=(0,1))
		self.conv2 = up_conv(in_size=524, out_size=256, padding=1, output_padding=(1,1))
		self.conv3 = up_conv(in_size=256, out_size=128, padding=1, output_padding=(1,1))
		self.conv4 = up_conv(in_size=128, out_size=64, padding=1, output_padding=(1,1))
		self.conv5 = up_conv(in_size=64, out_size=32, padding=1, output_padding=(1,1))
		self.conv6 = up_conv(in_size=32, out_size=1, padding=1, output_padding=(1,1))
		self.out = torch.nn.Sigmoid()

	def forward(self, x):
		"""
		Input: Z feature: (B, 1024, H, W)
		Output: Feature Maps: (B, 1, H, W)
		"""
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.out(x)
		
		return x
	
class DepthNet(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = ResnetEncoder(50, True)
		self.decoder = DepthDecoder(self.encoder.num_ch_enc)
	
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		# print('nw min',torch.min(x* 255))

		return x[("disp", 0)]*255
	
class PoseNet(torch.nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.encoder = Encoder()
		self.encoder.conv1 = conv(in_size=6, out_size=32, padding=1)
		self.linear1 = torch.nn.Linear(in_features=116736, out_features=512)
		torch.nn.init.xavier_uniform(self.linear1.weight)
		self.linear2 = torch.nn.Linear(in_features=512, out_features=128)
		torch.nn.init.xavier_uniform(self.linear2.weight)
		self.linear3 = torch.nn.Linear(in_features=128, out_features=64)
		torch.nn.init.xavier_uniform(self.linear3.weight)
		self.linear4 = torch.nn.Linear(in_features=64, out_features=6)
		torch.nn.init.xavier_uniform(self.linear4.weight)
		self.activation = torch.nn.LeakyReLU()
		self.activation1 = torch.nn.Hardtanh(min_val=-np.pi,max_val=np.pi)
  		
	def forward(self, other,center):
		x = torch.cat([other,center],dim=1)
		x1 = self.encoder(x)
		x1 = x1.view(x.shape[0], -1)
		x2 = self.linear1(x1)
		x2 = self.activation(x2)
		x3= self.linear2(x2)
		x4 = self.linear3(x3)
		x5 = self.linear4(x4)
		x6 = torch.ones(x5.shape)
		x6[:,3:6] = self.activation1(x5[:,3:6])
		x6[:,0:3] = self.activation1(x5[:,0:3])
		print('x6',x6)

		# return self.activation(x5)
		return x6

