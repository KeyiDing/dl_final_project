import torch
import numpy as np

def conv(in_size, out_size, kernel_size=3, padding=1):
    	return torch.nn.Sequential(
		torch.nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=2, padding=padding),
        torch.nn.BatchNorm2d(num_features=out_size),
		torch.nn.LeakyReLU(inplace=False),
	)

def downsample_conv(in_size, out_size, kernel_size=3, padding=1):
	return torch.nn.Sequential(
		torch.nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=2, padding=padding),
        torch.nn.BatchNorm2d(num_features=out_size),
		torch.nn.LeakyReLU(inplace=False),
	)
 
 
class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        
        super(Discriminator,self).__init__()
        self.conv1 = conv(in_size=3, out_size=6, kernel_size=3)
        self.conv2 = downsample_conv(in_size=6, out_size=128, kernel_size=3)
        self.conv3 = downsample_conv(in_size=128, out_size=256, kernel_size=3)
        self.conv4 = downsample_conv(in_size=256, out_size=512, kernel_size=3)
        self.conv5 = downsample_conv(in_size=512, out_size=1024, kernel_size=3)
        self.linear = torch.nn.Linear(479232,1)
        self.sig = torch.nn.Sigmoid()
        
    def forward(self,x):
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
        
        x = self.linear(x.view(x.shape[0], -1))
        x = self.sig(x)
        return x
    