from generator import DepthNet, PoseNet
from discriminator import Discriminator
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils

 
class DPGAN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.DepthNet = DepthNet()
        self.PoseNet = PoseNet()
        self.Discriminator = Discriminator()

    def forward(self, left, center, right):
        depthMap = self.DepthNet(center)
        pose_left = self.PoseNet(left, center)    # pose: (1x6)
        pose_right = self.PoseNet(right, center)    # pose: (1x6)
        pcd = utils.depth_rgb_to_pcd(depthMap, center)
        extrinsics_left = utils.pose_to_extrinsics(pose_left)
        extrinsics_right = utils.pose_to_extrinsics(pose_right)
        reproject_left = utils.reproject_pcd(pcd, extrinsics_left)
        reproject_right = utils.reproject_pcd(pcd, extrinsics_right)

        output_fake_left = self.Discriminator(reproject_left)
        output_fake_right = self.Discriminator(reproject_right)

        return output_fake_left, output_fake_right
    
    def train(self, train_loader):
        for i, data in enumerate(train_loader):
            left, center, right = data[0], data[1], data[2]
             self()






