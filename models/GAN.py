from models.generator import DepthNet, PoseNet
from models.discriminator import Discriminator
from torchvision.utils import save_image
import torch
import numpy as np
import matplotlib.pyplot as plt
import models.utils as utils
from PIL import Image
import os
from torchvision import transforms
import open3d as o3d


 
class DPGAN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.DepthNet = DepthNet()
        self.PoseNet = PoseNet()
        self.Discriminator = Discriminator()
        self.intrinsics = o3d.core.Tensor([[721.5377, 0, 596.5593],
                                           [0, 721.5377, 149.854],
                                           [0, 0, 1]])

    def forward(self, left, center, right):
        depthMap = self.DepthNet(center)
        pose_left = self.PoseNet(left, center)    # pose: (1x6)
        pose_right = self.PoseNet(right, center)    # pose: (1x6)
        pcd = utils.depth_rgb_to_pcd(depthMap, center,self.intrinsics)
        extrinsics_left = utils.pose_to_extrinsics(pose_left)
        extrinsics_right = utils.pose_to_extrinsics(pose_right)
        reproject_left = utils.reproject_pcd(pcd, self.intrinsics, extrinsics_left)
        reproject_right = utils.reproject_pcd(pcd, self.intrinsics, extrinsics_right)

        return reproject_left, reproject_right
    
    def train_discriminator(self, optimizer, criterion, reproject, real):
        optimizer.zero_grad()
        
        prob_fake = self.Discriminator(reproject)
        prob_real = self.Discriminator(real)
        batch_size = len(prob_fake)

        loss_real = criterion(prob_real, torch.ones(batch_size, 1))

        loss_fake = criterion(prob_fake, torch.zeros(batch_size, 1))

        loss_real.backward()
        loss_fake.backward()
        optimizer.step()

        return loss_real + loss_fake
    
    def train_generator(self, optimizer, criterion, reproject):
        optimizer.zero_grad()

        prob = self.Discriminator(reproject)
        batch_size = len(prob)
        loss = criterion(prob, torch.ones(batch_size,1))
 
        loss.backward()
        optimizer.step()

        return loss

    
    def train_model(self, train_loader, k, epochs):
        """
        @train_loader: Training set dataloader
        @k: Train discriminator k times before train generator once
        @epochs: Number of training epochs
        """

        criterion = torch.nn.BCELoss()
        optimizer_g = torch.optim.Adam(list(self.DepthNet.parameters())+list(self.PoseNet.parameters()), lr=1e-2)
        optimizer_d = torch.optim.Adam(self.Discriminator.parameters(), lr=1e-5)

        losses_g = []
        losses_d = []

        for epoch in range(epochs):
            self.DepthNet.train()
            self.PoseNet.train()
            self.Discriminator.train()

            loss_g = 0.0
            loss_d = 0.0
            print(f"Training epoch {epoch} of {epochs}")

            # for i, data in enumerate(train_loader):
            #     left, center, right = data[0], data[1], data[2]
            for i in range(1):
                convert_tensor = transforms.ToTensor()
                left, center, right = convert_tensor(Image.open("./images/2.png")), convert_tensor(Image.open("./images/1.png")), convert_tensor(Image.open("./images/0.png"))
                left = left[None,:,:,:]
                center = center[None,:,:,:]
                right = right[None,:,:,:]

                self.train()
                reproject_left, reproject_right = self(left, center, right)
                loss_d += self.train_discriminator(optimizer_d, criterion, reproject_left, left)
                loss_d += self.train_discriminator(optimizer_d, criterion, reproject_right, right)
                
                for j in range(k):
                    reproject_left, reproject_right = self(left, center, right)
                    loss_g += self.train_generator(optimizer_g, criterion, reproject_left)
                    loss_g += self.train_generator(optimizer_g, criterion, reproject_right)

                depthMap = self.DepthNet(center)
                
                save_image(depthMap, f"./output/depthMap_img{epoch}.png")
                save_image(reproject_left, f"./output/reproject_left_img{epoch}.png")

                epoch_loss_g = loss_g / 1
                epoch_loss_d = loss_d / 1
                for params in self.DepthNet.parameters():
                    print(params.grad)
                    
                losses_g.append(epoch_loss_g)
                losses_d.append(epoch_loss_d)

                print(f"Generator loss: {epoch_loss_g}, Discriminator loss: {epoch_loss_d}")
                print("---------------------------------------------------------")

        # torch.save(self.state_dict())

        print("DONE TRAINING")

        return losses_g, losses_d

                
                








