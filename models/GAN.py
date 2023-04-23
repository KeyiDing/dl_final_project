from generator import DepthNet, PoseNet
from discriminator import Discriminator
from torchvision.utils import save_image
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
from PIL import Image


 
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

        return reproject_left, reproject_right
    
    def train_discriminator(self, optimizer, criterion, reproject, real):
        optimizer.zero_grad()
        
        prob_fake = self.Discriminator(reproject)
        prob_real = self.Discriminator(real)

        loss_real = criterion(prob_real, torch.ones(1))

        loss_fake = criterion(prob_fake, torch.zeros(1))

        loss_real.backward()
        loss_fake.backward()
        optimizer.step()

        return loss_real + loss_fake
    
    def train_generator(self, optimizer, criterion, reproject):
        optimizer.zero_grad()

        prob = self.Discriminator(reproject)
        loss = criterion(prob, torch.ones(1))

        loss.backward()
        optimizer.step()

        return loss

    
    def train(self, train_loader, k, epochs):
        """
        @train_loader: Training set dataloader
        @k: Train discriminator k times before train generator once
        @epochs: Number of training epochs
        """

        criterion = torch.nn.BCELoss()
        optimizer_g = torch.optim.Adam(list(self.DepthNet.parameters())+list(self.PoseNet.parameters()), lr=1e-3)
        optimizer_d = torch.optim.Adam(self.Discriminator.parameters(), lr=1e-3)

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
                left, center, right = Image.open("../images/2.png"), Image.open("../images/1.png"), Image.open("../images/0.png")
                for j in range(k):
                    reproject_left, reproject_right = self(left, center, right)

                    loss_d += self.train_discriminator(optimizer_d, criterion, reproject_left, left)
                    loss_d += self.train_discriminator(optimizer_d, criterion, reproject_right, right)
                
                reproject_left, reproject_right = self(left, center, right)
                loss_g += self.train_generator(optimizer_g, criterion, reproject_left, reproject_right)

                self.DepthNet.eval()

                depthMap = self.DepthNet(center)
                save_image(depthMap, f"../output/depthMap_img{epoch}.png")

                epoch_loss_g = loss_g / i
                epoch_loss_d = loss_d / i
                losses_g.append(epoch_loss_g)
                losses_d.append(epoch_loss_d)

                print(f"Generator loss: {epoch_loss_g}, Discriminator loss: {epoch_loss_d}")
                print("---------------------------------------------------------")

        torch.save(self.state_dict())

        print("DONE TRAINING")

        return losses_g, losses_d

                
                








