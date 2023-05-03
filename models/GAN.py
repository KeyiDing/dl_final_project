from models.generator import DepthNet, PoseNet, PoseCNN
from models.discriminator import Discriminator
from models.wrap import inverse_warp
from models.cor_loss import CORLoss, PhotometricLoss, SmoothnessLoss, BLACKLoss
from torchvision.utils import save_image
import torch
import numpy as np
import matplotlib.pyplot as plt
# import models.utils as utils
from PIL import Image
import os
from torchvision import transforms
import shutil
from pathlib import Path
import pandas as pd
import models.utils as utils

# import open3d as o3d


 
class DPGAN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.DepthNet = DepthNet()
        self.PoseNet = PoseCNN(2)
        self.Discriminator = Discriminator()
        self.intrinsics = torch.tensor([[721.5377, 0, 596.5593],
                                           [0, 721.5377, 149.854],
                                           [0, 0, 1]])
        # self.intrinsics = torch.tensor([[721.5377/1242, 0, 596.5593/1242],
        #                                    [0, 721.5377/375, 149.854/375],
        #                                    [0, 0, 1]])

    def forward(self, left, center, right):
        depthMap = self.DepthNet(center)
        pose_left = self.PoseNet(left, center)    # pose: (1x6)
        pose_right = self.PoseNet(right, center)    # pose: (1x6)
        
        print(pose_left)
        
        # pose_left = torch.tensor([[0., 0, 25, 0,0,0]])
        # pose_right = torch.tensor([[0., 0, 50, 0,0,0]])
        # pcd = utils.depth_rgb_to_pcd(depthMap, center,self.intrinsics)
        # extrinsics_left = utils.pose_to_extrinsics(pose_left)
        # extrinsics_right = utils.pose_to_extrinsics(pose_right)
        # reproject_left = utils.reproject_pcd(pcd, self.intrinsics, extrinsics_left)
        # reproject_right = utils.reproject_pcd(pcd, self.intrinsics, extrinsics_right)
        reproject_left, valid_points, grid_left = inverse_warp(center, depthMap[:,0], pose_left,
                                                             self.intrinsics,
                                                             rotation_mode='euler', padding_mode='zeros')
        
        reproject_right, valid_points, grid_right = inverse_warp(center, depthMap[:,0], pose_right,
                                                             self.intrinsics,
                                                             rotation_mode='euler', padding_mode='zeros')

        return reproject_left, reproject_right, grid_left, grid_right
    
    def train_discriminator(self, optimizer, criterion, reproject, real):
        optimizer.zero_grad()
        
        prob_fake = self.Discriminator(reproject)
        prob_real = self.Discriminator(real)
        batch_size = len(prob_fake)

        loss_real = criterion(prob_real, torch.ones(batch_size, 1))

        loss_fake = criterion(prob_fake, torch.zeros(batch_size, 1))

        loss_real.backward()
        loss_fake.backward(retain_graph=True)
        optimizer.step()

        return loss_real + loss_fake
    
    def train_generator(self, optimizer1,optimizer2, l1_loss, black_loss, photo_loss, smooth_loss, left, right, reproject_left, reproject_right, grid_left, grid_right, depth):
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # prob = self.Discriminator(reproject_left)
        # batch_size = len(prob)
        # loss1 = criterion(prob, torch.ones(batch_size,1))


        # prob = self.Discriminator(reproject_right)
        # batch_size = len(prob)
        # loss2 = criterion(prob, torch.ones(batch_size,1))
        
        # loss3 = black_loss(left,reproject_left)
        # loss4 = black_loss(right,reproject_right)
        
        # loss5 = photo_loss(left,reproject_left)
        # loss6 = photo_loss(right,reproject_right)
        loss7 = smooth_loss(depth)
        # loss = loss1 + loss2 + 100*loss5 + 100*loss6 + loss7
        # loss = loss1 + loss2 + loss5 + loss6 + loss7
        # print(loss5, loss6, loss7)

        loss1 = photo_loss(left, reproject_left)
        loss2 = photo_loss(right, reproject_right)

        loss = loss1 + loss2
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        return loss

    
    def train_model(self, train_loader, k, epochs):
        """
        @train_loader: Training set dataloader
        @k: Train discriminator k times before train generator once
        @epochs: Number of training epochs
        """

        criterion = torch.nn.BCELoss()
        optimizer_depth = torch.optim.Adam(self.DepthNet.parameters(), lr=1e-5)
        optimizer_pose = torch.optim.Adam(self.PoseNet.parameters(), lr=1e-5)
        optimizer_d = torch.optim.Adam(self.Discriminator.parameters(), lr=1e-4)
        cor_loss = CORLoss()
        photo_loss = PhotometricLoss()
        smooth_loss = SmoothnessLoss()
        black_loss = BLACKLoss()
        l1_loss = torch.nn.L1Loss()

        losses_g = []
        losses_d = []
        
        if os.path.exists('./output'):
            shutil.rmtree('./output')
        Path('./output').mkdir(exist_ok=True)

        for epoch in range(epochs):
            
            self.DepthNet.train()
            self.PoseNet.train()
            self.Discriminator.train()

            # convert_tensor = transforms.ToTensor()
            convert_tensor = transforms.Compose([
                transforms.Resize((352,1216)),
                transforms.ToTensor(),
            ])
            left, center, right = convert_tensor(Image.open("./images/0.png")), convert_tensor(Image.open("./images/1.png")), convert_tensor(Image.open("./images/2.png"))
            center = center[None,:,:,:]
            depthMap = self.DepthNet(center)

            utils.save_img(depthMap, "start")

            loss_g = 0.0
            loss_d = 0.0
            print(f"Training epoch {epoch} of {epochs}")

            # for i, data in enumerate(train_loader):
            #     left, center, right = data[0], data[1], data[2]
            for i in range(1):
                
                left, center, right = convert_tensor(Image.open("./images/0.png")), convert_tensor(Image.open("./images/1.png")), convert_tensor(Image.open("./images/2.png"))
                left = left[None,:,:,:]
                center = center[None,:,:,:]
                right = right[None,:,:,:]

                self.train()
                reproject_left, reproject_right, grid_left, grid_right = self(left, center, right)
                utils.save_img(reproject_left, f"reproject_left_{epoch}")
                utils.save_img(left, "left")
                
                loss_d += self.train_discriminator(optimizer_d, criterion, reproject_left, left)
                loss_d += self.train_discriminator(optimizer_d, criterion, reproject_right, right)
                

                
                for j in range(k):
                    depthMap = self.DepthNet(center)
                    reproject_left, reproject_right,grid_left,grid_right = self(left, center, right)
                    loss_g += self.train_generator(optimizer_depth, optimizer_pose, l1_loss, black_loss, photo_loss, smooth_loss, left, right, reproject_left, reproject_right, grid_left, grid_right, depthMap)
                    # loss_g += self.train_generator(optimizer_g, criterion, reproject_right)

                depthMap = self.DepthNet(center)
                print(torch.mean(depthMap), torch.std(depthMap))
                
                utils.save_img(depthMap, epoch)
                save_image(reproject_left[0], f"./output/reproject_left_img{epoch}.png")
                save_image(reproject_right[0], f"./output/reproject_right_img{epoch}.png")

                epoch_loss_g = loss_g / 1
                epoch_loss_d = loss_d / 1
                
                # for params in self.DepthNet.parameters():
                #     print(params.grad)
                    
                # for params in self.PoseNet.parameters():
                #     print(params.grad)
                    
                losses_g.append(epoch_loss_g)
                losses_d.append(epoch_loss_d)

                print(f"Generator loss: {epoch_loss_g}, Discriminator loss: {epoch_loss_d}")
                print("---------------------------------------------------------")

        # torch.save(self.state_dict())

        print("DONE TRAINING")

        return losses_g, losses_d

                
                








