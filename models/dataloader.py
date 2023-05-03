import os
import torch  
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms

img_transform = transforms.Compose([
        transforms.ToTensor(),
])

class DepthDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images) // 3 - 1
    
    def __getitem__(self, idx):
        idx += 1
        left = Image.open(os.path.join(self.root_dir, self.images[idx]))
        middle = Image.open(os.path.join(self.root_dir, self.images[idx + 1]))
        right = Image.open(os.path.join(self.root_dir, self.images[idx + 2]))
        data = {"left": self.img_transform(np.array(left)), "middle":  self.img_transform(np.array(middle)), "right": self.img_transform(np.array(right))}
        return data
    
    def img_transform(self, img):
        img = self.transforms(img)
        return img
    