import os
import torch  

from torch.utils.data import Dataset
from skimage import io

class DepthDataset(Dataset):
    def __init__(self, input_dir, op, transforms=None):
        self.transforms = transforms
        self.op = op
        try:
            if self.op == 'train':
                self.data_dir = os.path.join(input_dir, 'train')
            elif self.op == 'val':
                self.data_dir = os.path.join(input_dir, 'validation')
            elif self.op == 'test':
                self.data_dir = os.path.join(input_dir, 'test')
        except ValueError:
            print('op should be either train, val or test!')
        
    def __len__(self):
        return len(next(os.walk(self.data_dir))[1])
    
    def __getitem__(self, idx):
        img_name = str(idx) + '_input.jpg'
        img = io.imread(os.path.join(self.data_dir, str(idx), img_name))
        return img
    
    def img_transform(self, img):
        img = self.transforms(img)
        return img
    