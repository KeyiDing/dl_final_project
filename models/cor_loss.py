import torch
import torch.nn as nn
import numpy as np

class CORLoss(nn.Module):
    def __init__(self):
        super(CORLoss,self).__init__()

    
    def forward(self,grid):
        # (b, h, w, 2)
        b = grid.shape[0]
        h = grid.shape[1]
        w = grid.shape[2]
        
        origin_grid = grid.clone()
        origin_grid[:,:,:,0] = (origin_grid[:,:,:,0]+1)*(w-1)/2
        origin_grid[:,:,:,1] = (origin_grid[:,:,:,1]+1)*(h-1)/2
        
        ref = torch.empty(h,w,2)
        x_ref = torch.tensor(np.arange(w))
        y_ref = torch.tensor(np.arange(h))
        
        x_ref = x_ref.repeat(h,1)
        y_ref = y_ref.repeat(w,1)
        y_ref = torch.transpose(y_ref,0,1)
        ref[:,:,0] = x_ref
        ref[:,:,1] = y_ref

        
        ref = ref.expand(b,h,w,2)
        return nn.functional.mse_loss(origin_grid,ref)