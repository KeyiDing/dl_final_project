import torch
import torch.nn as nn
import numpy as np

class BLACKLoss(nn.Module):
    def __init__(self):
        super(BLACKLoss,self).__init__()
        
    def forward(self,target, recon):
        count = (torch.sum(recon, 1) < 0).float() * (torch.sum(target, 1)
                                                          > 0).float()
        return torch.sum(count,(1,2))

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
    
    
class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, target, recon, mask=None):

        assert recon.dim(
        ) == 4, "expected recon dimension to be 4, but instead got {}.".format(
            recon.dim())
        assert target.dim(
        ) == 4, "expected target dimension to be 4, but instead got {}.".format(
            target.dim())
        assert recon.size()==target.size(), "expected recon and target to have the same size, but got {} and {} instead"\
            .format(recon.size(), target.size())
        diff = (target - recon).abs()
        diff = torch.sum(diff, 1)  # sum along the color channel

        # compare only pixels that are not black
        valid_mask = (torch.sum(recon, 1) > 0).float() * (torch.sum(target, 1)
                                                          > 0).float()
        if mask is not None:
            valid_mask = valid_mask * torch.squeeze(mask).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                print(
                    "warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size)."
                )
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :
                                                     -2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:
                                                   -1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()

        self.loss = second_derivative(depth)
        return self.loss