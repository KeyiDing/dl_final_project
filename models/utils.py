# import numpy as np
import torch
import open3d as o3d
from torchvision import transforms
import torch.utils.dlpack
import open3d.core as o3c

def depth_rgb_to_pcd(depth, rgb, intrinsics):
    """
    @depth: depth array (B x H x W x 1)
    @rgb: original image (B x H x W x 3)
    @intrinsics: camera intrinsics (3 x 3)

    Returns:
        o3d.t.geometry.PointCloud: pcd
    """
    depth = depth.view(1,352,1216,1)*255
    depth = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depth))
    # depth = np.transpose(depth, (0,2,3,1)) * 255

    depth_img = o3d.t.geometry.Image(depth[0])
    # depth_img = o3d.t.geometry.Image(depth[0])
    # rgb_img = o3d.t.geometry.Image(np.ascontiguousarray(np.transpose(rgb[0].detach().numpy(), (1,2,0))))
    # rgb = torch.permute(rgb[0],(1,2,0))
    rgb = rgb[0].view(352,1216,3)
    rgb = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(rgb.contiguous()))
    rgb_img = o3d.t.geometry.Image(rgb)
    rgbd_img = o3d.t.geometry.RGBDImage(rgb_img, depth_img)
    
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img,
                intrinsics,
                depth_scale=1.0,
                depth_max=255.0)
    return pcd


def reproject_pcd(pcd, intrinsics, extrinsics):
    """
    @pcd: o3d.t.geometry.PointCloud
    @extrinsics: camera extrinsics

    Returns:

    """

    rgbd_reproj = pcd.project_to_rgbd_image(
                    1216,
                    352,
                    intrinsics,
                    extrinsics,
                    depth_scale=1.0,
                    depth_max=255.0)
    convert_tensor = transforms.ToTensor()
    # print(type(rgbd_reproj.color.as_tensor()))
    color = rgbd_reproj.color.as_tensor()
    color = torch.utils.dlpack.from_dlpack(color.to_dlpack())
    return color[None,:,:,:]


def pose_to_extrinsics(pose):
    """
    @pose: Bx2x6 matrix, first 3 represent translation, last 3 represent rotation
    
    Return:
    Extrinsic matrix
    """
    # pose = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pose.squeeze()))
    # pose = pose.detach().numpy().squeeze()
    pose = pose.squeeze()

    camera_translation = pose[0:3]
    camera_rotation = pose[3:6]

    pitch = camera_rotation[0]
    yaw = camera_rotation[1]
    roll = camera_rotation[2]

    Rx = torch.tensor([[1, 0, 0],
                [0, torch.cos(pitch), -torch.sin(pitch)],
                [0, torch.sin(pitch), torch.cos(pitch)]])

    Ry = torch.tensor([[torch.cos(yaw), 0, torch.sin(yaw)],
                [0, 1, 0],
                [-torch.sin(yaw), 0, torch.cos(yaw)]])

    Rz = torch.tensor([[torch.cos(roll), -torch.sin(roll), 0],
                [torch.sin(roll), torch.cos(roll), 0],
                [0, 0, 1]])

    R = Rz @ Ry @ Rx

    extrinsic_matrix = torch.tensor([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = -camera_translation
    extrinsic_matrix = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(extrinsic_matrix))

    return extrinsic_matrix