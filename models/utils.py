import numpy as np
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
    depth = torch.permute(depth,(0,2,3,1))*255
    depth = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(depth))
    print(type(depth))
    # depth = np.transpose(depth, (0,2,3,1)) * 255

    depth_img = o3d.t.geometry.Image(depth[0])
    print("depth done")
    # depth_img = o3d.t.geometry.Image(depth[0])
    # rgb_img = o3d.t.geometry.Image(np.ascontiguousarray(np.transpose(rgb[0].detach().numpy(), (1,2,0))))
    rgb = torch.permute(rgb[0],(1,2,0))
    rgb = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(rgb.contiguous()))
    rgb_img = o3d.t.geometry.Image(rgb)
    rgbd_img = o3d.t.geometry.RGBDImage(rgb_img, depth_img)
    print("rgbd done")
    
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img,
                intrinsics,
                depth_scale=1.0,
                depth_max=255.0)
    print("pcd done")
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
    return convert_tensor(np.asarray(rgbd_reproj.color))[None,:,:,:]


def pose_to_extrinsics(pose):
    """
    @pose: Bx2x6 matrix, first 3 represent translation, last 3 represent rotation
    
    Return:
    Extrinsic matrix
    """
    # pose = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pose.squeeze()))
    # pose = pose.detach().numpy().squeeze()
    pose = pose.squeeze()

    camera_translation = np.array(pose[0:3])
    camera_rotation = np.array(pose[3:6])

    pitch = camera_rotation[0]
    yaw = camera_rotation[1]
    roll = camera_rotation[2]

    Rx = np.array([[1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]])

    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]])

    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1]])

    R = Rz @ Ry @ Rx

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = -camera_translation

    return extrinsic_matrix