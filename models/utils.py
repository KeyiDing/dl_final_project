import numpy as np
import torch
import open3d as o3d

def depth_rgb_to_pcd(depth, rgb, intrinsics):
    """
    @depth: depth array (B x H x W x 1)
    @rgb: original image (B x H x W x 3)
    @intrinsics: camera intrinsics (3 x 3)

    Returns:
        o3d.t.geometry.PointCloud: pcd
    """
    depth = np.transpose(depth.detach().numpy(), (0,2,3,1)) * 255
    depth_img = o3d.t.geometry.Image(depth[0])
    rgb_img = o3d.t.geometry.Image(rgb.astype(np.float32))
    rgbd_img = o3d.t.geometry.RGBDImage(rgb_img, depth_img)
    
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img,
                intrinsics,
                depth_scale=1.0,
                depth_max=255.0)

    return pcd


def reproject_pcd(pcd, extrinsics):
    """
    @pcd: o3d.t.geometry.PointCloud
    @extrinsics: camera extrinsics

    Returns:

    """

    rgbd_reproj = pcd.project_to_rgbd_image(
                    1242,
                    375,
                    extrinsics,
                    depth_scale=1.0,
                    depth_max=255.0)
    
    return rgbd_reproj
