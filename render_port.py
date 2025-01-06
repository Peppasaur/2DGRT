import torch
import numpy as np
from typing import NamedTuple
from scene import GaussianModel

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

def quaternion_to_rotation_matrix_torch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a sequence of quaternions to rotation matrices (PyTorch version).
    
    Parameters:
        quaternions (torch.Tensor): Input tensor of shape [n, 4], where each row is [w, x, y, z].
    
    Returns:
        torch.Tensor: Output tensor of shape [n, 9], where each row is a flattened 3x3 rotation matrix.
    """
    # Normalize the quaternions to avoid numerical instability
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute the rotation matrices
    rotation_matrices = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).T.reshape(-1, 9)

    return rotation_matrices

def extract_camera_position(view_matrix: torch.Tensor) -> torch.Tensor:
    """
    从视图矩阵中提取摄像机的世界坐标。
    
    参数:
        view_matrix (torch.Tensor): 形状为 (4, 4) 的视图矩阵。
    
    返回:
        torch.Tensor: 摄像机的世界坐标 (3,)。
    """
    if view_matrix.shape != (4, 4):
        raise ValueError("视图矩阵的形状必须为 (4, 4)")
    
    # 分离旋转矩阵和位移向量
    rotation_matrix = view_matrix[:3, :3]
    translation_vector = view_matrix[:3, 3]
    
    # 计算摄像机在世界坐标系中的位置
    camera_position = -torch.matmul(rotation_matrix.T, translation_vector)
    
    return camera_position

def compute_rays(view_matrix, H, W, fovx, fovy):
    """
    Compute ray origins and directions for a given camera view matrix, image resolution, and field of view.
    
    Parameters:
        view_matrix (numpy.ndarray): 4x4 view matrix.
        H (int): Height of the image.
        W (int): Width of the image.
        fovx (float): Field of view in the x-direction (in radians).
        fovy (float): Field of view in the y-direction (in radians).
    
    Returns:
        ray_o (numpy.ndarray): Origin of rays, shape (H, W, 3).
        ray_d (numpy.ndarray): Direction of rays, shape (H, W, 3).
    """
    # Extract rotation matrix R and translation vector T from the view matrix
    R = view_matrix[:3, :3]
    T = view_matrix[:3, 3]

    # Calculate the camera position in world coordinates
    camera_position = -np.dot(R.T, T)

    # Create an array to hold the ray origins and directions
    ray_o = np.zeros((H, W, 3), dtype=np.float32)
    ray_d = np.zeros((H, W, 3), dtype=np.float32)

    # Calculate the aspect ratio
    aspect_ratio = W / H

    # Compute focal length from field of view (fovx and fovy)
    fx = 0.5 * W / np.tan(fovx / 2)
    fy = 0.5 * H / np.tan(fovy / 2)

    # Generate normalized image coordinates
    x, y = np.meshgrid(
        (np.arange(W) - W / 2) / fx,  # Normalized x-coordinates
        (np.arange(H) - H / 2) / fy    # Normalized y-coordinates
    )
    
    # Create ray directions in camera space
    ray_d[:, :, 0] = x
    ray_d[:, :, 1] = y
    ray_d[:, :, 2] = -1  # Assume camera looks along the negative z-axis

    # Transform ray directions to world space
    ray_d = np.dot(R.T, ray_d.reshape(-1, 3).T).T.reshape(H, W, 3)

    # Normalize ray directions
    ray_d /= np.linalg.norm(ray_d, axis=2, keepdims=True)

    # Set ray origins (same for all rays, at camera position)
    ray_o[:, :, :] = camera_position

    return ray_o, ray_d

def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]
    
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def inv_trans(world_to_camera):
    """
    Converts a world-to-camera transformation matrix to a camera-to-world transformation matrix.

    Args:
        world_to_camera (np.ndarray): A 4x4 world-to-camera transformation matrix.

    Returns:
        np.ndarray: A 4x4 camera-to-world transformation matrix.
    """
    world_to_camera=world_to_camera.squeeze(0)
    if world_to_camera.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 matrix.")

    # Extract the rotation (R) and translation (t) parts
    R = world_to_camera[:3, :3]
    t = world_to_camera[:3, 3]

    # Compute the inverse rotation and translation
    R_inv = R.T
    t_inv = -R_inv @ t

    # Construct the camera-to-world matrix
    camera_to_world = torch.eye(4, device=world_to_camera.device, dtype=world_to_camera.dtype)
    camera_to_world[:3, :3] = R_inv
    camera_to_world[:3, 3] = t_inv
    camera_to_world=camera_to_world.unsqueeze(0)
    return camera_to_world

def render_process(gaussians,raster_settings):
    #focal = raster_settings.image_height / (2 * np.tan(np.radians(raster_settings.tanfovy) / 2))
    focal = raster_settings.image_height / (2 * raster_settings.tanfovy)
    intrinsics=np.array([focal, focal, raster_settings.image_width // 2, raster_settings.image_height // 2])
    
    #intrinsics=np.array([583.30588238,583.30588238,490.,272.])
    pose = raster_settings.viewmatrix.unsqueeze(0)
    pose=pose.transpose(1,2)
    pose=inv_trans(pose)
    '''
    pose=torch.tensor([[[ 1.,  0.,  0.,  0.],
         [ 0., -1.,  0.,  0.],
         [ 0.,  0., -1.,  5.],
         [ 0.,  0.,  0.,  1.]]], device='cuda:0')
    '''
    #print(pose)
    rays = get_rays(pose, intrinsics, raster_settings.image_height, raster_settings.image_width, -1)
    rays_o = rays['rays_o'].contiguous().view(-1, 3)
    rays_d = rays['rays_d'].contiguous().view(-1, 3)
    return gaussians.trace(rays_o,rays_d)
    #return Diff_Render.apply(means3D,scales,orien,cov,opacities,albedo,env_rgbs,space,viewmat,raster_settings)
    
    