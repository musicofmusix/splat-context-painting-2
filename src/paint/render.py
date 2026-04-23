# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from deps.gsplats.scene.gaussian_model import GaussianModel
from deps.gsplats.utils.camera_utils import Camera as GSCamera
import polyscope as ps

DEFAULT_DEVICE = torch.device("cuda")

def find_closest_component(connected_components):
    min_dist = float("inf")
    closest_component = None
    for i in range(len(connected_components)):
        component = connected_components[i]
        depth = get_depth_from_current_view(component)
        dist = depth.mean()
        if dist < min_dist and dist > 0:
            min_dist = dist
            closest_component = component
    return closest_component


def get_depth_from_current_view(points):
    cam_params = ps.get_view_camera_parameters()
    cam_pos = torch.tensor(cam_params.get_position(), device=DEFAULT_DEVICE)
    cam_dist = points - cam_pos
    depth = torch.norm(cam_dist, dim=1)
    return depth


def polyscope_to_gsplat_camera(
    camera: ps.CameraParameters = None, downsample_factor: int = 1
):
    window_w, window_h = ps.get_window_size()
    image_w = max(window_w // downsample_factor, 1)
    image_h = max(window_h // downsample_factor, 1)
    ps_camera = ps.get_view_camera_parameters() if camera is None else camera
    fov_y = max(ps_camera.get_fov_vertical_deg() * np.pi / 180.0, 1e-4)
    aspect = max(ps_camera.get_aspect(), 1e-6)
    fov_x = 2.0 * np.arctan(np.tan(0.5 * fov_y) * aspect)
    fov_x = max(float(fov_x), 1e-4)
    trans_mat = np.eye(4)
    trans_mat[1, 1] = -1.0
    trans_mat[2, 2] = -1.0
    gs_cam = GSCamera(
        colmap_id=0,
        R=np.float32(trans_mat[:3, :3] @ ps_camera.get_R()),
        T=np.float32(trans_mat[:3, :3] @ ps_camera.get_T()),
        E=np.float32(trans_mat @ ps_camera.get_E()),
        gt_alpha_mask=None,
        FoVx=fov_x,
        FoVy=fov_y,
        image=torch.zeros((3, image_h, image_w)),
        image_name="fake",
        uid=0,
    )
    return gs_cam


def gsplat_to_polyscope_camera(gs_camera: GSCamera):
    view_mat = gs_camera.world_view_transform.cpu().numpy().transpose()
    trans_mat = np.eye(4)
    trans_mat[1, 1] = -1.0
    trans_mat[2, 2] = -1.0
    view_mat = trans_mat @ view_mat
    aspect = ps.get_view_camera_parameters().get_aspect()
    fov_y = gs_camera.FoVy
    fov_x = 2.0 * np.arctan(np.tan(0.5 * fov_y) * aspect)
    ps_cam_param = ps.CameraParameters(
        ps.CameraIntrinsics(
            fov_vertical_deg=np.degrees(fov_y), fov_horizontal_deg=np.degrees(fov_x)
        ),
        ps.CameraExtrinsics(mat=view_mat),
    )
    return ps_cam_param

def project_gaussian_means_to_2d_pre_ndc_depth(
    model: GaussianModel, camera: ps.CameraParameters
):
    gaussians_pos = model.get_xyz.detach()
    N = gaussians_pos.shape[0]
    ones_padding = gaussians_pos.new_ones(N, 1)
    xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
    gsplat_cam = polyscope_to_gsplat_camera(camera)
    cam_view_projection_matrix = gsplat_cam.full_proj_transform.T[None].expand(N, 4, 4)
    transformed_xyz = cam_view_projection_matrix @ xyz_homogeneous
    transformed_xyz = transformed_xyz.squeeze(-1)
    return transformed_xyz[:, 2]


def project_gaussian_means_to_2d(model: GaussianModel, camera: ps.CameraParameters):
    gaussians_pos = model.get_xyz.detach()
    N = gaussians_pos.shape[0]
    ones_padding = gaussians_pos.new_ones(N, 1)
    xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
    gsplat_cam = polyscope_to_gsplat_camera(camera)
    cam_view_projection_matrix = gsplat_cam.full_proj_transform.T[None].expand(N, 4, 4)
    transformed_xyz = cam_view_projection_matrix @ xyz_homogeneous
    transformed_xyz = transformed_xyz.squeeze(-1)
    transformed_xyz /= transformed_xyz[:, -1:]
    return transformed_xyz


def project_gaussian_means_to_2d_pos(gaussians_pos, camera: ps.CameraParameters):
    N = gaussians_pos.shape[0]
    ones_padding = gaussians_pos.new_ones(N, 1)
    xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
    gsplat_cam = polyscope_to_gsplat_camera(camera)
    cam_view_projection_matrix = gsplat_cam.full_proj_transform.T[None].expand(N, 4, 4)
    transformed_xyz = cam_view_projection_matrix @ xyz_homogeneous
    transformed_xyz = transformed_xyz.squeeze(-1)
    transformed_xyz = transformed_xyz / transformed_xyz[:, -1:].clone()
    return transformed_xyz


def get_minimal_surface_mask_3d(
    mouse_pos, brush_radius_2d, scene_gs, z_buffer, sphere_radius_3d=0.05
):
    W, H = ps.get_window_size()
    brush_screen_bbox = [
        [mouse_pos[0] - brush_radius_2d, mouse_pos[1] - brush_radius_2d],
        [mouse_pos[0] + brush_radius_2d, mouse_pos[1] + brush_radius_2d],
    ]
    pc_screen_xyz = project_gaussian_means_to_2d_pos(scene_gs._xyz.detach(), None)
    pc_screen_xyz[:, 0].copy_(((pc_screen_xyz[:, 0] + 1) / 2.0 * W).round())
    pc_screen_xyz[:, 1].copy_(((pc_screen_xyz[:, 1] + 1) / 2.0 * H).round())
    screen_mask_bbox = (
        (pc_screen_xyz[:, 0] >= brush_screen_bbox[0][0])
        & (pc_screen_xyz[:, 0] <= brush_screen_bbox[1][0])
        & (pc_screen_xyz[:, 1] >= brush_screen_bbox[0][1])
        & (pc_screen_xyz[:, 1] <= brush_screen_bbox[1][1])
    )
    normalized_depths = project_gaussian_means_to_2d_pre_ndc_depth(scene_gs, None)
    z_h = z_buffer.shape[0]
    z_w = z_buffer.shape[1]
    x_coords = pc_screen_xyz[:, 0].clamp(0, z_w - 1).long()
    y_coords = pc_screen_xyz[:, 1].clamp(0, z_h - 1).long()
    visibility = torch.zeros(
        scene_gs._xyz.shape[0], dtype=torch.bool, device=z_buffer.device
    )
    valid = (x_coords >= 0) & (x_coords < z_w) & (y_coords >= 0) & (y_coords < z_h)
    tol = 1e-2
    vis_valid = torch.abs(
        normalized_depths[valid] - z_buffer[y_coords[valid], x_coords[valid]].squeeze()
    ) < tol
    visibility[valid] = vis_valid
    infront = normalized_depths > 0.0
    in_window = (
        (pc_screen_xyz[:, 0] >= 0)
        & (pc_screen_xyz[:, 0] < W)
        & (pc_screen_xyz[:, 1] >= 0)
        & (pc_screen_xyz[:, 1] < H)
    )
    visibility &= screen_mask_bbox & in_window & infront
    idx_vis = torch.nonzero(visibility, as_tuple=True)[0]
    if idx_vis.numel() == 0:
        return visibility
    screen_pts = pc_screen_xyz[idx_vis, :2]
    mouse_tensor = torch.tensor(mouse_pos, device=screen_pts.device)
    d2 = torch.norm(screen_pts - mouse_tensor.unsqueeze(0), dim=1)
    hit_idx = idx_vis[d2.argmin()]
    hit_center = scene_gs._xyz[hit_idx]
    d3 = torch.norm(scene_gs._xyz - hit_center.unsqueeze(0), dim=1)
    mask3d = d3 <= sphere_radius_3d
    visibility &= mask3d
    return visibility


def get_minimal_surface_mask(mouse_pos, radius, scene_gs, z_buffer):
    pc_screen_xyz = project_gaussian_means_to_2d_pos(scene_gs._xyz.detach(), None)
    W, H = ps.get_window_size()
    brush_screen_bbox = [
        [mouse_pos[0] - radius, mouse_pos[1] - radius],
        [mouse_pos[0] + radius, mouse_pos[1] + radius],
    ]
    pc_screen_xyz[:, 0].copy_(((pc_screen_xyz[:, 0] + 1) / 2.0 * W).round())
    pc_screen_xyz[:, 1].copy_((((pc_screen_xyz[:, 1] + 1) / 2.0) * H).round())
    screen_mask_bbox = (
        (pc_screen_xyz[:, 0] >= brush_screen_bbox[0][0])
        & (pc_screen_xyz[:, 0] <= brush_screen_bbox[1][0])
        & (pc_screen_xyz[:, 1] >= brush_screen_bbox[0][1])
        & (pc_screen_xyz[:, 1] <= brush_screen_bbox[1][1])
    )
    normalized_depths = project_gaussian_means_to_2d_pre_ndc_depth(scene_gs, None)
    z_buffer_height = z_buffer.shape[0]
    z_buffer_width = z_buffer.shape[1]
    x_coords = pc_screen_xyz[:, 0].clamp(0, z_buffer_width - 1).long()
    y_coords = pc_screen_xyz[:, 1].clamp(0, z_buffer_height - 1).long()
    tolerance = 1e-2
    visibility = torch.zeros(scene_gs._xyz.shape[0], dtype=torch.bool).to(
        device=z_buffer.device
    )
    valid_indices = (
        (x_coords >= 0)
        & (x_coords < z_buffer_width)
        & (y_coords >= 0)
        & (y_coords < z_buffer_height)
    )
    valid_depths = (
        normalized_depths
        - z_buffer[y_coords[valid_indices], x_coords[valid_indices]].squeeze()
    ) < tolerance
    visibility[valid_indices] = valid_depths
    infront = normalized_depths > -1.0
    in_window = (
        (pc_screen_xyz[:, 0] >= 0)
        & (pc_screen_xyz[:, 0] < W)
        & (pc_screen_xyz[:, 1] >= 0)
        & (pc_screen_xyz[:, 1] < H)
    )
    visibility &= screen_mask_bbox & in_window & infront
    return visibility
