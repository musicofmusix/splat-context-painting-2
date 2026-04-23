# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim

DEFAULT_DEVICE = torch.device("cuda")
MAX_DEPTH = 1000.0


def find_connected_components(points, threshold=0.1):
    connected_components = []
    while points.shape[0] > 0:
        point = points[0]
        mask = torch.norm(points - point, dim=1) < threshold
        connected_components.append(points[mask])
        points = points[~mask]
    return connected_components


def find_lower_centre(pts, dir):
    pts_mean = torch.mean(pts, dim=0)
    pts = pts - pts_mean
    min_proj = torch.min(torch.matmul(pts, dir))
    return torch.mean(pts[torch.matmul(pts, dir) == min_proj], dim=0) + pts_mean


def ray_sphere_intersection(sphere_center, sphere_radius, ray_origin, ray_direction):
    ray_direction = ray_direction / torch.linalg.norm(ray_direction)
    oc = ray_origin - sphere_center
    a = torch.dot(ray_direction, ray_direction)
    b = 2.0 * torch.dot(ray_direction, oc)
    c = torch.dot(oc, oc) - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_discriminant = torch.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    valid_ts = [t for t in [t1, t2] if t >= 0]
    if not valid_ts:
        return None
    t = min(valid_ts)
    intersection = ray_origin + t * ray_direction
    normal = (intersection - sphere_center) / sphere_radius
    return t, intersection, normal


def ray_plane_intersection(plane_normal, plane_point, ray_origin, ray_direction):
    ray_direction = ray_direction / torch.linalg.norm(ray_direction)
    ndotd = torch.dot(plane_normal.to(torch.float32), ray_direction.to(torch.float32))
    if abs(ndotd) < 1e-6:
        return None
    t = torch.dot(plane_normal, (plane_point - ray_origin)) / ndotd
    intersection = ray_origin + t * ray_direction
    return t, intersection


def get_plane_normal_and_point(model, mask):
    selected = model.get_xyz[mask]
    mean = selected.mean(dim=0)
    V = torch.pca_lowrank(selected)[2]
    if V.shape[0] < 3 or selected.shape[0] < 3:
        return torch.eye(4)
    normal = V[:, 2]
    world_ray = ps.screen_coords_to_world_ray(psim.GetMousePos())
    if np.dot(normal.detach().cpu().numpy(), world_ray) > 0:
        normal = -normal
    return normal, mean


def auto_orient_transform(model, mask, up=np.array([0, -1.0, 0])):
    selected = model.get_xyz[mask]
    mean = selected.mean(dim=0)
    V = torch.pca_lowrank(selected)[2]
    if V.shape[0] < 3 or selected.shape[0] < 3:
        return torch.eye(4), None, None
    vec1 = V[:, 2].detach().cpu().numpy()
    vec2 = up
    world_ray = ps.screen_coords_to_world_ray(psim.GetMousePos())
    if np.dot(vec1, world_ray) > 0:
        vec1 = -vec1
    if np.isclose(np.dot(vec1, vec2), 1.0, atol=1e-2):
        return torch.from_numpy(np.eye(4)), vec1, None
    if np.isclose(np.dot(vec1, vec2), -1.0, atol=1e-2):
        orthogonal_vector = (
            np.array([1, 0, 0]) if not np.isclose(vec1[0], 1.0) else np.array([0, 1, 0])
        )
        axis = np.cross(vec1, orthogonal_vector)
        axis = axis / np.linalg.norm(axis)
        res = np.eye(4)
        res[:3, :3] = R.from_rotvec(np.pi * axis).as_matrix()
        res[:3, 3:] = -mean.detach().cpu().numpy().reshape(3, 1)
        return torch.from_numpy(res), vec1, None
    rot, ssd = R.align_vectors(vec1, up)
    res = np.eye(4)
    res[:3, :3] = rot.as_matrix()
    res[:3, 3:] = -mean.detach().cpu().numpy().reshape(3, 1)
    return torch.from_numpy(res), vec1, rot.as_quat()


def quaternion_product(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=1)


def calculate_bbox_intersection_percentage(bbox1, bbox2):
    max_min = torch.max(bbox1[0], bbox2[0])
    min_max = torch.min(bbox1[1], bbox2[1])
    intersection_dims = torch.clamp(min_max - max_min, min=0)
    intersection_volume = intersection_dims.prod().item()
    bbox1_volume = (bbox1[1] - bbox1[0]).prod().item()
    bbox2_volume = (bbox2[1] - bbox2[0]).prod().item()
    union_volume = bbox1_volume + bbox2_volume - intersection_volume
    if union_volume > 0:
        percentage_intersection = (intersection_volume / union_volume) * 100
    else:
        percentage_intersection = 0
    return percentage_intersection


def rotation_matrix_to_quaternion(R_mat):
    if isinstance(R_mat, torch.Tensor):
        R_mat = R_mat.clone().detach().to(device=DEFAULT_DEVICE)
    else:
        R_mat = torch.tensor(R_mat, device=DEFAULT_DEVICE)
    R11 = R_mat[..., 0, 0]
    R22 = R_mat[..., 1, 1]
    R33 = R_mat[..., 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + R11 + R22 + R33, min=1e-12)) / 2.0
    R32 = R_mat[..., 2, 1]
    R23 = R_mat[..., 1, 2]
    R13 = R_mat[..., 0, 2]
    R31 = R_mat[..., 2, 0]
    R21 = R_mat[..., 1, 0]
    R12 = R_mat[..., 0, 1]
    x = (R32 - R23) / (4.0 * w)
    y = (R13 - R31) / (4.0 * w)
    z = (R21 - R12) / (4.0 * w)
    quats = torch.stack([w, x, y, z], dim=-1)
    return quats


def compute_relative_rotations(stamp_frame, new_frames):
    stamp_frame = torch.tensor(stamp_frame, device=DEFAULT_DEVICE, dtype=torch.float32)
    new_frames = torch.tensor(
        new_frames, dtype=stamp_frame.dtype, device=DEFAULT_DEVICE
    )
    new_frames_T = new_frames.transpose(-2, -1)
    R_change = torch.matmul(new_frames_T, stamp_frame.T)
    local_quats = rotation_matrix_to_quaternion(R_change)
    return local_quats
