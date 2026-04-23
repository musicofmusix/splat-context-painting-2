# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

DEFAULT_DEVICE = torch.device("cuda")


def cumulative_distances(points_torch: torch.Tensor) -> torch.Tensor:
    diffs = points_torch[1:] - points_torch[:-1]
    segment_lengths = torch.sqrt((diffs**2).sum(dim=1))
    cumsums = torch.cumsum(segment_lengths, dim=0)
    zero = torch.zeros(1, dtype=cumsums.dtype, device=cumsums.device)
    distances = torch.cat([zero, cumsums], dim=0)
    return distances


def compute_3d_smooth_curve_and_length_arc_length_torch(points, normals):
    distances = cumulative_distances(points)
    spline_x = NaturalCubicSpline(
        natural_cubic_spline_coeffs(distances, points[:, 0].unsqueeze(0).T)
    )
    spline_y = NaturalCubicSpline(
        natural_cubic_spline_coeffs(distances, points[:, 1].unsqueeze(0).T)
    )
    spline_z = NaturalCubicSpline(
        natural_cubic_spline_coeffs(distances, points[:, 2].unsqueeze(0).T)
    )
    normal_spline_x = NaturalCubicSpline(
        natural_cubic_spline_coeffs(distances, normals[:, 0].unsqueeze(0).T)
    )
    normal_spline_y = NaturalCubicSpline(
        natural_cubic_spline_coeffs(distances, normals[:, 1].unsqueeze(0).T)
    )
    normal_spline_z = NaturalCubicSpline(
        natural_cubic_spline_coeffs(distances, normals[:, 2].unsqueeze(0).T)
    )

    def ds(t):
        dx_dt = spline_x.derivative(t, 1)
        dy_dt = spline_y.derivative(t, 1)
        dz_dt = spline_z.derivative(t, 1)
        return torch.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)

    torch_curve = (spline_x, spline_y, spline_z)
    normal_torch_curve = (normal_spline_x, normal_spline_y, normal_spline_z)
    total_arc_length = distances[-1]
    return (
        torch_curve,
        normal_torch_curve,
        total_arc_length,
    )


def compute_distance_for_bounding(
    points, cam_center, look_dir, up_dir, fov_vertical_deg, aspect_ratio, margin=1.1
):
    z_cam = -look_dir / np.linalg.norm(look_dir)
    y_cam = up_dir / np.linalg.norm(up_dir)
    x_cam = np.cross(y_cam, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    R = np.vstack([x_cam, y_cam, z_cam]).T
    local_points = (points - cam_center) @ R
    min_vals = np.min(local_points, axis=0)
    max_vals = np.max(local_points, axis=0)
    width_local = max_vals[0] - min_vals[0]
    height_local = max_vals[1] - min_vals[1]
    fov_vert_rad = np.radians(fov_vertical_deg)
    half_fov_vert = fov_vert_rad / 2.0
    half_fov_horiz = np.arctan(aspect_ratio * np.tan(half_fov_vert))
    distance_for_width = (width_local / 2.0) / np.tan(half_fov_horiz)
    distance_for_height = (height_local / 2.0) / np.tan(half_fov_vert)
    distance = max(distance_for_width, distance_for_height)
    return distance * margin


def place_stamps_on_curve_cameras_torch(
    spline_points,
    spline_normals,
    total_length,
    stamp_length,
    stamp_points=None,
    overlap=0.0,
    cam_params=None,
):
    spline_x = spline_points[0]
    spline_y = spline_points[1]
    spline_z = spline_points[2]
    effective_length = stamp_length * (1 - overlap)
    num_stamps = int(total_length / effective_length)
    midpoints = np.array([])
    normals = np.array([])
    camera_positions = []
    distance_factor = 1.5
    for i in range(num_stamps):
        midpoint_param = torch.tensor(
            (i * effective_length + (stamp_length / 2)), device=DEFAULT_DEVICE
        )
        if midpoint_param > total_length:
            break
        x = spline_x.evaluate(midpoint_param).detach().cpu().numpy()[0]
        y = spline_y.evaluate(midpoint_param).detach().cpu().numpy()[0]
        z = spline_z.evaluate(midpoint_param).detach().cpu().numpy()[0]
        normal_x = spline_normals[0].evaluate(midpoint_param).detach().cpu().numpy()[0]
        normal_y = spline_normals[1].evaluate(midpoint_param).detach().cpu().numpy()[0]
        normal_z = spline_normals[2].evaluate(midpoint_param).detach().cpu().numpy()[0]
        normal = np.array([normal_x, normal_y, normal_z])
        normal = normal / np.linalg.norm(normal)
        if i == 0:
            midpoints = np.array([[x, y, z]])
            normals = np.array([normal])
        else:
            midpoints = np.append(midpoints, [[x, y, z]], axis=0)
            normals = np.append(normals, [normal], axis=0)
        if i > 0:
            overlap_center_param = midpoint_param + (stamp_length / 2) * overlap
            overlap_center_x = (
                spline_x.evaluate(overlap_center_param).detach().cpu().numpy()[0]
            )
            overlap_center_y = (
                spline_y.evaluate(overlap_center_param).detach().cpu().numpy()[0]
            )
            overlap_center_z = (
                spline_z.evaluate(overlap_center_param).detach().cpu().numpy()[0]
            )
            overlap_center_normal_x = (
                spline_normals[0]
                .evaluate(overlap_center_param)
                .detach()
                .cpu()
                .numpy()[0]
            )
            overlap_center_normal_y = (
                spline_normals[1]
                .evaluate(overlap_center_param)
                .detach()
                .cpu()
                .numpy()[0]
            )
            overlap_center_normal_z = (
                spline_normals[2]
                .evaluate(overlap_center_param)
                .detach()
                .cpu()
                .numpy()[0]
            )
            combined_midpoints = np.vstack((midpoints[-2:],))
            min_bounds = np.min(combined_midpoints, axis=0)
            max_bounds = np.max(combined_midpoints, axis=0)
            center = np.array([overlap_center_x, overlap_center_y, overlap_center_z])
            size = np.linalg.norm(max_bounds - min_bounds)
            bbox_center = (min_bounds + max_bounds) / 2.0
            fov_degrees = cam_params.get_fov_vertical_deg() if cam_params else 45.0
            aspect_ratio = cam_params.get_aspect() if cam_params else 1.78
            fov_rad = np.radians(fov_degrees)
            half_fov = fov_rad / 2.0
            camera_distance = distance_factor * (size) / np.tan(half_fov)
            cam_tangent = np.array(
                [
                    spline_x.derivative(overlap_center_param, 1)
                    .detach()
                    .cpu()
                    .numpy()[0],
                    spline_y.derivative(overlap_center_param, 1)
                    .detach()
                    .cpu()
                    .numpy()[0],
                    spline_z.derivative(overlap_center_param, 1)
                    .detach()
                    .cpu()
                    .numpy()[0],
                ]
            )
            cam_normal = np.array(
                [
                    overlap_center_normal_x,
                    overlap_center_normal_y,
                    overlap_center_normal_z,
                ]
            )
            binormal = np.cross(cam_tangent, cam_normal)
            binormal = binormal / np.linalg.norm(binormal)
            if stamp_points is not None:
                curr_stamp_points = stamp_points[i]
                prev_stamp_points = stamp_points[i - 1]
                combined_points = np.vstack(
                    [
                        prev_stamp_points.detach().cpu().numpy(),
                        curr_stamp_points.detach().cpu().numpy(),
                    ]
                )
                distance_side = compute_distance_for_bounding(
                    points=combined_points,
                    cam_center=bbox_center,
                    look_dir=binormal,
                    up_dir=cam_normal,
                    fov_vertical_deg=fov_degrees,
                    aspect_ratio=aspect_ratio,
                    margin=2.5,
                )
                distance_top = compute_distance_for_bounding(
                    points=combined_points,
                    cam_center=bbox_center,
                    look_dir=cam_normal,
                    up_dir=-cam_tangent,
                    fov_vertical_deg=fov_degrees,
                    aspect_ratio=aspect_ratio,
                    margin=1.75,
                )
                side_view_position = bbox_center + distance_side * binormal
                top_view_position = bbox_center + distance_top * cam_normal
            else:
                side_view_position = bbox_center + camera_distance * binormal
                top_view_position = bbox_center + camera_distance * cam_normal
            camera_positions.append(
                (
                    side_view_position,
                    top_view_position,
                    center,
                    cam_normal,
                    binormal,
                    cam_tangent,
                )
            )
    return midpoints, normals, camera_positions


def compute_simple_deformation_gpu(
    src_points_torch,
    idx,
    stamp_length,
    effective_length,
    stamp_center_torch,
    stamp_frame_torch,
    spline=None,
    torch_curve=None,
    timer=None,
):
    results = {}
    if timer:
        timer.start("compute_simple_deformation_gpu - transfer inputs")
    src_points_torch = src_points_torch.double()
    stamp_center_torch = stamp_center_torch.double()
    stamp_frame_torch = stamp_frame_torch.double()
    if timer:
        torch.cuda.synchronize()
        timer.stop("compute_simple_deformation_gpu - transfer inputs")
        timer.start("compute_simple_deformation_gpu - find local coords and t values")
    norm_f = stamp_frame_torch.norm(dim=1, keepdim=True)
    stamp_frame_torch = stamp_frame_torch / norm_f
    stamp_inv_torch = stamp_frame_torch.T
    temp = torch.mm(stamp_inv_torch, (src_points_torch - stamp_center_torch).T)
    local_coords_set = temp.T
    results["stamp_frame"] = stamp_frame_torch
    results["stamp_inv"] = stamp_inv_torch
    results["local_coords_set"] = local_coords_set
    t_values = idx * effective_length + 0.5 * stamp_length + local_coords_set[:, 0]
    if timer:
        torch.cuda.synchronize()
        timer.stop("compute_simple_deformation_gpu - find local coords and t values")
        timer.start("compute_simple_deformation_gpu - spline evaluations")
    if torch_curve is not None:
        torch_curve_x, torch_curve_y, torch_curve_z = torch_curve
        t_values_torch = t_values
        pos_torch = torch.stack(
            [
                torch_curve_x.evaluate(t_values_torch),
                torch_curve_y.evaluate(t_values_torch),
                torch_curve_z.evaluate(t_values_torch),
            ],
            dim=1,
        )[:, :, 0]
        tangents_torch = torch.stack(
            [
                torch_curve_x.derivative(t_values_torch, 1),
                torch_curve_y.derivative(t_values_torch, 1),
                torch_curve_z.derivative(t_values_torch, 1),
            ],
            dim=1,
        )[:, :, 0]
    else:
        spline_x, spline_y, spline_z = spline
        t_values_np = t_values.detach().cpu().numpy()
        pos_np = np.column_stack(
            [spline_x(t_values_np), spline_y(t_values_np), spline_z(t_values_np)]
        )
        tangents_np = np.column_stack(
            [
                spline_x(t_values_np, 1),
                spline_y(t_values_np, 1),
                spline_z(t_values_np, 1),
            ]
        )
        pos_torch = torch.tensor(pos_np, device=DEFAULT_DEVICE, dtype=torch.float32)
        tangents_torch = torch.tensor(
            tangents_np, device=DEFAULT_DEVICE, dtype=torch.float32
        )
    if timer:
        torch.cuda.synchronize()
        timer.stop("compute_simple_deformation_gpu - spline evaluations")
        timer.start("compute_simple_deformation_gpu - normalize, project and compute")
    results["pos"] = pos_torch
    results["tangents"] = tangents_torch
    tangents_norm = tangents_torch.norm(dim=1, keepdim=True)
    tangents_torch = tangents_torch / tangents_norm
    binormal = stamp_frame_torch[:, 2]
    binormal_batched = binormal.unsqueeze(0).expand(tangents_torch.shape[0], -1)
    binormal_norm = binormal_batched.norm(dim=1, keepdim=True)
    binormal_batched = binormal_batched / binormal_norm
    dot_products = (tangents_torch * binormal_batched).sum(dim=1, keepdim=True)
    projected_tangents = tangents_torch - dot_products * binormal_batched
    projected_norm = projected_tangents.norm(dim=1, keepdim=True)
    projected_tangents = projected_tangents / projected_norm
    results["projected_tangents"] = projected_tangents
    new_normals = torch.cross(binormal_batched, projected_tangents, dim=1)
    results["new_normals pre flip"] = new_normals.clone()
    dot_normal = (new_normals * stamp_frame_torch[:, 1]).sum(dim=1)
    results["correction dots"] = dot_normal
    correction_mask = dot_normal < 0
    results["correction mask"] = correction_mask.int()
    new_normals[correction_mask] = -1.0 * new_normals[correction_mask]
    new_normals = new_normals / new_normals.norm(dim=1, keepdim=True)
    results["new_normals post flip"] = new_normals
    new_binormals = binormal_batched
    results["new_binormals"] = new_binormals
    new_frames_torch = torch.stack(
        [projected_tangents, new_normals, new_binormals], dim=1
    )
    results["new_frames"] = new_frames_torch
    normal_component = local_coords_set[:, 1].unsqueeze(-1) * new_frames_torch[:, 1, :]
    binormal_component = (
        local_coords_set[:, 2].unsqueeze(-1) * new_frames_torch[:, 2, :]
    )
    deformed_positions_torch = pos_torch + normal_component + binormal_component
    results["deformed_positions"] = deformed_positions_torch
    results["normal"] = new_frames_torch[:, 1, :]
    results["binormal"] = new_frames_torch[:, 2, :]
    if timer:
        torch.cuda.synchronize()
        timer.stop("compute_simple_deformation_gpu - normalize, project and compute")
    new_frames = new_frames_torch.detach().cpu().numpy()
    return deformed_positions_torch.float(), new_frames, results
