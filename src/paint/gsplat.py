# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from torch import nn
from deps.gsplats.scene.gaussian_model import GaussianModel
from src.paint.geometry import quaternion_product
import glm
import random

DEFAULT_DEVICE = torch.device("cuda")
MAX_DEPTH = 1000.0

def _infer_sh_degree(gaussians_list):
    # Imported brushes can carry fewer SH coefficients than the live scene, so the
    # composite model has to derive its SH degree from the actual tensor widths.
    max_coeffs = max(
        getattr(g, "_features_dc").shape[1] + getattr(g, "_features_rest").shape[1]
        for g in gaussians_list
    )
    sh_degree = int(round(np.sqrt(max_coeffs) - 1))
    return max(sh_degree, 0)


def _pad_sh_features(feature_tensor, target_coeffs):
    current_coeffs = feature_tensor.shape[1]
    if current_coeffs == target_coeffs:
        return feature_tensor.data
    if current_coeffs > target_coeffs:
        return feature_tensor.data[:, :target_coeffs, :]
    # Pad lower-order SH tensors so mixed scene/brush composites still match the
    # feature layout expected by downstream rendering and embedding preview code.
    padding = feature_tensor.data.new_zeros(
        feature_tensor.shape[0], target_coeffs - current_coeffs, feature_tensor.shape[2]
    )
    return torch.cat([feature_tensor.data, padding], dim=1)


def _concat_attribute(gaussians_list, attribute_name, sh_degree):
    if attribute_name == "_features_rest":
        target_coeffs = (sh_degree + 1) ** 2 - 1
        concatenated = torch.cat(
            [
                _pad_sh_features(getattr(g, attribute_name), target_coeffs)
                for g in gaussians_list
            ],
            dim=0
        )
    else:
        concatenated = torch.cat(
            [getattr(g, attribute_name).data for g in gaussians_list], dim=0
        )
    source_attr = getattr(gaussians_list[0], attribute_name)
    requires_grad = bool(getattr(source_attr, "requires_grad", False))
    # Some attributes, like _has_embedding, are boolean tensors and must remain
    # non-trainable when wrapped back into nn.Parameter.
    if not torch.is_floating_point(concatenated) and not torch.is_complex(concatenated):
        requires_grad = False
    return nn.Parameter(concatenated, requires_grad=requires_grad)

def create_from_gaussians(gaussians_list):
    assert len(gaussians_list) > 0
    sh_degree = _infer_sh_degree(gaussians_list)
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.active_sh_degree = min(
        sh_degree,
        max(getattr(g, "active_sh_degree", 0) for g in gaussians_list),
    )
    attribute_names = tensor_attributes(gaussians_list[0])

    for attribute_name in attribute_names:
        setattr(
            gaussians,
            attribute_name,
            _concat_attribute(gaussians_list, attribute_name, sh_degree),
        )
    return gaussians


def composite_gaussians(gaussians_list):
    gaussians_composite = create_from_gaussians(gaussians_list)
    if hasattr(gaussians_composite, "_incidents_dc") and hasattr(
        gaussians_composite, "_incidents_rest"
    ):
        gaussians_composite._incidents_dc.data[:] = 0
        gaussians_composite._incidents_rest.data[:] = 0
    return gaussians_composite


def transform_gs(model, se_transform=None):
    with torch.no_grad():
        if se_transform is None:
            se_transform = torch.tensor(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                device=model._xyz.device,
                dtype=model._xyz.dtype,
            )
        rotation_matrix = se_transform[:3, :3]
        translation_vector = se_transform[:3, 3]
        model._xyz = (rotation_matrix @ model._xyz.T + translation_vector[:, None]).T
        np_quaternion = glm.quat_cast(glm.mat3(rotation_matrix.cpu().numpy()))
        rotation_quat = -model._xyz.new_tensor(np_quaternion, requires_grad=False)[None]
        model._rotation = quaternion_product(rotation_quat, model._rotation)
    return


def matrix_transform_gs(gs, transformation_matrix):
    with torch.no_grad():
        xyz = gs._xyz
        rotation = gs._rotation
        scaling = gs.scaling_activation(gs._scaling)
        gizmo_transform = transformation_matrix.detach().cpu().numpy()
        rs_transform = gizmo_transform[:3, :3]
        scale_transform = np.linalg.norm(rs_transform, axis=0, keepdims=True)
        rotation_transform = rs_transform / scale_transform
        np_quaternion = glm.quat_cast(glm.mat3(rotation_transform.astype(np.float32)))
        rotation_quat = -xyz.new_tensor(np_quaternion)[None]
        gizmo_transform = torch.from_numpy(gizmo_transform).to(xyz.device, xyz.dtype)[
            None
        ]
        xyz_hom = torch.cat([xyz, xyz.new_ones(xyz.shape[0], 1)], dim=1)
        transformed_xyz = (
            gizmo_transform.expand([xyz_hom.shape[0], 4, 4]) @ xyz_hom[:, :, None]
        )
        gs._xyz = transformed_xyz[:, :3, 0]
        gs._rotation = quaternion_product(rotation_quat, rotation)
        gs._scaling = gs.scaling_inverse_activation(
            scaling * scaling.new_tensor(scale_transform)
        )
    return gs


def jitter_scale_gs(gs, min_scale=0.9, max_scale=1.1):
    xyz = gs._xyz
    device = xyz.device
    dtype = xyz.dtype
    factor = random.uniform(min_scale, max_scale)
    center = xyz.mean(dim=0)
    T1 = torch.eye(4, dtype=dtype, device=device)
    T1[0, 3] = -center[0]
    T1[1, 3] = -center[1]
    T1[2, 3] = -center[2]
    T2 = torch.eye(4, dtype=dtype, device=device)
    T2[0, 3] = center[0]
    T2[1, 3] = center[1]
    T2[2, 3] = center[2]
    S = torch.eye(4, dtype=dtype, device=device)
    S[0, 0] = factor
    S[1, 1] = factor
    S[2, 2] = factor
    transform_mat = T2 @ S @ T1
    transformed_gs = composite_gaussians([gs])
    matrix_transform_gs(transformed_gs, transform_mat)
    return transformed_gs


def jitter_rotation_gs(gs, axis=(0.0, 0.0, 1.0), max_angle=0.1):
    xyz = gs._xyz
    device = xyz.device
    dtype = xyz.dtype
    angle = random.uniform(-max_angle, max_angle)
    if isinstance(axis, torch.Tensor):
        axis_t = axis.clone().detach().to(dtype=dtype, device=device)
    else:
        axis_t = torch.tensor(axis, dtype=dtype, device=device)
    axis_t = axis_t / torch.norm(axis_t)
    rot_vec = (angle * axis_t).cpu().numpy()
    rotation_mat_np = R.from_rotvec(rot_vec).as_matrix()
    R_mat = torch.eye(4, dtype=dtype, device=device)
    R_mat[:3, :3] = torch.from_numpy(rotation_mat_np).to(device=device, dtype=dtype)
    center = xyz.mean(dim=0)
    T1 = torch.eye(4, dtype=dtype, device=device)
    T1[0, 3] = -center[0]
    T1[1, 3] = -center[1]
    T1[2, 3] = -center[2]
    T2 = torch.eye(4, dtype=dtype, device=device)
    T2[0, 3] = center[0]
    T2[1, 3] = center[1]
    T2[2, 3] = center[2]
    transform_mat = T2 @ R_mat @ T1
    transformed_gs = composite_gaussians([gs])
    matrix_transform_gs(transformed_gs, transform_mat)
    return transformed_gs


def apply_mask_to_attributes(points, mask):
    updated_attributes = {}
    for attr_name in tensor_attributes(points):
        attr_value = getattr(points, attr_name)
        if attr_value is not None and isinstance(attr_value, nn.Parameter):
            if attr_value.data.dim() > 0 and mask.size(0) == attr_value.data.size(0):
                updated_attributes[attr_name] = nn.Parameter(
                    attr_value.data[mask], requires_grad=attr_value.requires_grad
                )
    for attr_name, new_attr_value in updated_attributes.items():
        setattr(points, attr_name, new_attr_value)


def tensor_attributes(points):
    attributes = [
        "_xyz",
        "_opacity",
        "_scaling",
        "_rotation",
        "_features_dc",
        "_features_rest",
        "_embedding",
        "_has_embedding",
    ]
    if hasattr(points, "_base_color"):
        attributes.extend(
            [
                "_base_color",
                "_roughness",
                "_metallic",
                "_incidents_dc",
                "_incidents_rest",
                "_visibility_dc",
                "_visibility_rest",
            ]
        )
    return attributes
