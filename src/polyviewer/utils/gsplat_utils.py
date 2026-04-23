# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import json
import numpy as np
import skimage.color
import torch
import glm
from deps.gsplats.utils.graphics_utils import focal2fov
from deps.gsplats.utils.system_utils import searchForMaxIteration
from deps.gsplats.scene.gaussian_model import GaussianModel
from deps.gsplats.scene.cameras import Camera as GSCamera
from src.polyviewer.utils.math_utils import quaternion_product


class GSplatPipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def _guess_camera_path(model_path):
    return os.path.join(model_path, "cameras.json")


def subsample_list_to_length(list_val, max_len):
    if len(list_val) < max_len or max_len < 0:
        return list_val
    orig_len = len(list_val)
    indices = range(0, orig_len)
    selected_indices = sorted(np.random.choice(indices, max_len, replace=False))
    res = [list_val[i] for i in selected_indices]
    assert len(res) == max_len
    return res


def load_cameras(model_path, device="cpu", subsample_to_length=-1):
    cam_path = _guess_camera_path(model_path)
    if not os.path.exists(cam_path):
        return None
    with open(cam_path) as f:
        data = json.load(f)
    if subsample_to_length > 0:
        data = subsample_list_to_length(data, int(subsample_to_length))
    return [_json_camera_to_camera(x, device=device) for x in data]


def try_load_camera(model_path, device="cuda"):
    cam_path = _guess_camera_path(model_path)
    if not os.path.exists(cam_path):
        return GSCamera(
            colmap_id=0,
            R=np.array(
                [
                    [-9.9037e-01, 2.3305e-02, -1.3640e-01],
                    [1.3838e-01, 1.6679e-01, -9.7623e-01],
                    [-1.6444e-09, -9.8571e-01, -1.6841e-01],
                ]
            ),
            T=np.array([6.8159e-09, 2.0721e-10, 4.03112e00]),
            FoVx=0.69111120,
            FoVy=0.69111120,
            image=torch.zeros((3, 800, 800)),
            gt_alpha_mask=None,
            image_name="fake",
            uid=0,
        )
    with open(cam_path) as f:
        data = json.load(f)
        raw_camera = data[0]
    return _json_camera_to_camera(raw_camera, device=device)


def _json_camera_to_camera(raw_camera, device, needs_fake_image=False):
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = raw_camera["rotation"]
    tmp[:3, 3] = raw_camera["position"]
    tmp[3, 3] = 1
    C2W = np.linalg.inv(tmp)
    R = C2W[:3, :3]
    T = C2W[:3, 3]
    if "transform" in raw_camera.keys():
        perm = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R = perm @ R
        T = perm @ T
    R = np.transpose(R)
    width = raw_camera["width"]
    height = raw_camera["height"]
    fovx = focal2fov(raw_camera["fx"], width)
    fovy = focal2fov(raw_camera["fy"], height)
    return GSCamera(
        colmap_id=0,
        R=R,
        T=T,
        FoVx=fovx,
        FoVy=fovy,
        image=torch.zeros((3, height, width)) if needs_fake_image else None,
        data_device=device,
        gt_alpha_mask=None,
        image_name="fake",
        uid=0,
        width=width,
        height=height,
    )


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    inv_rotation = np.eye(4)
    inv_rotation[:3, :3] = np.linalg.inv(transform[:3, :3])
    inv_translation = np.eye(4)
    inv_translation[:3, 3] = -transform[:3, 3]
    inv_world = inv_translation @ inv_rotation
    return inv_world


def transform_camera(camera: GSCamera, transform: np.ndarray, inverse=True) -> GSCamera:
    if inverse:
        transform = inverse_transform(transform)
    new_E = (camera.world_view_transform.cpu().numpy().transpose() @ transform).astype(
        np.float32
    )
    new_camera = GSCamera(
        colmap_id=camera.colmap_id,
        R=camera.R,
        T=camera.T,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=None,
        gt_alpha_mask=None,
        image_name=camera.image_name,
        uid=camera.uid,
        trans=camera.trans,
        scale=camera.scale,
        data_device=camera.data_device,
        E=new_E,
        width=camera.image_width,
        height=camera.image_height,
    )
    return new_camera


def to_np(x: torch.Tensor):
    return x.detach().cpu().numpy()


def toggle_off_gspalts(model: GaussianModel, mask: torch.BoolTensor):
    model._opacity[mask] = -10000.0


def toggle_on_gspalts(
    model: GaussianModel, mask: torch.BoolTensor, restored_buffer: torch.FloatTensor
):
    model._opacity[mask] = restored_buffer[mask]


def project_gaussian_means_to_2d(model: GaussianModel, gsplat_cam: GSCamera):
    gaussians_pos = model.get_xyz
    N = gaussians_pos.shape[0]
    ones_padding = gaussians_pos.new_ones(N, 1)
    xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
    cam_view_projection_matrix = gsplat_cam.full_proj_transform.T[None].expand(N, 4, 4)
    transformed_xyz = cam_view_projection_matrix @ xyz_homogeneous
    transformed_xyz = transformed_xyz.squeeze(-1)
    transformed_xyz /= transformed_xyz[:, -1:]
    return transformed_xyz


def highlight_gsplat_selection(
    features_dc: torch.Tensor, mask: torch.BoolTensor, selection_color: torch.Tensor
):
    features = skimage.color.rgb2hsv(features_dc[mask].detach().cpu().numpy())
    selection_hsv = skimage.color.rgb2hsv(selection_color[..., :3].cpu().numpy())[
        0, 0, :
    ]
    features[..., :1] = selection_hsv[:1]
    features = skimage.color.hsv2rgb(features)
    features_dc[mask] = (
        torch.from_numpy(features).to(features_dc.device).to(features_dc.dtype)
    )
    alpha = 0.5
    features_dc[mask] = (
        features_dc[mask] * (1 - alpha) + selection_color[..., :3] * alpha
    )


def compute_bounding_box(model: GaussianModel, mask=None, to_numpy=False):
    gaussians_pos = model.get_xyz
    if mask is not None:
        gaussians_pos = gaussians_pos[mask, ...]
    min_val = gaussians_pos.min(dim=0).values
    max_val = gaussians_pos.max(dim=0).values
    if to_numpy:
        min_val = min_val.detach().cpu().numpy()
        max_val = max_val.detach().cpu().numpy()
    return min_val, max_val


def transform_xyz(xyz: torch.Tensor, se_transform: torch.Tensor):
    return (
        se_transform[None, :3, :3] @ (xyz[:, :, None] + se_transform[None, :3, 3:])
    ).squeeze(-1)


def transform_rot(rot: torch.Tensor, se_transform: torch.Tensor):
    np_quaternion = glm.quat_cast(
        glm.mat3(se_transform.cpu().numpy().astype(np.float32))
    )
    rotation_quat = -rot.new_tensor(np_quaternion)[None]
    return quaternion_product(rotation_quat, rot)


def transform_gs(model: GaussianModel, se_transform: torch.Tensor):
    model._xyz = transform_xyz(model._xyz, se_transform)
    model._rotation = transform_rot(model._rotation, se_transform)
