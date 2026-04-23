#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        E=None,
        width=-1,
        height=-1,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        if image is None:
            self.original_image = None
            assert (
                width > 0 and height > 0
            ), "Must set width, height if not providing image"
            self.image_width = width
            self.image_height = height
        else:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            assert image is not None, "Must set image if setting gt mask"
            # TODO: why does this make sense? It just makes transparent areas darker
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            pass  # Why was this even needed here?
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if E is None:
            self.world_view_transform = (
                torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            )
        else:
            self.world_view_transform = (
                torch.tensor(E).transpose(0, 1).cuda()
            )  # TODO: does not use data_device

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def __str__(self):
        # Print the variables directly used to set GaussianRasterizationSettings first
        main_attrs = [
            "camera_center",
            "FoVx",
            "FoVy",
            "image_height",
            "image_width",
            "world_view_transform",
            "full_proj_transform",
        ]
        attrs = sorted(list([x for x in self.__dict__.keys() if x[0] != "_"]))
        for att in main_attrs:
            attrs.remove(att)
        return (
            f"gsplat Camera: \n"
            + "\n".join([f"{k}: {getattr(self, k)}" for k in main_attrs])
            + "\n"
            + "\n".join([f"{k}: {getattr(self, k)}" for k in attrs])
        )


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
