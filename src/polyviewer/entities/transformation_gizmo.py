# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import polyscope as ps
import glm
import copy


class TransformationGizmo:
    def __init__(self):
        self.point_cloud = ps.register_point_cloud(
            "transformation_gizmo", np.zeros([1, 3]), transparency=0.0, enabled=True
        )
        self.canonical_pos = None
        self.canonical_rotation = None
        self.canonical_scaling = None
        self.compensated_transform = None
        self._gizmo_size = 0.5
        self.is_visible = False

    def _set_gizmo_enabled(self, enabled: bool):
        if hasattr(self.point_cloud, "set_transform_gizmo_enabled"):
            self.point_cloud.set_transform_gizmo_enabled(enabled)
        elif enabled and hasattr(self.point_cloud, "enable_transformation_gizmo"):
            self.point_cloud.enable_transformation_gizmo()
        elif not enabled and hasattr(self.point_cloud, "disable_transformation_gizmo"):
            self.point_cloud.disable_transformation_gizmo()
        else:
            raise AttributeError(
                "Polyscope PointCloud does not expose a supported transformation gizmo API"
            )

    def show(self, pts: torch.Tensor, rotation: torch.Tensor, scaling: torch.Tensor):
        gizmo_center = (pts.max(dim=0)[0] + pts.min(dim=0)[0]) / 2.0
        gizmo_center = gizmo_center.cpu().numpy()
        gizmo_scale = np.array(
            glm.scale(np.array([self._gizmo_size, self._gizmo_size, self._gizmo_size]))
        )
        transform = np.array(glm.translate(gizmo_center)) @ gizmo_scale
        self.point_cloud.set_transform(transform)
        self._set_gizmo_enabled(True)
        self.canonical_pos = copy.deepcopy(pts)
        self.canonical_rotation = copy.deepcopy(rotation)
        self.canonical_scaling = copy.deepcopy(scaling)
        self.compensated_transform = transform
        self.is_visible = True

    def hide(self):
        self._set_gizmo_enabled(False)
        self.is_visible = False

    def get_transform(self):
        return np.array(
            self.point_cloud.get_transform() @ glm.inverse(self.compensated_transform)
        )
