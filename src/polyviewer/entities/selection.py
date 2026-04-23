# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
import torch

if TYPE_CHECKING:
    from payload import CallbackPayload


def regenerate_identifier():
    return torch.randint(0, 2**32, (1,))


@dataclass
class GSplatSelection:
    mask: torch.BoolTensor = None
    identifier: torch.IntTensor = regenerate_identifier()

    @torch.no_grad()
    def select(self, payload: CallbackPayload):
        from src.paint.render import project_gaussian_means_to_2d

        transformed_xyz = project_gaussian_means_to_2d(payload.model, payload.camera)
        x0, y0, x1, y1 = payload.drag_bounds
        self.mask = (
            (transformed_xyz[:, 0] >= x0)
            & (transformed_xyz[:, 0] <= x1)
            & (transformed_xyz[:, 1] >= y0)
            & (transformed_xyz[:, 1] <= y1)
        )
        enabled_gaussians = self.mask.new_zeros(self.mask.shape[0], dtype=torch.bool)
        for segment in payload.segments.values():
            if segment.is_enabled:
                enabled_gaussians |= segment.mask
        self.mask &= enabled_gaussians
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def add(self, payload: CallbackPayload):
        other = GSplatSelection().select(payload)
        if self.mask is not None:
            self.mask |= other.mask
        else:
            self.mask = other.mask
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def remove(self, payload: CallbackPayload):
        other = GSplatSelection().select(payload)
        if self.mask is not None:
            self.mask &= ~other.mask
        else:
            self.mask = other.mask & False
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def intersect(self, payload: CallbackPayload):
        other = GSplatSelection().select(payload)
        if self.mask is not None:
            self.mask &= other.mask
        else:
            self.mask = other.mask & False
        self.identifier = regenerate_identifier()
        return self

    def reset(self):
        self.mask = None
        self.identifier = regenerate_identifier()
        return self

    def select_all(self, payload: CallbackPayload):
        gaussians_pos = payload.model.get_xyz
        N = gaussians_pos.shape[0]
        self.mask = gaussians_pos.new_ones(N, dtype=torch.bool)
        self.identifier = regenerate_identifier()
        return self

    def set_mask(self, mask):
        self.mask = mask
        self.identifier = regenerate_identifier()

    def __len__(self):
        return 0 if self.mask is None else torch.sum(self.mask)
