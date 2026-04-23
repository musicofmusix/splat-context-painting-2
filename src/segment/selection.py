# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from abc import ABC, abstractmethod
import enum
import logging
import torch

logger = logging.getLogger(__name__)


def regenerate_identifier():
    return torch.randint(0, 2**32, (1,)).to(torch.int32)


class SelectAction(str, enum.Enum):
    NEW = "NEW"
    ADD = "ADD"
    REMOVE = "REMOVE"
    INTERSECT = "INTERSECT"


class Selection(ABC):
    def __init__(self):
        self.identifier: torch.IntTensor = regenerate_identifier()

    def apply(self, other, action: SelectAction):
        if action == SelectAction.NEW:
            return self.select(other)
        elif action == SelectAction.ADD:
            return self.add(other)
        elif action == SelectAction.REMOVE:
            return self.remove(other)
        elif action == SelectAction.INTERSECT:
            return self.intersect(other)
        else:
            raise RuntimeError(f"Unknown action {action}")

    @abstractmethod
    def select(self, other: Selection):
        raise NotImplementedError

    @abstractmethod
    def add(self, other: Selection):
        raise NotImplementedError

    @abstractmethod
    def remove(self, other: Selection):
        raise NotImplementedError

    @abstractmethod
    def intersect(self, other: Selection):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def select_all(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class DenseSelection(Selection):
    def __init__(self, mask: torch.BoolTensor):
        super().__init__()
        assert mask is not None
        self.mask: torch.BoolTensor = mask

    @torch.no_grad()
    def count(self):
        return self.mask.sum().item()

    @torch.no_grad()
    def select(self, other: DenseSelection):
        self.mask = other.mask
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def add(self, other: DenseSelection):
        if self.mask is not None:
            self.mask |= other.mask
        else:
            self.mask = other.mask
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def remove(self, other: DenseSelection):
        if self.mask is not None:
            self.mask &= ~other.mask
        else:
            self.mask = other.mask & False
        self.identifier = regenerate_identifier()
        return self

    @torch.no_grad()
    def intersect(self, other: DenseSelection):
        if self.mask is not None:
            self.mask &= other.mask
        else:
            self.mask = other.mask & False
        self.identifier = regenerate_identifier()
        return self

    def reset(self):
        self.mask[:] = False
        self.identifier = regenerate_identifier()
        return self

    def select_all(self):
        self.mask[:] = True
        self.identifier = regenerate_identifier()
        return self

    def set_mask(self, mask):
        self.mask = mask
        self.identifier = regenerate_identifier()
        return self

    def __len__(self):
        return 0 if self.mask is None else torch.sum(self.mask)


class SplatPointSelection(DenseSelection):
    def __init__(self, mask):
        super().__init__(mask)

    @staticmethod
    def from_empty(num_points, device) -> SplatPointSelection:
        return SplatPointSelection(
            torch.zeros((num_points,), device=device).to(torch.bool)
        )

    @staticmethod
    def from_point_mask(mask) -> SplatPointSelection:
        return SplatPointSelection(mask)

    @staticmethod
    def from_image_mask(splatted_points, image_mask) -> SplatPointSelection:
        assert len(image_mask.shape) == 2, f"{image_mask.shape}"
        dims = (
            torch.tensor([image_mask.shape[1], image_mask.shape[0]])
            .to(splatted_points.dtype)
            .to(splatted_points.device)
            .reshape(-1, 2)
        )
        coord_xy = torch.round((splatted_points[:, :2] + 1) * 0.5 * dims).to(torch.long)
        coord_invalid = (
            (coord_xy[:, 0] < 0)
            & (coord_xy[:, 1] < 0)
            & (coord_xy[:, 0] >= image_mask.shape[1])
            & (coord_xy[:, 1] >= image_mask.shape[0])
        )
        coord_xy[:, 0] = coord_xy[:, 0].clamp(0, image_mask.shape[1] - 1)
        coord_xy[:, 1] = coord_xy[:, 1].clamp(0, image_mask.shape[0] - 1)
        in_mask = image_mask[coord_xy[:, 1], coord_xy[:, 0]]
        mask = in_mask & ~coord_invalid
        return SplatPointSelection(mask)

    @staticmethod
    @torch.no_grad()
    def from_bounds(splatted_points, bounds) -> SplatPointSelection:
        x0, y0, x1, y1 = bounds
        mask = (
            (splatted_points[:, 0] >= x0)
            & (splatted_points[:, 0] <= x1)
            & (splatted_points[:, 1] >= y0)
            & (splatted_points[:, 1] <= y1)
        )
        return SplatPointSelection(mask.to(torch.bool))
