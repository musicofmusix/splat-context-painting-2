# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import polyscope as ps
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict
from deps.gsplats.gaussian_renderer import GaussianModel
from src.segment.selection import SplatPointSelection
from src.segment.entities import Segment


class TransformationGizmo: ...


class GuiMode(Enum):
    VIEW = 0
    SELECT = 1
    EDIT = 2
    MODAL_OPEN = 3
    ORIENT_TOOL = 4


@dataclass
class CallbackPayload:
    model: GaussianModel = None
    model_transform: torch.Tensor = None
    camera: ps.CameraParameters = None
    last_selection: SplatPointSelection = None
    selection_preview: SplatPointSelection = None
    last_context_selection: object = None
    drag_bounds: Tuple[float, float, float, float] = None
    segments: Dict[str, Segment] = None
    gui_mode: GuiMode = None
    transformation_gizmo: TransformationGizmo = None
