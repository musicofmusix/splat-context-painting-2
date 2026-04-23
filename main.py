# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import numpy as np
import torch
import polyscope as ps

from src.paint.gui import PSGUI, DEFAULT_CONFIG
from deps.gsplats.gaussian_renderer import render
from deps.gsplats.scene.gaussian_model import GaussianModel
from src.polyviewer.utils.gsplat_utils import GSplatPipelineParams


def main():
    parser = argparse.ArgumentParser(description="Polyscope GUI for Splat Painting")
    parser.add_argument("--scene", type=str, help="Path to the ply file to load")
    parser.add_argument(
        "--brush", type=str, help="Path to the directory with brush data"
    )
    args = parser.parse_args()

    background = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(0)
    render_fn = render
    gaussians.load_ply(args.scene)
    pipe = GSplatPipelineParams()
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": background,
    }

    gui = PSGUI(
        DEFAULT_CONFIG,
        None,
        render_fn=render_fn,
        render_kwargs=render_kwargs,
        scene_path=args.scene,
        brush_path=args.brush
    )
    while not ps.window_requests_close():
        gui.refresh_canvas()

if __name__ == "__main__":
    main()
