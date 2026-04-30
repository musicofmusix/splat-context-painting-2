# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
import polyscope as ps

from src.paint.gui import PSGUI, DEFAULT_CONFIG
from deps.gsplats.gaussian_renderer import render
from deps.gsplats.scene.gaussian_model import GaussianModel
from src.polyviewer.utils.gsplat_utils import GSplatPipelineParams


def pick_scene_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select scene PLY file",
        initialdir=os.path.dirname(os.path.abspath(__file__)),
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def main():
    parser = argparse.ArgumentParser(description="Polyscope GUI for Splat Painting")
    parser.add_argument("--scene", type=str, help="Path to the ply file to load")
    args = parser.parse_args()

    scene_path = args.scene
    if not scene_path:
        scene_path = pick_scene_file()
    if not scene_path:
        print("No scene file selected. Exiting.")
        sys.exit(0)

    background = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(0)
    render_fn = render
    gaussians.load_ply(scene_path)
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
        scene_path=scene_path,
    )
    while not ps.window_requests_close():
        gui.refresh_canvas()

if __name__ == "__main__":
    main()
