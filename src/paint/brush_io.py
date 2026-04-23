# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dotenv import dotenv_values
import glob
import logging
import os.path
import torch
import torchvision.utils
import yaml
from deps.gsplats.scene import GaussianModel

logger = logging.getLogger(__name__)


def default_save_path():
    config = dotenv_values(".env")
    path = config.get("DEFAULT_BRUSH_SAVE_DIR")
    if path is None:
        path = "/tmp"
    return path


def _to_meta_path(ply_path, extension):
    parts = ply_path.split(".")
    if parts[-1] != "ply":
        logger.warning(f'Unexpected extension "{parts[-1]}"; expected "ply"')
    parts[-1] = extension
    return ".".join(parts)


def _to_yml_path(ply_path):
    return _to_meta_path(ply_path, "yml")


def _to_pt_path(ply_path):
    return _to_meta_path(ply_path, "pt")


def _to_jpg_path(ply_path):
    return _to_meta_path(ply_path, "jpg")


def import_brushes(directory):
    path_list = glob.glob(os.path.join(directory, "**", f"*.ply"), recursive=True)

    def _get_name(path):
        return ".".join(os.path.basename(path).split(".")[:-1])

    return {_get_name(x): import_brush(x) for x in path_list}


def import_brush(ply_path, sh_degree=3):
    model = GaussianModel(sh_degree)
    model.load_ply(ply_path)
    meta = {}
    yml_path = _to_yml_path(ply_path)
    if os.path.exists(yml_path):
        with open(yml_path, "r") as file:
            meta = yaml.safe_load(file)
    pt_path = _to_pt_path(ply_path)
    if os.path.exists(pt_path):
        meta.update(torch.load(pt_path))
    preview = None
    jpg_path = _to_jpg_path(ply_path)
    if os.path.exists(jpg_path):
        preview = torchvision.io.read_image(jpg_path)
    return model, meta, preview


def export_brush(ply_path, model: GaussianModel, meta=None, preview=None):
    model.save_ply(ply_path)
    if preview is not None:
        torchvision.utils.save_image(preview, _to_jpg_path(ply_path))
    if meta is not None:
        simple_meta = {}
        torch_meta = {}
        for k, v in meta.items():
            if torch.is_tensor(v):
                torch_meta[k] = v
            else:
                simple_meta[k] = v
        if len(simple_meta) > 0:
            with open(_to_yml_path(ply_path), "w") as file:
                yaml.dump(simple_meta, file)
        if len(torch_meta) > 0:
            torch.save(torch_meta, _to_pt_path(ply_path))
