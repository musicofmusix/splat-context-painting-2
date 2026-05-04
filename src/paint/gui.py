# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import math
from concurrent.futures import ThreadPoolExecutor
import glm
import os
import time
from src.paint.render import *
from src.paint.gsplat import *
from src.paint.geometry import *
from src.paint.curve import *
from typing import Tuple
from dataclasses import dataclass
from src.polyviewer.entities.payload import CallbackPayload, GuiMode
from src.polyviewer.entities.selection import GSplatSelection
from src.polyviewer.entities.transformation_gizmo import TransformationGizmo
from src.polyviewer.gui.event_handlers.drag_handler import DragHandler
from src.polyviewer.entities.segment import Segment
import polyscope as ps
import polyscope.imgui as psim
from src.paint.brush_io import import_brush, export_brush
from deps.gsplats.scene.gaussian_model import GaussianModel

from src.context.dino import dinov3_process
from deps.gsplats.utils.sh_utils import eval_sh

DEFAULT_DEVICE = torch.device("cuda")
MAX_DEPTH = 1000.0

# Bbox wireframe overlay constants.
# Edge ordering matches render_clip_box(); each entry is (face_a, face_b).
# Face indices: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
BBOX_EDGE_FACES = [
    (2, 4), (0, 4), (0, 2), (1, 4), (1, 2),
    (0, 3), (3, 4), (2, 5), (0, 5), (1, 3),
    (1, 5), (3, 5),
]
def imgui_col(r, g, b, a=255):
    return (a << 24) | (b << 16) | (g << 8) | r


BBOX_SOLID_COL = imgui_col(255, 255, 255, 220)
BBOX_DASH_COL  = imgui_col(255, 255, 255, 70)


UP_AXIS_VECS = {
    "x_up": np.array([1., 0., 0.]), "neg_x_up": np.array([-1., 0., 0.]),
    "y_up": np.array([0., 1., 0.]), "neg_y_up": np.array([0., -1., 0.]),
    "z_up": np.array([0., 0., 1.]), "neg_z_up": np.array([0., 0., -1.]),
}

# fit oriented bbox
def fit_obb(pts, up):
    center = pts.mean(axis=0)
    # Project out the up component → 2D horizontal coords for PCA
    horiz = pts - np.outer(pts @ up, up)
    _, _, Vt = np.linalg.svd(horiz - horiz.mean(axis=0), full_matrices=False)
    ax0 = Vt[0]                    # primary horizontal axis
    ax1 = np.cross(up, ax0)        # secondary horizontal axis, orthogonal by construction
    ax1 /= np.linalg.norm(ax1)
    axes = np.stack([ax0, ax1, up])  # rows: 3 orthonormal axes
    proj = (pts - center) @ axes.T
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]],
        [hi[0], hi[1], lo[2]], [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]], [hi[0], hi[1], hi[2]],
    ])
    nodes = (center + corners @ axes).astype(np.float32)
    normals = np.vstack([-axes[0], axes[0], -axes[1], axes[1], -axes[2], axes[2]]).astype(np.float32)
    return nodes, normals


def draw_dashed_line(draw_list, p0, p1, col, thickness, dash=5.0, gap=4.0):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-3:
        return
    ux, uy = dx / length, dy / length
    t = 0.0
    while t < length:
        t1 = min(t + dash, length)
        draw_list.AddLine(
            (p0[0] + ux * t,  p0[1] + uy * t),
            (p0[0] + ux * t1, p0[1] + uy * t1),
            col, thickness,
        )
        t += dash + gap


@dataclass
class UIConstants:
    SUPPORTED_RENDER_MODES: Tuple[str, ...] = ("render", "depth", "embedding", "heatmap")
    SUPPORTED_SELECTION_MODES: Tuple[str, ...] = ("conn_comps", "iterative_bbox")
    SUPPORTED_DISPLAY_MODES: Tuple[str, ...] = ("scene", "brush")

@dataclass
class UIState:
    gui_mode: GuiMode = GuiMode.VIEW
    selection_mode_idx: int = 1
    display_mode_idx: int = 0
    render_mode_idx: int = 0
    prev_render_mode_idx: int = None
    scene_path: str = None
    save_path: str = ""
    load_path: str = ""
    enable_key_bindings: bool = False
    show_selection_overlays: bool = True

@dataclass
class RenderState:
    canvas_color_renderbuffer: any = None
    canvas_depth_renderbuffer: any = None
    canvas_scalar_renderbuffer: any = None
    canvas_size: Tuple = None
    canvas_res_downsample: int = 1
    live_update: bool = True
    is_rendering_enabled: bool = True
    curr_rb: torch.Tensor = None
    curr_depth: torch.Tensor = None
    dist_cam: torch.Tensor = None
    background: torch.Tensor = None

@dataclass
class GUIConfig:
    program_name: str = "GSplats Viewer"
    up_axis: str = "neg_y_up"
    front_axis: str = "neg_z_front"
    navigation_style: str = "free"
    enable_vsync: bool = False
    max_fps: int = -1
    background_color: Tuple = (0.0, 0.0, 0.0)
    window_size: Tuple = (1280, 720)

DEFAULT_CONFIG = GUIConfig(
    up_axis="neg_y_up", front_axis="neg_z_front", navigation_style="turntable"
)

def _as_device_tensor(val):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.clone().detach().to(device=DEFAULT_DEVICE, dtype=torch.float32)
    return torch.tensor(val, device=DEFAULT_DEVICE, dtype=torch.float32)

def _mat3_to_quat(mat, ref_tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()
    return ref_tensor.new_tensor(glm.quat_cast(glm.mat3(mat.astype(np.float32))))[None]

class BrushManager:
    def __init__(self, device):
        self.device = device
        self.brush_gs: GaussianModel = None
        self.original_brush_gs: GaussianModel = None
        self.stroke_pca_directions: torch.Tensor = torch.eye(3, device=self.device, dtype=torch.float32)
        self.original_stroke_pca_directions: torch.Tensor = torch.eye(3, device=self.device, dtype=torch.float32)

    def load_brush(self, path_to_ply):
        self.brush_gs = None
        self.original_brush_gs = None
        self.stroke_pca_directions = torch.eye(3, device=self.device, dtype=torch.float32)
        self.original_stroke_pca_directions = torch.eye(3, device=self.device, dtype=torch.float32)
        if path_to_ply is None:
            return
        
        self.original_brush_gs, pca_dirs, _ = import_brush(os.path.join(os.getcwd(), path_to_ply))
        self.brush_gs = copy.deepcopy(self.original_brush_gs)
        
        if pca_dirs and "pca_directions" in pca_dirs:
            self.stroke_pca_directions = pca_dirs["pca_directions"]
            self.original_stroke_pca_directions = self.stroke_pca_directions.clone().detach()
            self.stroke_pca_directions = self.stroke_pca_directions.detach()
        else:
            self.stroke_pca_directions = torch.eye(3, device=self.device, dtype=torch.float32)
            self.original_stroke_pca_directions = torch.eye(3, device=self.device, dtype=torch.float32)

    def save_brush(self, path_to_ply):
        if self.brush_gs is not None:
            if path_to_ply is None:
                path_to_ply = "data/brushes/brush.ply"
            brush_dict = {"pca_directions": self.original_stroke_pca_directions}
            export_brush(
                os.path.join(os.getcwd(), path_to_ply),
                self.original_brush_gs,
                meta=brush_dict,
            )

    def clear_brush(self):
        self.brush_gs = None
        self.original_brush_gs = None
        self.stroke_pca_directions = torch.eye(3, device=self.device, dtype=torch.float32)
        self.original_stroke_pca_directions = torch.eye(3, device=self.device, dtype=torch.float32)

    def center_original_brush(self):
        if self.original_brush_gs is not None:
            self.original_brush_gs._xyz = (
                self.original_brush_gs._xyz
                - self.original_brush_gs._xyz.mean(dim=0)
            )

    def commit_transform(self):
        if self.brush_gs is None or self.original_brush_gs is None:
            return
        self.original_brush_gs._xyz = self.brush_gs._xyz.clone()
        self.original_brush_gs._scaling = self.brush_gs._scaling.clone()
        self.original_brush_gs._rotation = self.brush_gs._rotation.clone()
        self.original_stroke_pca_directions = self.stroke_pca_directions.clone()

    def reset_transform(self):
        if self.brush_gs is None or self.original_brush_gs is None:
            return
        self.brush_gs._xyz = self.original_brush_gs._xyz.clone()
        self.brush_gs._scaling = self.original_brush_gs._scaling.clone()
        self.brush_gs._rotation = self.original_brush_gs._rotation.clone()
        self.stroke_pca_directions = self.original_stroke_pca_directions.clone()

class PSGUI:
    def __init__(
        self,
        conf: GUIConfig,
        scene_bbox,
        render_fn,
        render_kwargs,
        scene_path=None,
    ):
        self.ui_constants = UIConstants()
        self.ui_state = UIState()
        self.render_state = RenderState()
        self.brush_manager = BrushManager(DEFAULT_DEVICE)

        self.object_name = "gsplat object"
        self.prev_scene = None

        self.sphere_center = torch.tensor(
            [0.0, 0.0, 0.0], device=DEFAULT_DEVICE, dtype=torch.float32
        )
        self.sphere_radius = 1
        self.ui_state.scene_path = scene_path
        self.render_state.background = torch.tensor(
            conf.background_color, dtype=torch.float32, device=DEFAULT_DEVICE
        )
        self.render_fn = render_fn
        self.render_kwargs = render_kwargs
        initial_camera = None
        self.model = self.render_kwargs["pc"]
        self.clip_box_xmax = 0.5
        self.clip_box_xmin = -0.5
        self.clip_box_ymax = -0.9
        self.clip_box_ymin = -1.1
        self.clip_box_zmax = -0.2
        self.clip_box_zmin = -0.4
        self.clip_box_coords = [
            [self.clip_box_xmin, self.clip_box_ymin, self.clip_box_zmin],
            [self.clip_box_xmax, self.clip_box_ymax, self.clip_box_zmax],
        ]
        self.clip_box_nodes = None
        self.clip_box_face_normals = None
        self.world_up = UP_AXIS_VECS.get(conf.up_axis, np.array([0., -1., 0.]))
        self.current_hover_normal = None
        self.ground_plane = False
        self.ground_plane_point = torch.tensor(
            [0.0, 0.0, 0.0], device=DEFAULT_DEVICE, dtype=torch.float32
        )
        self.ground_plane_normal = torch.tensor(
            [0.0, 0.0, 1.0], device=DEFAULT_DEVICE, dtype=torch.float32
        )
        self.enable_surface_orientation_estimation = True
        self.use_surface_proxy = False
        ps.set_program_name(conf.program_name)
        ps.set_use_prefs_file(False)
        ps.set_up_dir(conf.up_axis)
        ps.set_front_dir(conf.front_axis)
        ps.set_navigation_style(conf.navigation_style)
        ps.set_enable_vsync(conf.enable_vsync)
        ps.set_max_fps(conf.max_fps)
        ps.set_background_color(conf.background_color)
        ps.set_background_color([1, 1, 1, 0])
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(*conf.window_size)
        ps.set_give_focus_on_show(True)
        ps.set_automatically_compute_scene_extents(False)
        bbox_min = self.model.get_xyz.min(dim=0).values
        bbox_max = self.model.get_xyz.max(dim=0).values
        radius = 1

        # async embedding parameters
        self.embedding_loop_enabled = False
        self.embedding_scene_snapshot = None
        self.embedding_encoder_thread = ThreadPoolExecutor(max_workers=1)
        self.embedding_encoder_job = None
        self.embedding_loop_interval = 0.5
        self.embedding_loop_prevtime = 0.0
        self.embedding_last_camera_mat = None

        # embedding update exponential moving average param
        # higher = trust the past, lower = trust the present
        self.ema_alpha = 0.33
        self.morph_radius = 0.05
        self.heatmap_gamma: float = 5.0  # power applied to per-pixel percentile rank (1=linear, >1=exaggerate top)
        self.voxel_resolution: int = 64  # cells along the longest scene axis for 3D heatmap
        self._use_3d_heatmap: bool = False  # when True, render uses _cached_heatmap_per_gaussian from 3D NCC
        self._seg_bbox_idx = None

        self.context_selection_mask = None
        self._embedding_pca_rgb_size = None

        self._cached_context_mask = None
        self._cached_context_pc = None
        self._cached_scene_mask = None
        self._cached_scene_pc = None
        self._cached_heatmap_per_gaussian = None
        self._dino_pca_mean: torch.Tensor = None   # (DINO_DIM,)  fit on first DINO call
        self._dino_pca_basis: torch.Tensor = None  # (DINO_DIM, EMBEDDING_DIM)
        self._embedding_best_coverage: torch.Tensor = None  # (N,) best 1/patch_density seen per gaussian
        ps.set_bounding_box(bbox_min - radius, bbox_max + radius)
        ps.init()
        
        self.gizmo = TransformationGizmo()
        self.drag_handler = DragHandler()
        self.current_selection: GSplatSelection = GSplatSelection()
        self.selection_identifier = (
            self.current_selection.identifier.detach().cpu().numpy()
        )
        self.ui_state.gui_mode: GuiMode = GuiMode.VIEW
        self.segments: Dict[str, Segment] = self.init_segments()
        self.last_cam_dist = MAX_DEPTH
        self.last_composite_bbox = None
        self.last_rotation = None
        self.blend_percentage = 0.25
        self.brush_stroke_gs = None
        self.z_offset = 0.0
        self.brush_edit_mode = False
        self.conn_comp_thresh = 0.1
        self.W, self.H = ps.get_window_size()
        if initial_camera is not None:  
            view_mat = np.zeros((4, 4))
            view_mat[:3, :3] = initial_camera.R
            view_mat[:3, 3] = initial_camera.T
            view_mat[3, 3] = 1.0
            aspect = ps.get_view_camera_parameters().get_aspect()
            fov_y = initial_camera.FoVy
            fov_x = 2.0 * np.arctan(np.tan(0.5 * fov_y) * aspect)
            ps_cam_param = ps.CameraParameters(
                ps.CameraIntrinsics(
                    fov_vertical_deg=np.degrees(fov_y),
                    fov_horizontal_deg=np.degrees(fov_x),
                ),
                ps.CameraExtrinsics(mat=view_mat),
            )
            ps.set_view_camera_parameters(ps_cam_param)
        else:
            ps.look_at(camera_location=(0.54, 3.93, 0.67), target=(0.68, 4.91, 0.84))
        if scene_bbox is not None:
            bbox_min, bbox_max = scene_bbox
            nodes = np.array(
                [
                    [bbox_min[0], bbox_min[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_max[2]],
                    [bbox_min[0], bbox_max[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_max[2]],
                ]
            )
            edges = np.array(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [1, 4],
                    [1, 5],
                    [2, 6],
                    [2, 4],
                    [3, 5],
                    [3, 6],
                    [4, 7],
                    [5, 7],
                    [6, 7],
                ]
            )
            ps.register_curve_network("bbox", nodes, edges)
        ps.set_user_callback(self.ui_callback)
        self.frame_scene()
        self.refresh_canvas(force=True)

    # Setup / scene state
    def init_segments(self):
        segments = dict()
        N = self.model.get_xyz.shape[0]
        segments["Background"] = Segment(
            mask=self.model.get_xyz.new_ones(N, dtype=torch.bool), name="Background"
        )
        return segments

    # Camera / render pipeline
    def frame_scene(self):
        points = self.model.get_xyz.detach().cpu().numpy()
        if points.shape[0] == 0:
            return

        scene_center = points.mean(axis=0)

        cam_params = ps.get_view_camera_parameters()
        cam_pos = cam_params.get_position()
        look_dir = scene_center - cam_pos
        up_dir = cam_params.get_up_dir()
        fov = cam_params.get_fov_vertical_deg()
        aspect = cam_params.get_aspect()

        distance = compute_distance_for_bounding(points, scene_center, look_dir, up_dir, fov, aspect)
        new_cam_pos = scene_center + (look_dir / np.linalg.norm(look_dir)) * distance
        ps.look_at(new_cam_pos, scene_center)

    def resolve_render_pc(self, mode="render", override_pc=None, composite_brush=True):
        if override_pc is not None:
            return override_pc
        scene_pc = self.render_kwargs["pc"]
        brush_pc = self.brush_manager.brush_gs
        if brush_pc is None:
            return scene_pc
        if self.ui_state.display_mode_idx == 1:
            return brush_pc
        if not composite_brush or mode == "dist_to_cam":
            return scene_pc
        layers = [scene_pc, brush_pc]
        if self.brush_stroke_gs is not None:
            layers.append(self.brush_stroke_gs)
        return composite_gaussians(layers)

    @torch.no_grad()
    def render_gaussians(
        self,
        mode="render",
        camera: ps.CameraParameters = None,
        override_pc=None,
        composite_brush=True,
    ):
        cam = polyscope_to_gsplat_camera(
            camera, downsample_factor=self.render_state.canvas_res_downsample
        )
        pc = self.resolve_render_pc(mode, override_pc, composite_brush)
        render_pkg = self.render_fn(
            viewpoint_camera=cam,
            pc=pc,
            pipe=self.render_kwargs["pipe"],
            bg_color=self.render_kwargs["bg_color"],
        )
        if mode == "passthrough":
            return render_pkg["render"].permute(1, 2, 0).contiguous(), render_pkg["depth"][0]
        if mode == "depth":
            output = render_pkg["depth"][0]
        elif mode == "dist_to_cam":
            background = torch.ones_like(self.render_state.background) * MAX_DEPTH
            world_xyz = self.render_fn(
                viewpoint_camera=cam,
                pc=pc,
                pipe=self.render_kwargs["pipe"],
                bg_color=background,
                override_color=pc.get_xyz,
            )["render"]
            cam_params = ps.get_view_camera_parameters() if camera is None else camera
            cam_pos = torch.tensor(cam_params.get_position(), device=DEFAULT_DEVICE)
            output = torch.sqrt(((world_xyz - cam_pos[:, None, None]) ** 2).sum(dim=0))
        else:
            output = render_pkg["render"]

        if output.ndim == 2:
            output = output[None]
        return output.permute(1, 2, 0).contiguous()

    @torch.no_grad()
    def render_scene_query_buffers(self, camera: ps.CameraParameters = None):
        cam = polyscope_to_gsplat_camera(
            camera, downsample_factor=self.render_state.canvas_res_downsample
        )
        scene_pc = self.render_kwargs["pc"]
        background = torch.ones_like(self.render_state.background) * MAX_DEPTH
        scene_xyz = scene_pc.get_xyz
        query_pkg = self.render_fn(
            viewpoint_camera=cam,
            pc=scene_pc,
            pipe=self.render_kwargs["pipe"],
            bg_color=background,
            override_color=scene_xyz,
        )
        cam_params = ps.get_view_camera_parameters() if camera is None else camera
        cam_pos = torch.tensor(cam_params.get_position(), device=DEFAULT_DEVICE)
        cam_dist = query_pkg["render"] - cam_pos[:, None, None]
        cam_dist = torch.sqrt((cam_dist**2).sum(dim=0))
        depth = query_pkg["depth"][0][None].permute(1, 2, 0).contiguous()
        cam_dist = torch.where(
            depth[..., 0] < MAX_DEPTH - 1,
            cam_dist,
            torch.full_like(cam_dist, MAX_DEPTH),
        )
        return depth, cam_dist

    def refresh_canvas(self, force=False):
        window_w, window_h = ps.get_window_size()
        window_w = window_w // self.render_state.canvas_res_downsample
        window_h = window_h // self.render_state.canvas_res_downsample
        style = self.ui_constants.SUPPORTED_RENDER_MODES[self.ui_state.render_mode_idx]
        if (
            force
            or self.ui_state.prev_render_mode_idx != self.ui_state.render_mode_idx
            or self.render_state.canvas_size != (window_w, window_h)
        ):
            self.ui_state.prev_render_mode_idx = self.ui_state.render_mode_idx
            self.render_state.canvas_size = (window_w, window_h)
            if style == "depth":
                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = MAX_DEPTH
                ps.add_scalar_image_quantity(
                    self.object_name,
                    dummy_vals,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="jet",
                    vminmax=(0, MAX_DEPTH),
                )
                self.render_state.canvas_color_renderbuffer = None
                self.render_state.canvas_scalar_renderbuffer = ps.get_quantity_buffer(
                    self.object_name, "values"
                )
            else:
                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)
                dummy_vals = np.ones((window_h, window_w), dtype=np.float32) * MAX_DEPTH
                ps.add_raw_color_alpha_render_image_quantity(
                    self.object_name,
                    dummy_vals,
                    dummy_image,
                    enabled=self.render_state.is_rendering_enabled,
                )
                self.render_state.canvas_color_renderbuffer = ps.get_quantity_buffer(
                    self.object_name, "colors"
                )
                self.render_state.canvas_depth_renderbuffer = ps.get_quantity_buffer(
                    self.object_name, "depths"
                )
                self.render_state.canvas_scalar_renderbuffer = None

        if self.brush_manager.brush_gs is not None:
            query_depth, query_cam_dist = self.render_scene_query_buffers()
        else:
            query_depth, query_cam_dist = self.render_state.curr_depth, self.render_state.dist_cam

        if style == "heatmap":
            scene = self.render_kwargs["pc"]
            cam = polyscope_to_gsplat_camera(
                None, downsample_factor=self.render_state.canvas_res_downsample
            )
            if (self._use_3d_heatmap
                    and self._cached_heatmap_per_gaussian is not None
                    and self._cached_heatmap_per_gaussian.shape[0] == scene._xyz.shape[0]):
                scores = self._cached_heatmap_per_gaussian.clamp(0.0, 1.0)
                override_color = scores.unsqueeze(1).expand(-1, 3).contiguous()
                black_bg = torch.zeros(3, device=DEFAULT_DEVICE)
                render_pkg_out = self.render_fn(
                    viewpoint_camera=cam, pc=scene,
                    pipe=self.render_kwargs["pipe"],
                    bg_color=black_bg,
                    override_color=override_color,
                )
                rgb_out = render_pkg_out["render"].permute(1, 2, 0).contiguous()
                scene_depth = render_pkg_out["depth"][0].unsqueeze(-1)
                self.render_state.curr_rb = rgb_out
                self.render_state.curr_depth = query_depth
                self.render_state.dist_cam = query_cam_dist
                rgb_rgba = torch.cat((rgb_out, torch.ones_like(rgb_out[:, :, 0:1])), dim=-1)
                self.render_state.canvas_color_renderbuffer.update_data_from_device(rgb_rgba.detach())
                self.render_state.canvas_depth_renderbuffer.update_data_from_device(scene_depth.detach())
            ps.frame_tick()
            return

        cam = polyscope_to_gsplat_camera(
            None, downsample_factor=self.render_state.canvas_res_downsample
        )
        display_pc = self.resolve_render_pc("render", composite_brush=True)
        style_override_color = self.get_render_style_color(display_pc, style)
        style_override_color = self.apply_selection_colors(display_pc, cam, style_override_color)
        render_pkg = self.render_fn(
            viewpoint_camera=cam,
            pc=display_pc,
            pipe=self.render_kwargs["pipe"],
            bg_color=self.render_kwargs["bg_color"],
            override_color=style_override_color,
        )
        style_render = render_pkg["render"]
        depth_output = render_pkg["depth"][0][None]
        rb = style_render.permute(1, 2, 0).contiguous()
        depth = depth_output.permute(1, 2, 0).contiguous()
        self.render_state.curr_rb = rb
        self.render_state.curr_depth = query_depth
        self.render_state.dist_cam = query_cam_dist
        if style == "depth":
            self.render_state.canvas_scalar_renderbuffer.update_data_from_device(depth.detach())
        else:
            rb = torch.cat((rb, torch.ones_like(rb[:, :, 0:1])), dim=-1)
            self.render_state.canvas_color_renderbuffer.update_data_from_device(rb.detach())
            self.render_state.canvas_depth_renderbuffer.update_data_from_device(depth.detach())
        ps.frame_tick()

    def apply_selection_colors(self, pc, cam, override_color):
        subject_mask = self.current_selection.mask
        context_mask = self.context_selection_mask
        has_subject = subject_mask is not None and torch.any(subject_mask)
        has_context = context_mask is not None and torch.any(context_mask)
        if not has_subject and not has_context:
            return override_color

        if override_color is None:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - cam.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            override_color = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5, 0.0)
        else:
            override_color = override_color.clone()

        N = self.render_kwargs["pc"].get_xyz.shape[0]
        if has_subject:
            override_color[:N][subject_mask] = (
                override_color[:N][subject_mask] * 0.3
                + torch.tensor([1.0, 0.15, 0.15], device=DEFAULT_DEVICE) * 0.7
            )
        if has_context:
            override_color[:N][context_mask] = (
                override_color[:N][context_mask] * 0.3
                + torch.tensor([0.15, 0.15, 1.0], device=DEFAULT_DEVICE) * 0.7
            )
        return override_color

    def get_render_style_color(self, pc, style):
        if style == "render":
            return None

        valid = pc._has_embedding.bool()
        override_color = torch.zeros(
            (pc._embedding.shape[0], 3), device=pc._embedding.device, dtype=torch.float32
        )
        if valid.sum().item() == 0:
            return None

        if style == "embedding":
            emb_rgb = getattr(self, "_embedding_pca_rgb", None)
            if emb_rgb is None or self._embedding_pca_rgb_size != pc._embedding.shape[0]:
                valid_embeddings = pc._embedding[valid]
                emb_rgb = torch.zeros((pc._embedding.shape[0], 3), device=pc._embedding.device, dtype=torch.float32)

                pca_bases = getattr(self, "_embedding_pca_bases", None)
                if pca_bases is None:
                    mean = valid_embeddings.mean(0, keepdim=True)
                    _, _, vh = torch.linalg.svd(valid_embeddings - mean, full_matrices=False)
                    basis = vh[:3]
                    self._embedding_pca_bases = (mean, basis)
                else:
                    mean, basis = pca_bases

                reduced = (valid_embeddings - mean) @ basis.T
                mins = reduced.min(0, keepdim=True).values
                maxs = reduced.max(0, keepdim=True).values
                emb_rgb[valid] = (reduced - mins) / (maxs - mins).clamp_min(1e-8)
                self._embedding_pca_rgb = emb_rgb
                self._embedding_pca_rgb_size = pc._embedding.shape[0]

            override_color[valid] = emb_rgb[valid].to(
                device=override_color.device, dtype=override_color.dtype
            )
            return override_color

        return None

    # Selection / context
    def render_clip_box(self, pts_np):
        self.clip_box_nodes, self.clip_box_face_normals = fit_obb(pts_np, self.world_up)

    def draw_clip_box_overlay(self):
        if self.clip_box_nodes is None:
            return
        W, H = ps.get_window_size()
        nodes_t = torch.tensor(self.clip_box_nodes, device=DEFAULT_DEVICE)
        ndc = project_gaussian_means_to_2d_pos(nodes_t, None).cpu().numpy()
        sx = (ndc[:, 0] + 1) * 0.5 * W
        sy = (ndc[:, 1] + 1) * 0.5 * H

        cam_pos = np.array(ps.get_view_camera_parameters().get_position())
        center = self.clip_box_nodes.mean(axis=0)
        face_front = [
            float(np.dot(n, cam_pos - center)) > 0
            for n in self.clip_box_face_normals
        ]

        edges = [
            (0,1),(0,2),(0,3),(1,4),(1,5),
            (2,6),(2,4),(3,5),(3,6),(4,7),(5,7),(6,7),
        ]
        dl = psim.GetBackgroundDrawList()
        for i, (a, b) in enumerate(edges):
            p0 = (float(sx[a]), float(sy[a]))
            p1 = (float(sx[b]), float(sy[b]))
            fa, fb = BBOX_EDGE_FACES[i]
            if face_front[fa] or face_front[fb]:
                dl.AddLine(p0, p1, BBOX_SOLID_COL, 1.8)
            else:
                draw_dashed_line(dl, p0, p1, BBOX_DASH_COL, 1.4)

    def update_bbox_from_selection(self):
        xyz = self.render_kwargs["pc"].get_xyz
        mask = self.current_selection.mask
        if mask is not None:
            selected_xyz = xyz[mask]
            if selected_xyz.shape[0] == 0:
                return
            if self.ui_state.selection_mode_idx == 1:
                bbox_min = selected_xyz.min(dim=0).values
                bbox_max = selected_xyz.max(dim=0).values
                bbox_min = bbox_min.detach().cpu().numpy()
                bbox_max = bbox_max.detach().cpu().numpy()
                self.clip_box_xmin, self.clip_box_ymin, self.clip_box_zmin = bbox_min
                self.clip_box_xmax, self.clip_box_ymax, self.clip_box_zmax = bbox_max
                self.clip_box_coords = [
                    [self.clip_box_xmin, self.clip_box_ymin, self.clip_box_zmin],
                    [self.clip_box_xmax, self.clip_box_ymax, self.clip_box_zmax],
                ]
                pts_np = selected_xyz.detach().cpu().numpy()
                self.render_clip_box(pts_np)
            else:
                connected_components = find_connected_components(
                    selected_xyz, threshold=self.conn_comp_thresh
                )
                closest_component = find_closest_component(connected_components)
                if closest_component is not None:
                    bbox_min = closest_component.min(dim=0).values
                    bbox_max = closest_component.max(dim=0).values
                    bbox_min = bbox_min.detach().cpu().numpy()
                    bbox_max = bbox_max.detach().cpu().numpy()
                    self.clip_box_xmin, self.clip_box_ymin, self.clip_box_zmin = (
                        bbox_min
                    )
                    self.clip_box_xmax, self.clip_box_ymax, self.clip_box_zmax = (
                        bbox_max
                    )
                    self.clip_box_coords = [
                        [self.clip_box_xmin, self.clip_box_ymin, self.clip_box_zmin],
                        [self.clip_box_xmax, self.clip_box_ymax, self.clip_box_zmax],
                    ]
                    self.render_clip_box(closest_component.detach().cpu().numpy())
        return

    def clear_context_selection(self):
        self.context_selection_mask = None
        ps.register_point_cloud("context_xyz", np.zeros((0, 3)))
        context_pc = ps.get_point_cloud("context_xyz")
        context_pc.set_color((1.0, 1.0, 0.0))

    def recompute_context_selection(self):
        if self.current_selection.mask is None or not torch.any(self.current_selection.mask):
            self.context_selection_mask = None

    def clear_selection(self):
        self.current_selection.reset()
        self.selection_identifier = self.current_selection.identifier.detach().cpu().numpy()
        self.clear_context_selection()
        ps.register_point_cloud("selected_xyz", np.zeros((0, 3)))
        self.clip_box_nodes = None
        self.clip_box_face_normals = None


    # Brush setup / painting
    def save_brush(self, path_to_ply=None):
        if path_to_ply is None or not str(path_to_ply).strip():
            path_to_ply = "data/brushes/brush.ply"

        save_path = os.path.join(os.getcwd(), path_to_ply)
        selection = self.current_selection.mask

        if selection is not None and torch.any(selection):
            selected_gs = copy.deepcopy(self.render_kwargs["pc"])
            apply_mask_to_attributes(selected_gs, selection.detach())
            export_brush(save_path, selected_gs, meta=None)
            return

        self.brush_manager.save_brush(path_to_ply)

    def save_heatmap_snapshot(self):
        import datetime
        pc = self.render_kwargs["pc"]
        N = pc._xyz.shape[0]

        data = {
            "xyz":              pc._xyz.detach().cpu().numpy(),
            "features_dc":      pc._features_dc.detach().cpu().numpy(),
            "features_rest":    pc._features_rest.detach().cpu().numpy(),
            "scaling":          pc._scaling.detach().cpu().numpy(),
            "rotation":         pc._rotation.detach().cpu().numpy(),
            "opacity":          pc._opacity.detach().cpu().numpy(),
            "active_sh_degree": np.array(pc.active_sh_degree),
            "max_sh_degree":    np.array(pc.max_sh_degree),
            "spatial_lr_scale": np.array(pc.spatial_lr_scale),
        }

        emb = pc._embedding.detach().cpu().numpy().copy()
        no_emb = ~pc._has_embedding.bool().cpu().numpy()
        emb[no_emb] = np.nan
        data["embeddings"] = emb

        if self._cached_heatmap_per_gaussian is not None:
            data["heatmap_values"] = self._cached_heatmap_per_gaussian.cpu().numpy()
        else:
            data["heatmap_values"] = np.full(N, np.nan, dtype=np.float32)

        if self.current_selection.mask is not None:
            data["subject_indices"] = torch.nonzero(
                self.current_selection.mask, as_tuple=False
            ).squeeze(1).cpu().numpy()
        else:
            data["subject_indices"] = np.array([], dtype=np.int64)

        if self.context_selection_mask is not None:
            data["context_indices"] = torch.nonzero(
                self.context_selection_mask, as_tuple=False
            ).squeeze(1).cpu().numpy()
        else:
            data["context_indices"] = np.array([], dtype=np.int64)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("snapshots", f"heatmap_snapshot_{ts}.npz")
        np.savez_compressed(path, **data)
        print(f"Snapshot saved at {path}")

    def load_brush(self, path_to_ply):
        self.brush_stroke_gs = None
        self.ui_state.display_mode_idx = 0
        self.brush_manager.load_brush(path_to_ply)
        self.segments = self.init_segments()

    def clip_points(self, points, clip=True):
        points = copy.deepcopy(self.render_kwargs["pc"])
        if self.ui_state.selection_mode_idx == 0:
            mask = (
                (points._xyz[:, 0] >= self.clip_box_xmin)
                & (points._xyz[:, 0] <= self.clip_box_xmax)
                & (points._xyz[:, 1] >= self.clip_box_ymin)
                & (points._xyz[:, 1] <= self.clip_box_ymax)
                & (points._xyz[:, 2] >= self.clip_box_zmin)
                & (points._xyz[:, 2] <= self.clip_box_zmax)
            )
        else:
            mask = self.current_selection.mask.detach()
        if mask is None or mask.sum() == 0:
            if self.brush_manager.original_brush_gs is not None:
                mean = self.brush_manager.original_brush_gs._xyz.mean(dim=0, keepdim=True)
                self.show_pca_vectors(
                    "brush_stroke_center", mean.detach().cpu().numpy(),
                    self.brush_manager.original_stroke_pca_directions,
                    colors=((1,0,0),(0,1,0),(0,0,1)),
                )
            return points
        apply_mask_to_attributes(points, mask)
        self.brush_manager.brush_gs = points
        self.brush_stroke_gs = points
        self.brush_manager.original_brush_gs = copy.deepcopy(points)
        point_cloud = self.brush_manager.original_brush_gs._xyz
        mean = point_cloud.mean(dim=0, keepdim=True)
        self.show_pca_vectors(
            "brush_stroke_center", mean.detach().cpu().numpy(),
            self.brush_manager.original_stroke_pca_directions,
            colors=((1,0,0),(0,1,0),(0,0,1)),
        )
        if clip:
            self.tighten_bbox_clip()
        return points

    def transform_brush(self, transformation_gizmo):
        with torch.no_grad():
            xyz = transformation_gizmo.canonical_pos
            rotation = transformation_gizmo.canonical_rotation
            scaling = self.brush_manager.brush_gs.scaling_activation(
                transformation_gizmo.canonical_scaling
            )
            gizmo_transform = transformation_gizmo.get_transform()
            rs_transform = gizmo_transform[:3, :3]
            scale_transform = np.linalg.norm(rs_transform, axis=0, keepdims=True)
            rotation_transform = rs_transform / scale_transform
            rotation_quat = -_mat3_to_quat(rotation_transform, xyz)
            gizmo_transform = torch.from_numpy(gizmo_transform).to(
                xyz.device, xyz.dtype
            )[None]
            xyz_hom = torch.cat([xyz, xyz.new_ones(xyz.shape[0], 1)], dim=1)
            transformed_xyz = (
                gizmo_transform.expand([xyz_hom.shape[0], 4, 4]) @ xyz_hom[:, :, None]
            )
            self.brush_manager.brush_gs._xyz = transformed_xyz[:, :3, 0]
            self.brush_manager.brush_gs._rotation = quaternion_product(rotation_quat, rotation)
            self.brush_manager.brush_gs._scaling = self.brush_manager.brush_gs.scaling_inverse_activation(
                scaling * scaling.new_tensor(scale_transform)
            )
        return

    def apply_surface_rotation(self, rotation, centre):
        gs = self.brush_manager.brush_gs
        gs._xyz -= centre
        gs._xyz = torch.matmul(gs._xyz, rotation.T)
        rotation_quat = -_mat3_to_quat(rotation, gs._xyz)
        gs._rotation = quaternion_product(rotation_quat, gs._rotation)
        gs._xyz += centre

    def tighten_bbox_clip(self):
        self.clip_box_xmin = self.brush_manager.brush_gs._xyz[:, 0].detach().cpu().min()
        self.clip_box_xmax = self.brush_manager.brush_gs._xyz[:, 0].detach().cpu().max()
        self.clip_box_ymin = self.brush_manager.brush_gs._xyz[:, 1].detach().cpu().min()
        self.clip_box_ymax = self.brush_manager.brush_gs._xyz[:, 1].detach().cpu().max()
        self.clip_box_zmin = self.brush_manager.brush_gs._xyz[:, 2].detach().cpu().min()
        self.clip_box_zmax = self.brush_manager.brush_gs._xyz[:, 2].detach().cpu().max()
        self.clip_box_coords = [
            [self.clip_box_xmin, self.clip_box_ymin, self.clip_box_zmin],
            [self.clip_box_xmax, self.clip_box_ymax, self.clip_box_zmax],
        ]
        return

    def show_pca_vectors(self, cloud_name, center_np, pca_dirs, scale=1.0, colors=((1,0,0),(0,0,1),(0,1,0))):
        cloud = ps.register_point_cloud(cloud_name, center_np)
        for i, color in enumerate(colors):
            v = scale * pca_dirs[:, i].unsqueeze(0).detach().cpu().numpy()
            cloud.add_vector_quantity(f"pca_direction{i+1}", v, vectortype="ambient", color=color, enabled=True)

    def visualize_pca(self):
        if self.brush_manager.brush_gs is None or self.brush_manager.stroke_pca_directions is None:
            return
        if self.ui_state.display_mode_idx != 1:
            return
        max_vals = self.brush_manager.brush_gs._xyz.max(dim=0)[0]
        min_vals = self.brush_manager.brush_gs._xyz.min(dim=0)[0]
        length = 2.0 * torch.norm(max_vals - min_vals).item()
        self.show_pca_vectors(
            "brush axis", np.array([[0.0, 0.0, 0.0]]),
            self.brush_manager.stroke_pca_directions, scale=length,
        )

    def clone_composite(self):
        if psim.IsKeyDown(psim.ImGuiKey_LeftCtrl) and self.brush_manager.brush_gs is not None:
            last_bbox = self.last_composite_bbox
            curr_bbox = [
                self.brush_manager.brush_gs._xyz.min(dim=0).values,
                self.brush_manager.brush_gs._xyz.max(dim=0).values,
            ]
            if last_bbox is None:
                self.brush_stroke_gs = self.brush_manager.brush_gs
                self.last_composite_bbox = curr_bbox
                return
            percentage_intersection = calculate_bbox_intersection_percentage(
                curr_bbox, last_bbox
            )
            if percentage_intersection < self.blend_percentage:
                current_brush_gs = composite_gaussians([self.brush_manager.brush_gs])
                if self.brush_stroke_gs is not None and current_brush_gs is not None:
                    self.brush_stroke_gs = composite_gaussians(
                        [self.brush_stroke_gs, current_brush_gs]
                    )
                self.last_composite_bbox = curr_bbox
        if psim.IsKeyReleased(psim.ImGuiKey_LeftCtrl) and self.brush_stroke_gs is not None:
            self.prev_scene = self.render_kwargs["pc"]
            self.render_kwargs["pc"] = composite_gaussians(
                [self.render_kwargs["pc"], self.brush_stroke_gs]
            )
            self.model = self.render_kwargs["pc"]
            self._invalidate_filtered_pc_cache()
            self.segments: Dict[str, Segment] = self.init_segments()
            self.brush_stroke_gs = self.brush_manager.brush_gs
        return

    def update_transform_from_mouse_pos(self):
        mouse_pos = psim.GetMousePos()
        window_w, window_h = ps.get_window_size()

        if self.render_state.curr_depth is not None:
            depth_render = self.render_state.curr_depth
        else:
            depth_render = self.render_gaussians("depth", composite_brush=False)
        if self.render_state.dist_cam is not None:
            cam_dist_render = self.render_state.dist_cam
        else:
            cam_dist_render = self.render_gaussians("dist_to_cam")
        render_h, render_w = cam_dist_render.shape[:2]
        if (
            mouse_pos[0] < 0
            or mouse_pos[0] >= render_w
            or mouse_pos[1] < 0
            or mouse_pos[1] >= render_h
        ):
            return
        cam_dist = cam_dist_render[int(mouse_pos[1]), int(mouse_pos[0])]
        cam_params = ps.get_view_camera_parameters()
        cam_pos = torch.tensor(cam_params.get_position(), device=DEFAULT_DEVICE)
        world_ray = torch.tensor(
            ps.screen_coords_to_world_ray(mouse_pos),
            device=DEFAULT_DEVICE,
            dtype=torch.float32,
        )
        if cam_dist < MAX_DEPTH - 1 and not self.use_surface_proxy:
            self.last_cam_dist = cam_dist
        elif self.ground_plane:
            cam_dist, _ = ray_plane_intersection(
                self.ground_plane_normal, self.ground_plane_point, cam_pos, world_ray
            )
            self.current_hover_normal = self.ground_plane_normal
        elif self.use_surface_proxy:
            int_elem = ray_sphere_intersection(
                self.sphere_center, self.sphere_radius, cam_pos, world_ray
            )
            if int_elem is not None:
                cam_dist, _, self.current_hover_normal = int_elem
        if cam_dist < 0.0 or cam_dist >= MAX_DEPTH - 1:
            return
        world_pos = cam_pos + cam_dist * world_ray
        if (
            self.use_surface_proxy
            and self.current_hover_normal is not None
            and self.z_offset is not None
        ):
            world_pos -= 1e-1 * self.z_offset * self.current_hover_normal
        if self.brush_manager.brush_gs is not None:
            self.brush_manager.brush_gs._xyz = self.brush_manager.original_brush_gs._xyz.clone()
            self.brush_manager.brush_gs._rotation = self.brush_manager.original_brush_gs._rotation.clone()
            self.brush_manager.brush_gs._scaling = self.brush_manager.original_brush_gs._scaling.clone()
            self.brush_manager.stroke_pca_directions[:] = self.brush_manager.original_stroke_pca_directions[:]
            brush_mask = get_minimal_surface_mask_3d(
                mouse_pos, 15, self.render_kwargs["pc"], depth_render
            )
            if self.use_surface_proxy:
                brush_mask = torch.zeros_like(
                    brush_mask, dtype=brush_mask.dtype, device=brush_mask.device
                )
            if self.enable_surface_orientation_estimation:
                rotation, new_normal, _ = auto_orient_transform(
                    self.render_kwargs["pc"],
                    brush_mask,
                    self.brush_manager.stroke_pca_directions[:, 2].detach().cpu().numpy(),
                )
                if new_normal is None:
                    new_normal = self.current_hover_normal
                    if new_normal is None:
                        return
            else:
                rotation = torch.eye(4, device=DEFAULT_DEVICE, dtype=torch.float32)
                new_normal = None
            lower_centre = find_lower_centre(
                self.brush_manager.brush_gs._xyz, self.brush_manager.stroke_pca_directions[:, 2]
            )
            new_normal = _as_device_tensor(new_normal)
            if brush_mask.sum() != 0:
                scene_pts = self.render_kwargs["pc"]._xyz
                brush_mask_centre = scene_pts[brush_mask].mean(dim=0)
                dist = torch.norm(cam_pos - brush_mask_centre)
                if new_normal is not None:
                    brush_mask_centre += 1e-2 * dist * self.z_offset * new_normal
            else:
                brush_mask_centre = None
            if brush_mask_centre is not None:
                self.brush_manager.brush_gs._xyz += (brush_mask_centre - lower_centre).to(DEFAULT_DEVICE, torch.float32)
            elif self.use_surface_proxy:
                brush_mask_centre = world_pos
                self.brush_manager.brush_gs._xyz += (brush_mask_centre - lower_centre).to(DEFAULT_DEVICE, torch.float32)
            else:
                return
            rotation = rotation.to(device=DEFAULT_DEVICE, dtype=torch.float32)[:3, :3]
            if new_normal is not None and (
                self.current_hover_normal is None
                or not torch.allclose(new_normal, self.current_hover_normal, atol=1e-2)
            ):
                self.apply_surface_rotation(rotation, brush_mask_centre)
                self.brush_manager.stroke_pca_directions = torch.matmul(
                    self.brush_manager.stroke_pca_directions.T, rotation.T
                ).T
                self.current_hover_normal = new_normal
                self.last_rotation = rotation
            elif self.last_rotation is not None:
                self.apply_surface_rotation(self.last_rotation, brush_mask_centre)
        return

    # Embeddings / segmentation
    def backproject_visible_embeddings(self, projected, pre_ndc_depth, patch_embeddings, image_hw, depth_buffer):
        scene = self.render_kwargs["pc"]

        h, w = image_hw

        # can be fractional
        x_coords = (((projected[:, 0] + 1.0) / 2.0) * w).round().long()
        y_coords = (((projected[:, 1] + 1.0) / 2.0) * h).round().long()

        z_buffer = depth_buffer.squeeze(-1)

        # size N boolean mask
        within_bounds = (
            (x_coords >= 0)
            & (x_coords < w)
            & (y_coords >= 0)
            & (y_coords < h)
            & (pre_ndc_depth > 0.0)
        )
        if within_bounds.sum() == 0:
            return 0

        # visible = size N bool mask of gaussians that are within bounds AND on the surface/visible from camera
        visible = torch.zeros(scene._xyz.shape[0], device=DEFAULT_DEVICE, dtype=torch.bool)
        z_match = torch.abs(pre_ndc_depth[within_bounds] - z_buffer[y_coords[within_bounds], x_coords[within_bounds]].squeeze()) < 0.01
        visible[within_bounds] = z_match
        if visible.sum() == 0:
            return 0

        # xs, ys = size M pixel coords list of visible gaussians
        ys = y_coords[visible].to(device=DEFAULT_DEVICE, dtype=torch.long)
        xs = x_coords[visible].to(device=DEFAULT_DEVICE, dtype=torch.long)
        patch_embeddings = patch_embeddings.squeeze(0).to(DEFAULT_DEVICE)

        # embeddings are not image size for patched encoders. need to scale then sample.
        patch_h, patch_w, embed_dim = patch_embeddings.shape
        # grid_sample expects (1, D, H, W); normalise pixel coords to [-1, 1] at patch-cell centres
        grid_x = (xs.float() + 0.5) / w * 2.0 - 1.0
        grid_y = (ys.float() + 0.5) / h * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2)
        feat_map = patch_embeddings.permute(2, 0, 1).unsqueeze(0)  # (1, D, pH, pW)
        # size MxDINO_DIM embeddings for visible gaussians (raw 384-dim)
        sampled_embeddings = torch.nn.functional.grid_sample(
            feat_map, grid, mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(1).T  # (M, DINO_DIM)

        # --- PCA projection 384 → EMBEDDING_DIM ---
        target_dim = scene._embedding.shape[1]
        if self._dino_pca_basis is None:
            if sampled_embeddings.shape[0] < target_dim:
                # Too few visible gaussians to fit a reliable PCA; skip this frame.
                return int(visible.sum().item())
            mean = sampled_embeddings.mean(0)
            _, _, V = torch.pca_lowrank(sampled_embeddings - mean, q=target_dim, niter=4)
            self._dino_pca_mean = mean          # (DINO_DIM,)
            self._dino_pca_basis = V            # (DINO_DIM, EMBEDDING_DIM)
            print(f"[dino pca] fitted {target_dim}-dim basis from {sampled_embeddings.shape[0]} embeddings")
        sampled_embeddings = (sampled_embeddings - self._dino_pca_mean) @ self._dino_pca_basis  # (M, EMBEDDING_DIM)

        # --- coverage weight: 1 / (number of visible gaussians sharing the same DINO patch cell) ---
        patch_xi = (xs.float() * patch_w / w).long().clamp(0, patch_w - 1)
        patch_yi = (ys.float() * patch_h / h).long().clamp(0, patch_h - 1)
        patch_flat = patch_yi * patch_w + patch_xi  # (M,)
        patch_counts = torch.bincount(patch_flat, minlength=patch_h * patch_w)  # (patch_h*patch_w,)
        new_coverage = 1.0 / patch_counts[patch_flat].float()  # (M,) — higher when fewer gaussians share a patch

        N_total = scene._xyz.shape[0]
        if self._embedding_best_coverage is None or self._embedding_best_coverage.shape[0] != N_total:
            self._embedding_best_coverage = torch.zeros(N_total, device=DEFAULT_DEVICE, dtype=torch.float32)

        # indices into the full gaussian array for the M visible gaussians
        visible_global_idx = torch.nonzero(visible, as_tuple=False).squeeze(1)  # (M,)

        with torch.no_grad():
            best_so_far = self._embedding_best_coverage[visible_global_idx]  # (M,)
            improves = new_coverage >= best_so_far  # (M,) bool — this view is at least as good as the best

            improve_local = improves & scene._has_embedding.data[visible_global_idx]   # already has emb, and improves
            new_local = improves & ~scene._has_embedding.data[visible_global_idx]      # no emb yet, and improves

            improve_global = visible_global_idx[improve_local]
            new_global = visible_global_idx[new_local]

            # EMA blend only when the new view beats or matches the stored best; favour the new (closer) view
            scene._embedding.data[improve_global] = (
                self.ema_alpha * scene._embedding.data[improve_global]
                + (1.0 - self.ema_alpha) * sampled_embeddings[improve_local]
            )
            scene._embedding.data[new_global] = sampled_embeddings[new_local]
            scene._has_embedding.data[new_global] = True

            # update best coverage for all gaussians that improved
            self._embedding_best_coverage[visible_global_idx[improves]] = new_coverage[improves]

        return int(visible.sum().item())

    @torch.no_grad()
    def drive_run_loop(self):
        if self.embedding_encoder_job is None:
            _cp = ps.get_view_camera_parameters()
            current_cam_state = np.concatenate([_cp.get_position(), _cp.get_up_dir()])
            if self.embedding_last_camera_mat is not None and np.allclose(current_cam_state, self.embedding_last_camera_mat):
                return
            self.embedding_last_camera_mat = current_cam_state

            now = time.perf_counter()
            if now - self.embedding_loop_prevtime < self.embedding_loop_interval:
                return

            self.embedding_loop_prevtime = now

            # PART1: snapshot current view (sync)
            scene = self.render_kwargs["pc"]
            projected, pre_ndc_depth = project_gaussian_means_to_2d_pos_and_depth(scene, None)
            rgb, depth = self.render_gaussians("passthrough", composite_brush=False)
            rgb_np = rgb.detach().cpu().numpy()

            self.embedding_scene_snapshot = {
                "projected": projected,
                "pre_ndc_depth": pre_ndc_depth,
                "depth": depth,
                "image_hw": rgb_np.shape[:2],
            }

            # PART2: run DINOv3 on view render (ASYNC)
            self.embedding_encoder_job = self.embedding_encoder_thread.submit(dinov3_process, rgb_np)
            return

        if not self.embedding_encoder_job.done():
            return

        cls_token, patch_embeddings = self.embedding_encoder_job.result()

        # PART3: backproject 2D embeddings into scene (sync)
        num_recolored = self.backproject_visible_embeddings(
            self.embedding_scene_snapshot["projected"],
            self.embedding_scene_snapshot["pre_ndc_depth"],
            patch_embeddings,
            self.embedding_scene_snapshot["image_hw"],
            self.embedding_scene_snapshot["depth"],
        )
        print(f"Updated embeddings for {num_recolored} visible gaussians")

        self._embedding_pca_rgb = None
        self._embedding_pca_rgb_size = None
        self.embedding_encoder_job = None
        self.embedding_scene_snapshot = None

    @torch.no_grad()
    def segment_from_selection(self):
        mask = self.current_selection.mask
        if mask is None or not torch.any(mask):
            return
        if self.context_selection_mask is not None:
            mask = mask | self.context_selection_mask
            self.context_selection_mask = None

        scene = self.render_kwargs["pc"]
        has_embedding = scene._has_embedding.data.bool()
        sel = mask & has_embedding
        bbox_idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
        if bbox_idx.numel() < 2:
            return

        bbox_emb = torch.nn.functional.normalize(scene._embedding.data[bbox_idx], dim=1)
        ndc = project_gaussian_means_to_2d_pos(scene._xyz.detach(), None)
        bbox_ndc_pos = ndc[bbox_idx, :2]
        bbox_center = bbox_ndc_pos.mean(dim=0)
        ca = (bbox_ndc_pos - bbox_center).norm(dim=1).argmin().item()
        cb = (bbox_emb @ bbox_emb[ca]).argmin().item()
        centroids = torch.stack([bbox_emb[ca], bbox_emb[cb]])

        labels = torch.zeros(bbox_idx.numel(), dtype=torch.long, device=DEFAULT_DEVICE)
        for _ in range(10):
            labels = (bbox_emb @ centroids.T).argmax(dim=1)
            new_centroids = torch.stack([
                torch.nn.functional.normalize(
                    bbox_emb[labels == k].mean(dim=0, keepdim=True) if (labels == k).any() else centroids[k:k+1],
                    dim=1,
                ).squeeze(0)
                for k in range(2)
            ])
            if torch.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids

        cluster_centers = torch.stack([
            bbox_ndc_pos[labels == k].mean(dim=0) if (labels == k).any() else bbox_center
            for k in range(2)
        ])
        fg = (cluster_centers - bbox_center).norm(dim=1).argmin().item()
        bg = 1 - fg

        N = scene._xyz.shape[0]
        selection = torch.zeros(N, dtype=torch.bool, device=DEFAULT_DEVICE)
        selection[bbox_idx[labels == fg]] = True
        self.current_selection.set_mask(selection)
        self.selection_identifier = self.current_selection.identifier.detach().cpu().numpy()

        context_mask = torch.zeros(N, dtype=torch.bool, device=DEFAULT_DEVICE)
        context_mask[bbox_idx[labels == bg]] = True
        self.context_selection_mask = context_mask
        self._seg_bbox_idx = bbox_idx
        self.update_bbox_from_selection()

    @torch.no_grad()
    def morph_mask(self, context: bool = False):
        mask = self.context_selection_mask if context else self.current_selection.mask
        if mask is None or not torch.any(mask):
            return
        if self._seg_bbox_idx is None:
            return
        xyz = self.render_kwargs["pc"]._xyz.detach()

        bbox_idx = self._seg_bbox_idx
        bbox_mask = mask[bbox_idx]
        bbox_xyz = xyz[bbox_idx]

        test_local_idx = torch.nonzero(bbox_mask, as_tuple=False).squeeze(1)
        if test_local_idx.numel() == 0:
            return

        new_mask = mask.clone()
        chunk_size = 512

        for start in range(0, test_local_idx.shape[0], chunk_size):
            chunk_local = test_local_idx[start:start + chunk_size]
            chunk_global = bbox_idx[chunk_local]

            dists = torch.cdist(bbox_xyz[chunk_local], bbox_xyz)
            in_radius = dists <= self.morph_radius
            in_radius[torch.arange(chunk_local.shape[0], device=DEFAULT_DEVICE), chunk_local] = False

            n_total = in_radius.sum(dim=1).float()
            n_in    = (in_radius & bbox_mask.unsqueeze(0)).sum(dim=1).float()

            has_neighbors = n_total > 0
            fraction = torch.where(has_neighbors, n_in / n_total, torch.zeros_like(n_in))
            new_mask[chunk_global[has_neighbors & (fraction < 0.5)]] = False

        if context:
            self.context_selection_mask = new_mask
        else:
            self.current_selection.set_mask(new_mask)
            self.selection_identifier = self.current_selection.identifier.detach().cpu().numpy()
            self.update_bbox_from_selection()

    @torch.no_grad()
    def swap_selection_context(self):
        if self.current_selection.mask is None or self.context_selection_mask is None:
            return
        old_selection = self.current_selection.mask.clone()
        old_context = self.context_selection_mask.clone()

        self.current_selection.set_mask(old_context)
        self.selection_identifier = self.current_selection.identifier.detach().cpu().numpy()
        self.context_selection_mask = old_selection

        self.update_bbox_from_selection()

    def _invalidate_filtered_pc_cache(self):
        self._cached_context_mask = None
        self._cached_context_pc = None
        self._cached_scene_mask = None
        self._cached_scene_pc = None
        self._use_3d_heatmap = False

    def _filtered_pc_for_style(self, style):
        """Return a cached filtered GaussianModel for context/scene styles, or None if no selection.

        Cache is keyed on mask tensor identity — masks are always replaced, never mutated,
        so an `is` check is a reliable zero-cost hit test.
        """
        pc_full = self.render_kwargs["pc"]
        if style == "context":
            mask = self.context_selection_mask
            if mask is None or not torch.any(mask):
                return None
            if mask is not self._cached_context_mask:
                pc = copy.deepcopy(pc_full)
                apply_mask_to_attributes(pc, mask.bool())
                self._cached_context_mask = mask
                self._cached_context_pc = pc
            return self._cached_context_pc
        elif style == "scene":
            subject_mask = self.current_selection.mask
            N = pc_full._xyz.shape[0]
            if subject_mask is not None and torch.any(subject_mask):
                scene_mask = ~subject_mask.bool()
            else:
                scene_mask = None
            if subject_mask is not self._cached_scene_mask:
                pc = copy.deepcopy(pc_full)
                if scene_mask is not None:
                    apply_mask_to_attributes(pc, scene_mask)
                self._cached_scene_mask = subject_mask
                self._cached_scene_pc = pc
            return self._cached_scene_pc
        return None

    # 3D template-matching heatmap
    @torch.no_grad()
    def voxelize_gaussians(self, pc, xyz_min, cell_size, grid_dims):
        """Sparse voxelization of a GaussianModel into a regular grid.
        Returns:
            cell_coords:      (C, 3) int64  — occupied cell coordinates
            cell_embeddings:  (C, D) float  — L2-normalised avg embedding per cell
            gauss_vox_coords: (N, 3) int64  — voxel coordinate of every gaussian
            occupancy:        (X, Y, Z) bool — dense occupancy mask
        Returns (None, None, None, None) if no gaussian has a valid embedding.
        """
        xyz = pc._xyz.detach()                       # (N, 3)
        has_emb = pc._has_embedding.data.bool()      # (N,)
        emb = pc._embedding.data                     # (N, D)
        device = xyz.device
        X, Y, Z = grid_dims

        # Per-gaussian voxel coordinates (clamped to grid)
        coords_float = (xyz - xyz_min.unsqueeze(0)) / cell_size
        coords_int = coords_float.floor().long()
        coords_int[:, 0] = coords_int[:, 0].clamp(0, X - 1)
        coords_int[:, 1] = coords_int[:, 1].clamp(0, Y - 1)
        coords_int[:, 2] = coords_int[:, 2].clamp(0, Z - 1)

        valid_idx = has_emb.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return None, None, None, None

        valid_coords = coords_int[valid_idx]                                  # (M, 3)
        valid_emb = torch.nn.functional.normalize(emb[valid_idx].float(), dim=1)  # (M, D)

        # Linear cell index for grouping
        linear = valid_coords[:, 0] * (Y * Z) + valid_coords[:, 1] * Z + valid_coords[:, 2]
        sort_order = linear.argsort()
        sorted_linear = linear[sort_order]
        sorted_emb = valid_emb[sort_order]

        unique_cells, inverse, counts = torch.unique(sorted_linear, return_inverse=True, return_counts=True)
        C = unique_cells.shape[0]
        D = sorted_emb.shape[1]

        # Scatter-add then re-normalise
        cell_emb_sum = torch.zeros(C, D, device=device, dtype=torch.float32)
        cell_emb_sum.scatter_add_(0, inverse.unsqueeze(1).expand(-1, D), sorted_emb)
        cell_embeddings = torch.nn.functional.normalize(cell_emb_sum, dim=1)  # (C, D)

        # Recover 3D integer coordinates from linear index
        cell_x = unique_cells // (Y * Z)
        cell_y = (unique_cells % (Y * Z)) // Z
        cell_z = unique_cells % Z
        cell_coords = torch.stack([cell_x, cell_y, cell_z], dim=1)           # (C, 3)

        occupancy = torch.zeros(X, Y, Z, dtype=torch.bool, device=device)
        occupancy[cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2]] = True

        return cell_coords, cell_embeddings, coords_int, occupancy

    @torch.no_grad()
    def _embedding_cross_correlation_3d(
        self,
        scene_cell_coords, scene_cell_embs, scene_grid_dims,
        ctx_cell_coords, ctx_cell_embs, ctx_box_dims,
    ):
        """FFT-based 3D normalised cross-correlation, one embedding dimension at a time.
        scene_cell_coords: (C_s, 3) int64
        scene_cell_embs:   (C_s, D) float  — L2-normalised
        scene_grid_dims:   (X, Y, Z) tuple
        ctx_cell_coords:   (C_c, 3) int64  — relative to ctx bounding-box corner
        ctx_cell_embs:     (C_c, D) float  — L2-normalised
        ctx_box_dims:      (kX, kY, kZ) tuple
        Returns (X, Y, Z) float similarity tensor.
        """
        X, Y, Z = scene_grid_dims
        kX, kY, kZ = ctx_box_dims
        D = scene_cell_embs.shape[1]
        device = scene_cell_embs.device

        # Zero-pad to avoid circular-convolution wrap-around
        pX, pY, pZ = X + kX, Y + kY, Z + kZ

        # --- active-cell-count denominator (computed once, not per dimension) ---
        scene_occ = torch.zeros(pX, pY, pZ, device=device)
        scene_occ[
            scene_cell_coords[:, 0],
            scene_cell_coords[:, 1],
            scene_cell_coords[:, 2],
        ] = 1.0

        ctx_occ = torch.zeros(pX, pY, pZ, device=device)
        ctx_occ[ctx_cell_coords[:, 0], ctx_cell_coords[:, 1], ctx_cell_coords[:, 2]] = 1.0
        ctx_occ = torch.roll(ctx_occ, shifts=(-(kX // 2), -(kY // 2), -(kZ // 2)), dims=(0, 1, 2))

        active_cnt = torch.fft.irfftn(
            torch.fft.rfftn(ctx_occ).conj() * torch.fft.rfftn(scene_occ),
            s=(pX, pY, pZ),
        ).round().clamp_min(1.0)

        n_kernel_active = float(ctx_occ.sum().item())
        if n_kernel_active < 1.0:
            return torch.zeros(X, Y, Z, device=device)

        # Free temporary large tensors before the D-loop
        del scene_occ, ctx_occ

        # --- accumulate correlation one dimension at a time ---
        correlation = torch.zeros(pX, pY, pZ, device=device)

        for d in range(D):
            scene_slice = torch.zeros(pX, pY, pZ, device=device)
            scene_slice[
                scene_cell_coords[:, 0],
                scene_cell_coords[:, 1],
                scene_cell_coords[:, 2],
            ] = scene_cell_embs[:, d]

            ctx_slice = torch.zeros(pX, pY, pZ, device=device)
            ctx_slice[
                ctx_cell_coords[:, 0],
                ctx_cell_coords[:, 1],
                ctx_cell_coords[:, 2],
            ] = ctx_cell_embs[:, d]
            ctx_slice = torch.roll(ctx_slice, shifts=(-(kX // 2), -(kY // 2), -(kZ // 2)), dims=(0, 1, 2))

            correlation += torch.fft.irfftn(
                torch.fft.rfftn(ctx_slice).conj() * torch.fft.rfftn(scene_slice),
                s=(pX, pY, pZ),
            )

        similarity = correlation / (n_kernel_active * active_cnt).sqrt()
        return similarity[:X, :Y, :Z].contiguous()

    @torch.no_grad()
    def compute_3d_heatmap(self):
        """One-shot 3D template matching: voxelise scene and context, run FFT NCC,
        assign per-gaussian scores, and switch the renderer to the cached result."""
        pc_context = self._filtered_pc_for_style("context")
        pc_scene = self._filtered_pc_for_style("scene")
        if pc_context is None or pc_scene is None:
            print("3D Heatmap: need both context and scene selections.")
            return

        scene_full = self.render_kwargs["pc"]
        xyz_min = scene_full._xyz.detach().min(dim=0).values
        xyz_max = scene_full._xyz.detach().max(dim=0).values
        extent = xyz_max - xyz_min
        cell_size = extent.max().item() / max(self.voxel_resolution, 1)
        if cell_size <= 0:
            return

        grid_dims = tuple(
            max(int(math.ceil(extent[i].item() / cell_size)), 1) for i in range(3)
        )
        X, Y, Z = grid_dims
        print(f"3D Heatmap: grid {X}×{Y}×{Z}, cell size {cell_size:.4f}")

        scene_cell_coords, scene_cell_embs, _, _ = self.voxelize_gaussians(
            pc_scene, xyz_min, cell_size, grid_dims
        )
        if scene_cell_coords is None:
            print("3D Heatmap: scene has no embeddings.")
            return

        ctx_cell_coords_global, ctx_cell_embs, _, _ = self.voxelize_gaussians(
            pc_context, xyz_min, cell_size, grid_dims
        )
        if ctx_cell_coords_global is None:
            print("3D Heatmap: context has no embeddings.")
            return

        # Express context coords relative to its own bounding box corner
        ctx_min = ctx_cell_coords_global.min(dim=0).values
        ctx_max = ctx_cell_coords_global.max(dim=0).values
        ctx_cell_coords = ctx_cell_coords_global - ctx_min
        kX = int((ctx_max[0] - ctx_min[0]).item()) + 1
        kY = int((ctx_max[1] - ctx_min[1]).item()) + 1
        kZ = int((ctx_max[2] - ctx_min[2]).item()) + 1

        if kX > X // 2 or kY > Y // 2 or kZ > Z // 2:
            print(
                f"3D Heatmap: warning — context kernel ({kX}×{kY}×{kZ}) "
                f"exceeds 50% of scene grid ({X}×{Y}×{Z}) on at least one axis"
            )

        similarity = self._embedding_cross_correlation_3d(
            scene_cell_coords, scene_cell_embs, grid_dims,
            ctx_cell_coords, ctx_cell_embs, (kX, kY, kZ),
        )  # (X, Y, Z)

        # Assign score to every gaussian in the full scene via voxel lookup
        xyz_full = scene_full._xyz.detach()
        coords_float = (xyz_full - xyz_min.unsqueeze(0)) / cell_size
        coords_int = coords_float.floor().long()
        coords_int[:, 0] = coords_int[:, 0].clamp(0, X - 1)
        coords_int[:, 1] = coords_int[:, 1].clamp(0, Y - 1)
        coords_int[:, 2] = coords_int[:, 2].clamp(0, Z - 1)

        gauss_scores = similarity[coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]]

        # Percentile-rank transform + gamma (same as 2D path)
        flat = gauss_scores.flatten()
        _, sort_idx = flat.sort()
        ranks = flat.new_empty(flat.numel())
        ranks[sort_idx] = torch.linspace(0.0, 1.0, flat.numel(), device=flat.device, dtype=flat.dtype)
        gauss_scores = ranks.pow(self.heatmap_gamma)

        self._cached_heatmap_per_gaussian = gauss_scores
        self._use_3d_heatmap = True
        print("3D Heatmap: done.")

    # UI / interaction
    def draw_render_panel(self):
        if psim.TreeNode("Render"):
            psim.PushItemWidth(100)
            _, self.ui_state.render_mode_idx = psim.Combo(
                "Channel", self.ui_state.render_mode_idx, self.ui_constants.SUPPORTED_RENDER_MODES
            )
            if self.ui_constants.SUPPORTED_RENDER_MODES[self.ui_state.render_mode_idx] == "heatmap":
                _, self.heatmap_gamma = psim.SliderFloat(
                    "Heatmap gamma", self.heatmap_gamma, v_min=1.0, v_max=10.0
                )
                _, self.voxel_resolution = psim.SliderInt(
                    "Voxel resolution", self.voxel_resolution, v_min=32, v_max=256
                )
                if psim.Button("Compute"):
                    self.compute_3d_heatmap()
                psim.SameLine()
                if psim.Button("Snapshot"):
                    self.save_heatmap_snapshot()

            if psim.Button("Undo"):
                if self.prev_scene is not None:
                    self.render_kwargs["pc"] = self.prev_scene
                    self._invalidate_filtered_pc_cache()

            _, self.embedding_loop_enabled = psim.Checkbox(
                "Embedding Loop", self.embedding_loop_enabled
            )

            if psim.Button("Segment"):
                self.segment_from_selection()

            overlays_changed, self.ui_state.show_selection_overlays = psim.Checkbox(
                "Show Selection Overlays", self.ui_state.show_selection_overlays
            )

            _, self.morph_radius = psim.SliderFloat(
                "Morph Radius", self.morph_radius, v_min=0.001, v_max=0.25
            )
            if psim.Button("Erode Subject"):
                self.morph_mask()
            psim.SameLine()
            if psim.Button("Erode Context"):
                self.morph_mask(context=True)
            psim.SameLine()
            if psim.Button("Swap FG/BG"):
                self.swap_selection_context()

            if psim.Button("Clear Selection"):
                self.clear_selection()

            psim.PopItemWidth()
            psim.TreePop()

    def draw_scene_settings_panel(self):
        if psim.TreeNode("Scene Settings"):
            if psim.Button("Reset Scene"):
                gaussians = GaussianModel(3)
                gaussians.load_ply(self.ui_state.scene_path)
                self.render_kwargs["pc"] = gaussians
                self._invalidate_filtered_pc_cache()
            if psim.Button("Save Scene"):
                self.render_kwargs["pc"].save_ply(
                    os.path.join("asset_data/current_scene.ply")
                )
            psim.TreePop()

    def draw_brush_setup_panel(self):
        disp_change = False
        if psim.TreeNode("Brush Setup"):
            _, self.ui_state.enable_key_bindings = psim.Checkbox(
                "Enable Key Bindings", self.ui_state.enable_key_bindings
            )
            _, self.enable_surface_orientation_estimation = psim.Checkbox(
                "Estimate Surface Orientation", self.enable_surface_orientation_estimation
            )

            if psim.Button("Setup Brush"):
                if self.current_selection.mask is not None:
                    self.clip_points(self.render_kwargs["pc"])
                self.ui_state.display_mode_idx = 1
                disp_change = True
                self.clip_box_coords = [[0, 0, 0], [0, 0, 0]]
                self.render_clip_box()
            if self.ui_state.display_mode_idx == 1:
                if psim.Button("Create Brush"):
                    self.exit_brush_edit_modes()
                    self.ui_state.display_mode_idx = 0
                    ps.remove_point_cloud("brush axis", error_if_absent=False)
                    ps.remove_point_cloud("brush_stroke_center", error_if_absent=False)
                    disp_change = True
            
            input_text, self.ui_state.save_path = psim.InputText("Enter Save Path", self.ui_state.save_path)
            if input_text:
                self.ui_state.enable_key_bindings = False
            if psim.Button("Save Brush"):
                self.save_brush(self.ui_state.save_path)
                if not self.ui_state.enable_key_bindings:
                    self.ui_state.enable_key_bindings = True
            
            input_text, self.ui_state.load_path = psim.InputText("Enter Load Path", self.ui_state.load_path)
            if input_text:
                self.ui_state.enable_key_bindings = False
            if psim.Button("Load Brush"):
                self.load_brush(self.ui_state.load_path)
            if psim.Button("Clear Brush"):
                self.exit_brush_edit_modes()
                self.brush_manager.clear_brush()
                self.brush_stroke_gs = None
                self.ui_state.display_mode_idx = 0

            psim.TreePop()
        return disp_change

    def draw_brush_parameters_panel(self):
        if psim.TreeNode("Brush Parameters"):
            _, self.blend_percentage = psim.SliderFloat("Spacing", self.blend_percentage, v_min=-1.0, v_max=1.0)
            _, self.z_offset = psim.SliderFloat("Z Offset", self.z_offset, v_min=-2.5, v_max=2.5)
            psim.TreePop()

    def toggle_brush_edit_mode(self):
        self.brush_edit_mode = not self.brush_edit_mode
        if self.brush_edit_mode:
            self.gizmo._gizmo_size = (
                2 * torch.norm(
                    self.brush_manager.brush_gs._xyz.max(dim=0)[0]
                    - self.brush_manager.brush_gs._xyz.min(dim=0)[0]
                ).detach().cpu().numpy()
            )
            self.gizmo.show(
                self.brush_manager.brush_gs._xyz.detach(),
                self.brush_manager.brush_gs._rotation.detach(),
                self.brush_manager.brush_gs._scaling.detach(),
            )
            rotation_full = np.eye(4)
            rotation_full[:3, :3] = self.brush_manager.stroke_pca_directions.detach().cpu().numpy()
            curr_transform = self.gizmo.compensated_transform
            curr_transform = np.matmul(curr_transform, rotation_full)
            self.gizmo.point_cloud.set_transform(curr_transform)
            self.gizmo.compensated_transform = curr_transform
        else:
            self.transform_brush(self.gizmo)
            gizmo_rotation = self.gizmo.get_transform()[:3, :3]
            rot_tensor = torch.tensor(gizmo_rotation, device=DEFAULT_DEVICE, dtype=torch.float32)
            self.brush_manager.stroke_pca_directions = torch.matmul(rot_tensor, self.brush_manager.stroke_pca_directions)
            self.brush_manager.original_stroke_pca_directions = torch.matmul(rot_tensor, self.brush_manager.original_stroke_pca_directions)
            self.brush_manager.commit_transform()
            self.gizmo.hide()

    def exit_brush_edit_modes(self):
        if self.brush_edit_mode:
            self.toggle_brush_edit_mode()
        self.gizmo.hide()

    def handle_key_bindings(self):
        io = psim.GetIO()

        if psim.IsKeyDown(psim.ImGuiKey_7):  # '7' key
            self.brush_manager.clear_brush()
            self.brush_stroke_gs = None
            self.ui_state.display_mode_idx = 0

        if self.ui_state.enable_key_bindings:
            if io.KeyShift:
                if self.brush_manager.brush_gs is not None:
                    curr_brush_mask = get_minimal_surface_mask_3d(
                        psim.GetMousePos(), 15, self.render_kwargs["pc"],
                        self.render_gaussians("depth", composite_brush=False)
                    )
                    _, new_normal, _ = auto_orient_transform(
                        self.render_kwargs["pc"], curr_brush_mask,
                        self.brush_manager.stroke_pca_directions[:, 2].detach().cpu().numpy()
                    )
                    self.ground_plane_normal, self.ground_plane_point = get_plane_normal_and_point(self.render_kwargs["pc"], curr_brush_mask)
                    self.ground_plane_normal = torch.tensor(new_normal, device=DEFAULT_DEVICE, dtype=torch.float32)
                    self.ground_plane = True

            if psim.IsKeyPressed(psim.ImGuiKey_R):
                self.toggle_brush_edit_mode()

    def ui_callback(self):
        self.model = self.render_kwargs["pc"]
        payload = CallbackPayload(
            model=self.model,
            camera=ps.get_view_camera_parameters(),
            last_selection=self.current_selection,
            selection_preview=self.current_selection,
            segments=self.segments,
            gui_mode=self.ui_state.gui_mode,
            transformation_gizmo=self.gizmo,
        )
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)

        self.draw_render_panel()
        self.draw_scene_settings_panel()
        disp_change = self.draw_brush_setup_panel()

        if disp_change and self.ui_state.display_mode_idx == 1:
            ps.set_ground_plane_mode("tile")
            ps.set_ground_plane_height_factor(0.1)
            if self.brush_manager.original_brush_gs is not None:
                self.brush_manager.center_original_brush()
                self.brush_manager.reset_transform()

        if self.ui_state.display_mode_idx == 1:
            if psim.Button("Edit Brush"):
                self.toggle_brush_edit_mode()
        if disp_change and self.ui_state.display_mode_idx == 0:
            ps.set_ground_plane_mode("none")

        self.draw_brush_parameters_panel()
        self.handle_key_bindings()

        if self.brush_edit_mode:
            self.transform_brush(self.gizmo)

        if (
            self.brush_manager.brush_gs is not None
            and not self.brush_edit_mode
            and self.ui_state.display_mode_idx == 0
        ):
            self.update_transform_from_mouse_pos()
            self.clone_composite()

        if self.brush_manager.brush_gs is not None:
            self.visualize_pca() 

        with torch.no_grad():
            self.drag_handler.handle_callback(payload)
            if (
                payload.last_selection.identifier.detach().cpu().numpy()
                != self.selection_identifier
                and not self.drag_handler.is_drag_in_progress()
            ):
                self.current_selection = payload.last_selection
                self.selection_identifier = (
                    self.current_selection.identifier.detach().cpu().numpy()
                )
                self.recompute_context_selection()
                self.update_bbox_from_selection()

        self.draw_clip_box_overlay()

        if self.embedding_loop_enabled:
            self.drive_run_loop()
