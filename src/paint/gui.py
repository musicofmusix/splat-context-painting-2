# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
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

DEFAULT_DEVICE = torch.device("cuda")
MAX_DEPTH = 1000.0

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
    selected_brush: int = 0
    supported_brushes: list = None
    supported_brush_paths: list = None
    brush_path: str = None
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
    window_size: Tuple = (1920, 1080)

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

def find_ply_files_in_dir_relative(directory):
    cwd = os.getcwd()
    ply_paths = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".ply"):
            full_path = os.path.join(directory, filename)
            relative_path = os.path.relpath(full_path, start=cwd)
            ply_paths.append(relative_path)
    return ply_paths

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
        brush_path=None
    ):
        self.ui_constants = UIConstants()
        self.ui_state = UIState()
        self.render_state = RenderState()
        self.brush_manager = BrushManager(DEFAULT_DEVICE)

        self.object_name = "gsplat object"
        self.prev_scene = None

        self.ui_state.brush_path = brush_path
        self.sphere_center = torch.tensor(
            [0.0, 0.0, 0.0], device=DEFAULT_DEVICE, dtype=torch.float32
        )
        self.sphere_radius = 1
        if brush_path is not None and not brush_path.endswith(".ply"):
            self.ui_state.supported_brush_paths = find_ply_files_in_dir_relative(brush_path)
            self.ui_state.supported_brushes = [
                os.path.basename(brush) for brush in self.ui_state.supported_brush_paths
            ]
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

        # embedding update exponential moving average param
        # higher = trust the past, lower = trust the present
        self.ema_alpha = 0.75
        self.segmentation_mode_enabled = False

        # segmentation BFS parameters
        self.grow_radius = 0.05
        self.similarity_threshold = 0.9
        self.heatmap_threshold = 0.9
        self.context_grow_radius = 0.04
        self.context_selection_mask = None
        self.context_prototype = None
        self._embedding_cache_token = 0
        self._context_prototype_token = 0
        self._embedding_pca_rgb_size = None
        self._heatmap_values = None
        self._snap_heatmap_values = None
        self._snap_heatmap_cache_size = None
        self._heatmap_colors = None
        self._heatmap_cache_size = None
        self._heatmap_color_threshold = None
        self._snap_candidates_mask = None
        self._snap_candidates_idx = None
        self._snap_candidates_screen_xy = None
        self._snap_pos_3d = None
        self._last_mouse_pos = None
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
        if brush_path is not None and brush_path.endswith(".ply"):
            self.load_brush(brush_path)
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

        if self.brush_manager.brush_gs is not None or self.context_prototype is not None:
            query_depth, query_cam_dist = self.render_scene_query_buffers()
        else:
            query_depth, query_cam_dist = self.render_state.curr_depth, self.render_state.dist_cam
        cam = polyscope_to_gsplat_camera(
            None, downsample_factor=self.render_state.canvas_res_downsample
        )
        display_pc = self.resolve_render_pc("render", composite_brush=True)
        self.refresh_snap_candidates()
        style_override_color = self.get_render_style_color(display_pc, style)
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

    def refresh_snap_candidates(self):
        """Compute snap candidates from scene pc every frame, independent of render style."""
        if self.context_prototype is None:
            self._snap_candidates_mask = None
            self._snap_candidates_idx = None
            self._snap_candidates_screen_xy = None
            return
        scene_pc = self.render_kwargs["pc"]
        scene_valid = scene_pc._has_embedding.bool()
        if self._snap_heatmap_values is None or self._snap_heatmap_cache_size != scene_pc._embedding.shape[0]:
            self._snap_heatmap_values, _ = self.get_normalized_heatmap_similarity(scene_pc)
            self._snap_heatmap_cache_size = scene_pc._embedding.shape[0]
        self._snap_candidates_mask = self.compute_visible_valid_snap_mask(
            scene_pc, self._snap_heatmap_values, scene_valid
        )

    def compute_visible_valid_snap_mask(self, pc, heatmap_values, valid):
        """Returns [N] bool mask: above-threshold Gaussians that are depth-visible from the current camera."""
        depth_buffer = self.render_state.curr_depth
        if depth_buffer is None:
            return None

        above_threshold = valid & (heatmap_values >= self.heatmap_threshold)
        if above_threshold.sum() == 0:
            return above_threshold

        W, H = ps.get_window_size()
        W = W // self.render_state.canvas_res_downsample
        H = H // self.render_state.canvas_res_downsample

        # Project only the above-threshold subset: one matmul gives both screen xy and depth
        above_xyz = pc._xyz.detach()[above_threshold]
        K = above_xyz.shape[0]
        gsplat_cam = polyscope_to_gsplat_camera(None)
        mvp = gsplat_cam.full_proj_transform.T[None].expand(K, 4, 4)
        proj = (mvp @ torch.cat([above_xyz, above_xyz.new_ones(K, 1)], dim=1).unsqueeze(-1)).squeeze(-1)
        pre_ndc_depth = proj[:, 2]
        screen_xy = proj[:, :2] / proj[:, 3:4]
        x_coords = ((screen_xy[:, 0] + 1) / 2.0 * W).round().long()
        y_coords = ((screen_xy[:, 1] + 1) / 2.0 * H).round().long()

        z_h, z_w = depth_buffer.shape[0], depth_buffer.shape[1]
        within_bounds = (
            (x_coords >= 0) & (x_coords < z_w) &
            (y_coords >= 0) & (y_coords < z_h) &
            (pre_ndc_depth > 0.0)
        )
        z_match = within_bounds & (
            torch.abs(pre_ndc_depth - depth_buffer[y_coords.clamp(0, z_h - 1), x_coords.clamp(0, z_w - 1)].squeeze()) < 1e-2
        )

        snap_mask = torch.zeros(pc._xyz.shape[0], dtype=torch.bool, device=pc._xyz.device)
        snap_mask[above_threshold] = z_match

        # Cache SS coords and indices for use in snapping logic
        snap_idx = torch.nonzero(snap_mask, as_tuple=False).squeeze(1)
        self._snap_candidates_idx = snap_idx
        self._snap_candidates_screen_xy = (
            torch.stack([x_coords[z_match].float(), y_coords[z_match].float()], dim=1)
            if snap_idx.numel() > 0
            else torch.zeros((0, 2), device=pc._xyz.device)
        )

        return snap_mask

    def find_nearest_snap_candidate_ss(self, candidate_pixel, search_radius):
        """Returns the scene Gaussian index of the nearest snap candidate within search_radius pixels, or None."""
        if self._snap_candidates_screen_xy is None or self._snap_candidates_screen_xy.shape[0] == 0:
            return None
        cand_xy = self._snap_candidates_screen_xy.to(device=candidate_pixel.device, dtype=candidate_pixel.dtype)
        dists = torch.norm(cand_xy - candidate_pixel.unsqueeze(0), dim=1)
        min_dist, min_idx = dists.min(dim=0)
        if min_dist.item() > search_radius:
            return None
        return self._snap_candidates_idx[min_idx].item()

    def update_snap_pos(self):
        """Continuity-first snapping: tracks the nearest visible valid Gaussian in screen space."""
        io = psim.GetIO()
        mouse_pos = psim.GetMousePos()
        W, H = ps.get_window_size()

        snapping_active = (
            psim.IsKeyDown(psim.ImGuiKey_S)
            and self.context_prototype is not None
            and self._snap_candidates_idx is not None
            and self._snap_candidates_idx.numel() > 0
        )

        if not snapping_active:
            if self._snap_pos_3d is not None:
                self._snap_pos_3d = None
                ps.register_point_cloud("snap_target", np.zeros((0, 3)))
            self._last_mouse_pos = None
            return

        mouse_delta = (0.0, 0.0)
        if self._last_mouse_pos is not None:
            mouse_delta = (
                mouse_pos[0] - self._last_mouse_pos[0],
                mouse_pos[1] - self._last_mouse_pos[1],
            )
        self._last_mouse_pos = mouse_pos

        search_radius = 80.0  # pixels

        # Build candidate pixel: project previous snap pos + mouse delta, or raw mouse if no prior snap
        if self._snap_pos_3d is not None:
            screen_xyz = project_gaussian_means_to_2d_pos(self._snap_pos_3d.unsqueeze(0), None)
            snap_ss_x = float((screen_xyz[0, 0] + 1) / 2.0 * W) + mouse_delta[0]
            snap_ss_y = float((screen_xyz[0, 1] + 1) / 2.0 * H) + mouse_delta[1]
            candidate_pixel = torch.tensor(
                [max(0.0, min(W - 1, snap_ss_x)), max(0.0, min(H - 1, snap_ss_y))],
                device=DEFAULT_DEVICE, dtype=torch.float32,
            )
        else:
            candidate_pixel = torch.tensor(
                [float(mouse_pos[0]), float(mouse_pos[1])],
                device=DEFAULT_DEVICE, dtype=torch.float32,
            )

        winner_idx = self.find_nearest_snap_candidate_ss(candidate_pixel, search_radius)

        # Continuity fallback: if projected candidate failed, try raw mouse position
        if winner_idx is None and self._snap_pos_3d is not None:
            mouse_pixel = torch.tensor(
                [float(mouse_pos[0]), float(mouse_pos[1])],
                device=DEFAULT_DEVICE, dtype=torch.float32,
            )
            winner_idx = self.find_nearest_snap_candidate_ss(mouse_pixel, search_radius)

        if winner_idx is not None:
            self._snap_pos_3d = self.render_kwargs["pc"]._xyz[winner_idx].detach().clone()
            dot = ps.register_point_cloud("snap_target", self._snap_pos_3d.cpu().numpy()[None], radius=0.005)
            dot.set_color((0.0, 1.0, 0.4))
        else:
            self._snap_pos_3d = None
            ps.register_point_cloud("snap_target", np.zeros((0, 3)))

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

        if style == "heatmap" and self.context_prototype is not None:
            heatmap_values = self._heatmap_values
            if heatmap_values is None or self._heatmap_cache_size != pc._embedding.shape[0]:
                heatmap_values, _ = self.get_normalized_heatmap_similarity(pc)
                self._heatmap_values = heatmap_values
                self._heatmap_cache_size = pc._embedding.shape[0]

            heatmap_colors = torch.zeros(
                (pc._embedding.shape[0], 3),
                device=pc._embedding.device,
                dtype=torch.float32,
            )
            if self._snap_candidates_mask is not None:
                n_scene = self._snap_candidates_mask.shape[0]
                heatmap_colors[:n_scene][self._snap_candidates_mask] = 1.0
            else:
                above_threshold = valid & (heatmap_values >= self.heatmap_threshold)
                heatmap_colors[above_threshold] = 1.0
            override_color[valid] = heatmap_colors[valid].to(
                device=override_color.device, dtype=override_color.dtype
            )
            return override_color

        return None

    # Selection / context
    def render_clip_box(self):
        bbox_min, bbox_max = self.clip_box_coords
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
        ps.register_curve_network("clip bbox", nodes, edges)
        return

    def update_bbox_from_selection(self):
        xyz = self.render_kwargs["pc"].get_xyz
        mask = self.current_selection.mask
        if mask is not None:
            selected_xyz = xyz[mask]
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
                ps.register_point_cloud(
                    "selected_xyz", selected_xyz.detach().cpu().numpy()
                )
                self.render_clip_box()
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
                    self.render_clip_box()
        self.update_selection_visualizations()
        return

    def invalidate_heatmap_caches(self):
        self._heatmap_values = None
        self._snap_heatmap_values = None
        self._snap_heatmap_cache_size = None
        self._heatmap_colors = None
        self._heatmap_cache_size = None

    def clear_context_selection(self):
        self.context_selection_mask = None
        self.context_prototype = None
        self._context_prototype_token += 1
        self.invalidate_heatmap_caches()
        ps.register_point_cloud("context_xyz", np.zeros((0, 3)))
        context_pc = ps.get_point_cloud("context_xyz")
        context_pc.set_color((1.0, 1.0, 0.0))

    def recompute_context_selection(self):
        scene = self.render_kwargs["pc"]
        xyz = scene._xyz.detach()
        selection = self.current_selection.mask

        if selection is None or not torch.any(selection):
            self.context_selection_mask = None
            self.context_prototype = None
            self.invalidate_heatmap_caches()
            return

        has_embedding_full = scene._has_embedding.data.bool()
        outside_selection = ~selection
        context_selection = torch.zeros_like(selection)
        selected_idx = torch.nonzero(selection, as_tuple=False).squeeze(1)

        if selected_idx.numel() > 0:
            selected_xyz = xyz[selected_idx]
            all_idx = torch.arange(xyz.shape[0], device=DEFAULT_DEVICE)
            outside_idx = all_idx[outside_selection & has_embedding_full]
            outside_xyz = xyz[outside_idx]

            if outside_idx.numel() > 0:
                chunk_size = 128
                near_boundary = torch.zeros(outside_idx.shape[0], dtype=torch.bool, device=DEFAULT_DEVICE)
                chunk_start = 0
                while chunk_start < selected_idx.shape[0]:
                    selected_chunk = selected_xyz[chunk_start : chunk_start + chunk_size]
                    chunk_start += chunk_size
                    near_boundary |= (
                        torch.cdist(selected_chunk, outside_xyz) <= self.context_grow_radius
                    ).any(dim=0)

                context_selection[outside_idx[near_boundary]] = True

        self.context_selection_mask = context_selection
        if torch.any(context_selection):
            context_embeddings = scene._embedding.data[context_selection]
            context_embeddings = torch.nn.functional.normalize(context_embeddings, dim=1)
            prototype = context_embeddings.mean(dim=0, keepdim=True)
            self.context_prototype = torch.nn.functional.normalize(prototype, dim=1).squeeze(0)
        else:
            self.context_prototype = None
        self._context_prototype_token += 1
        self.invalidate_heatmap_caches()

    def clear_selection(self):
        self.current_selection.reset()
        self.selection_identifier = self.current_selection.identifier.detach().cpu().numpy()
        self.clear_context_selection()
        ps.register_point_cloud("selected_xyz", np.zeros((0, 3)))
        ps.remove_curve_network("clip bbox", error_if_absent=False)

    def update_selection_visualizations(self):
        xyz = self.render_kwargs["pc"].get_xyz
        show_overlays = self.ui_state.show_selection_overlays

        if (
            show_overlays
            and self.current_selection.mask is not None
            and torch.any(self.current_selection.mask)
        ):
            selected_xyz = xyz[self.current_selection.mask].detach().cpu().numpy()
        else:
            selected_xyz = np.zeros((0, 3))

        ps.register_point_cloud("selected_xyz", selected_xyz)

        show_context_selection = show_overlays and self.ui_state.display_mode_idx == 0

        if (
            show_context_selection
            and self.context_selection_mask is not None
            and torch.any(self.context_selection_mask)
        ):
            context_xyz = xyz[self.context_selection_mask].detach().cpu().numpy()
        else:
            context_xyz = np.zeros((0, 3))

        ps.register_point_cloud("context_xyz", context_xyz)
        context_pc = ps.get_point_cloud("context_xyz")
        context_pc.set_color((1.0, 1.0, 0.0))

    def get_normalized_heatmap_similarity(self, pc):
        valid = pc._has_embedding.bool()
        similarity = torch.zeros(
            (pc._embedding.shape[0],), device=pc._embedding.device, dtype=torch.float32
        )

        if self.context_prototype is None or valid.sum().item() == 0:
            return similarity, valid

        embedding_norm = torch.nn.functional.normalize(pc._embedding[valid], dim=1)
        prototype = self.context_prototype.to(
            device=embedding_norm.device, dtype=embedding_norm.dtype
        )
        valid_similarity = torch.clamp(embedding_norm @ prototype, 0.0, 1.0)
        similarity[valid] = valid_similarity
        return similarity, valid

    # Brush setup / painting
    def save_brush(self, path_to_ply=None):
        if path_to_ply is None or not str(path_to_ply).strip():
            path_to_ply = "data/brushes/brush.ply"

        save_path = os.path.join(os.getcwd(), path_to_ply)
        selection = self.current_selection.mask

        if selection is not None and torch.any(selection):
            selected_gs = copy.deepcopy(self.render_kwargs["pc"])
            apply_mask_to_attributes(selected_gs, selection.detach())
            meta = {}
            if self.context_prototype is not None:
                meta["prototype_embedding"] = self.context_prototype.detach().cpu()
            export_brush(save_path, selected_gs, meta=meta if len(meta) > 0 else None)
            return

        self.brush_manager.save_brush(path_to_ply)

    def load_brush(self, path_to_ply):
        self.brush_stroke_gs = None
        self.ui_state.display_mode_idx = 0
        self.brush_manager.load_brush(path_to_ply)
        _, brush_meta, _ = import_brush(os.path.join(os.getcwd(), path_to_ply))
        prototype_embedding = None if brush_meta is None else brush_meta.get("prototype_embedding")
        if prototype_embedding is not None:
            if not torch.is_tensor(prototype_embedding):
                prototype_embedding = torch.tensor(prototype_embedding, dtype=torch.float32)
            self.context_prototype = prototype_embedding.detach().to(
                device=DEFAULT_DEVICE, dtype=torch.float32
            )
        else:
            self.context_prototype = None
        self._context_prototype_token += 1
        self.invalidate_heatmap_caches()
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
            self.segments: Dict[str, Segment] = self.init_segments()
            self.brush_stroke_gs = self.brush_manager.brush_gs
        return

    def update_transform_from_mouse_pos(self):
        mouse_pos = psim.GetMousePos()
        window_w, window_h = ps.get_window_size()

        if self._snap_pos_3d is not None:
            screen_xyz = project_gaussian_means_to_2d_pos(self._snap_pos_3d.unsqueeze(0), None)
            snap_ss_x = float((screen_xyz[0, 0] + 1) / 2.0 * window_w)
            snap_ss_y = float((screen_xyz[0, 1] + 1) / 2.0 * window_h)
            mouse_pos = (
                max(0.0, min(float(window_w - 1), snap_ss_x)),
                max(0.0, min(float(window_h - 1), snap_ss_y)),
            )
        if self.render_state.curr_depth is not None:
            depth_render = self.render_state.curr_depth
        else:
            depth_render = self.render_gaussians("depth", composite_brush=False)
        if self.render_state.dist_cam is not None:
            cam_dist_render = self.render_state.dist_cam
        else:
            cam_dist_render = self.render_gaussians("dist_to_cam")
        if (
            mouse_pos[0] < 0
            or mouse_pos[0] >= window_w
            or mouse_pos[1] < 0
            or mouse_pos[1] >= window_h
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
        patch_h, patch_w, _ = patch_embeddings.shape
        patch_y = torch.clamp((ys.float() * patch_h / max(h, 1)).floor().long(), 0, patch_h - 1)
        patch_x = torch.clamp((xs.float() * patch_w / max(w, 1)).floor().long(), 0, patch_w - 1)

        # size MxD embeddings for visual gaussians
        sampled_embeddings = patch_embeddings[patch_y, patch_x]

        with torch.no_grad():
            # for entries that are both visible AND already has embeddings, use EMA to update
            # size K list of gaussian indices that are visible and has/no embeddings
            has_embedding_visible_idx = torch.nonzero(scene._has_embedding.data[visible], as_tuple=False).squeeze(1)
            no_embedding_visible_idx = torch.nonzero(~scene._has_embedding.data[visible], as_tuple=False).squeeze(1)
            # size N list of gaussian indices that are visible and has/no embeddings
            has_embedding_visible = visible & scene._has_embedding.data
            no_embedding_visible = visible & ~scene._has_embedding.data

            # remember, sampled_embeddings is size MxD (visible gaussians only), while _embedding is size N (all gaussians)
            scene._embedding.data[has_embedding_visible] = (
                        self.ema_alpha * scene._embedding.data[has_embedding_visible]
                        + (1.0 - self.ema_alpha) * sampled_embeddings[has_embedding_visible_idx]
                )

            # direct assignment for gaussians that do not yet have embeddings yet
            scene._embedding.data[no_embedding_visible] = sampled_embeddings[no_embedding_visible_idx]
            scene._has_embedding.data[no_embedding_visible] = True

        return int(visible.sum().item())

    @torch.no_grad()
    def drive_run_loop(self):
        if self.embedding_encoder_job is None:
            now = time.perf_counter()
            if now - self.embedding_loop_prevtime < self.embedding_loop_interval:
                return

            self.embedding_loop_prevtime = now

            # PART1: snapshot current view (sync)
            scene = self.render_kwargs["pc"]
            projected = project_gaussian_means_to_2d_pos(scene._xyz.detach(), None)
            pre_ndc_depth = project_gaussian_means_to_2d_pre_ndc_depth(scene, None)
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
        self._embedding_pca_bases = None
        self._embedding_cache_token += 1
        self.invalidate_heatmap_caches()
        self.embedding_encoder_job = None
        self.embedding_scene_snapshot = None

    @torch.no_grad()
    # all masks operate in the full N gaussian array space, for readability
    # although, one might want to operate in M has-embedding space for speed later
    def handle_segmentation_click(self):
        if not self.segmentation_mode_enabled:
            return

        io = psim.GetIO()
        if io.WantCaptureMouse or not psim.IsMouseClicked(psim.ImGuiMouseButton_Left):
            return

        if self.render_state.curr_depth is not None:
            depth_render = self.render_state.curr_depth
        else:
            depth_render = self.render_gaussians("depth", composite_brush=False)

        # select all surface gaussians within a small 2D radius around the mouse cursor
        mask = get_minimal_surface_mask_3d(
            psim.GetMousePos(), 5, self.render_kwargs["pc"], depth_render
        )

        hit_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if hit_idx.numel() == 0:
            return

        scene = self.render_kwargs["pc"]
        if not torch.any(scene._has_embedding.data[hit_idx]):
            return

        # prepare masks
        has_embedding = scene._has_embedding.data.bool()
        has_embedding_idx = torch.nonzero(has_embedding, as_tuple=False).squeeze(1)
        hit_has_embedding_idx = hit_idx[has_embedding[hit_idx]]
        xyz = scene._xyz.detach()

        # prepare the single seed vector to grow our selection from
        median_embedding = torch.median(scene._embedding.data[hit_has_embedding_idx], dim=0).values
        seed = torch.nn.functional.normalize(median_embedding.unsqueeze(0), dim=1).squeeze(0)

        # normalise all embeddings
        embedding_norm = torch.nn.functional.normalize(scene._embedding.data[has_embedding], dim=1)

        # all gaussians with an embedding that exceed the similarity threshold
        candidates_idx = has_embedding_idx[(embedding_norm @ seed) >= self.similarity_threshold]
        candidates_xyz = xyz[candidates_idx]

        # build BFS frontier; start with our initial hit gaussians
        frontier_idx = hit_has_embedding_idx
        # smaller = faster but less accurate
        frontier_chunk_size = 128

        # selected gaussians
        selection = torch.zeros(xyz.shape[0], dtype=torch.bool, device=DEFAULT_DEVICE)
        selection[frontier_idx] = True

        # run wave-based BFS
        # each wave is split into chunks, to lessen the impact of O(N^2) cdist
        while frontier_idx.numel() > 0:
            near_candidate = torch.zeros(candidates_idx.shape[0], dtype=torch.bool, device=DEFAULT_DEVICE)

            chunk_start = 0
            while chunk_start < frontier_idx.shape[0]:
                frontier_chunk = frontier_idx[chunk_start : chunk_start + frontier_chunk_size]
                chunk_start += frontier_chunk_size

                frontier_xyz = xyz[frontier_chunk]

                # check which candidates lie within the radius of ANY frontier point in this chunk
                chunk_near_candidate = torch.cdist(frontier_xyz, candidates_xyz) <= self.grow_radius
                near_candidate |= chunk_near_candidate.any(dim=0)

            # accept all candidates reached in this wave that are not selected yet
            accepted_candidates_idx = candidates_idx[near_candidate & ~selection[candidates_idx]]
            selection[accepted_candidates_idx] = True
            frontier_idx = accepted_candidates_idx

        if io.KeyShift and self.current_selection.mask is not None:
            selection = selection | self.current_selection.mask

        self.current_selection.set_mask(selection)
        self.selection_identifier = self.current_selection.identifier.detach().cpu().numpy()

        self.recompute_context_selection()
        self.update_bbox_from_selection()

    # UI / interaction
    def draw_render_panel(self):
        if psim.TreeNode("Render"):
            psim.PushItemWidth(100)
            _, self.ui_state.render_mode_idx = psim.Combo(
                "Channel", self.ui_state.render_mode_idx, self.ui_constants.SUPPORTED_RENDER_MODES
            )
            if self.ui_state.supported_brushes is not None:
                brush_change, self.ui_state.selected_brush = psim.Combo(
                    "Brush", self.ui_state.selected_brush, self.ui_state.supported_brushes
                )
                if brush_change:
                    self.load_brush(self.ui_state.supported_brush_paths[self.ui_state.selected_brush])

            if psim.Button("Undo"):
                if self.prev_scene is not None:
                    self.render_kwargs["pc"] = self.prev_scene

            _, self.embedding_loop_enabled = psim.Checkbox(
                "Embedding Loop", self.embedding_loop_enabled
            )

            _, self.segmentation_mode_enabled = psim.Checkbox(
                "Segmentation Mode", self.segmentation_mode_enabled
            )

            overlays_changed, self.ui_state.show_selection_overlays = psim.Checkbox(
                "Show Selection Overlays", self.ui_state.show_selection_overlays
            )
            if overlays_changed:
                self.update_selection_visualizations()

            if psim.Button("Clear Selection"):
                self.clear_selection()

            _, self.grow_radius = psim.SliderFloat(
                "Selection Radius", self.grow_radius, v_min=0.001, v_max=0.25
            )
            _, self.similarity_threshold = psim.SliderFloat(
                "Similarity Threshold", self.similarity_threshold, v_min=0.0, v_max=1.0
            )
            _, self.heatmap_threshold = psim.SliderFloat(
                "Heatmap Threshold", self.heatmap_threshold, v_min=0.0, v_max=1.0
            )
            _, self.context_grow_radius = psim.SliderFloat(
                "Context Radius", self.context_grow_radius, v_min=0.001, v_max=0.25
            )

            psim.PopItemWidth()
            psim.TreePop()

    def draw_scene_settings_panel(self):
        if psim.TreeNode("Scene Settings"):
            if psim.Button("Reset Scene"):
                gaussians = GaussianModel(3)
                gaussians.load_ply(self.ui_state.scene_path)
                self.render_kwargs["pc"] = gaussians
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
                self.update_selection_visualizations()
                disp_change = True
                self.clip_box_coords = [[0, 0, 0], [0, 0, 0]]
                self.render_clip_box()
            if self.ui_state.display_mode_idx == 1:
                if psim.Button("Create Brush"):
                    self.exit_brush_edit_modes()
                    self.ui_state.display_mode_idx = 0
                    self.update_selection_visualizations()
                    ps.remove_point_cloud("brush axis", error_if_absent=False)
                    ps.remove_point_cloud("brush_stroke_center", error_if_absent=False)
                    disp_change = True
            
            input_text, self.ui_state.save_path = psim.InputText("Enter Save Path", self.ui_state.save_path)
            if self.ui_state.display_mode_idx == 1:
                ps.register_point_cloud("selected_xyz", np.array([[0.0, 0.0, 0.0]]))
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
                self.update_selection_visualizations()
            
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
        self.update_snap_pos()

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

        self.handle_segmentation_click()

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

        if self.embedding_loop_enabled:
            self.drive_run_loop()
