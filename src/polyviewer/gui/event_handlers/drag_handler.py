# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import Tuple
from src.polyviewer.entities.payload import CallbackPayload
from src.polyviewer.entities.selection import GSplatSelection


class DragHandler:
    GIZMOS_GROUP = "GIZMO_ELEMENTS"

    def __init__(self, continuous_selection=False):
        selection_rect = ps.register_curve_network(
            "selection_rect",
            nodes=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            edges="loop",
            color=(1.0, 1.0, 1.0),
            enabled=False,
        )
        selection_rect.set_radius(0.005, relative=False)
        # selection_rect.set_is_using_screen_coords(True)
        gizmos_group = ps.create_group(DragHandler.GIZMOS_GROUP)
        # gizmos_group.set_is_hide_from_ui(True)
        gizmos_group.set_hide_descendants_from_structure_lists(True)
        gizmos_group.set_show_child_details(False)
        selection_rect.add_to_group(gizmos_group)
        self.is_dragging: bool = False
        self.lastMouseClick: Tuple[float, float] = None
        self.lastDragDelta = None
        self.navigation_style: Tuple[float, float] = None
        self.continuous_selection: bool = continuous_selection
        self.assigned_mouse_button = psim.ImGuiMouseButton_Middle

    @property
    def selection_rect(self):
        return ps.get_curve_network("selection_rect")

    def _drag_start(self):
        self.lastMouseClick = psim.GetMousePos()
        self.navigation_style = ps.get_navigation_style()
        ps.set_navigation_style("none")
        self.is_dragging = True

    def _drag_move(self):
        window_w, window_h = ps.get_window_size()
        self.selection_rect.set_enabled(True)
        currMousePos = psim.GetMousePos()
        self.lastDragDelta = (
            currMousePos[0] - self.lastMouseClick[0],
            currMousePos[1] - self.lastMouseClick[1],
        )
        x0 = self.lastMouseClick[0] / window_w
        y0 = self.lastMouseClick[1] / window_h
        x1 = x0 + self.lastDragDelta[0] / window_w
        y1 = y0 + self.lastDragDelta[1] / window_h
        x0 = 2.0 * x0 - 1.0
        x1 = 2.0 * x1 - 1.0
        y0 = 1.0 - 2.0 * y0
        y1 = 1.0 - 2.0 * y1

        p0 = ps.screen_coords_to_world_position([self.lastMouseClick[0], self.lastMouseClick[1]])
        p1 = ps.screen_coords_to_world_position([currMousePos[0], self.lastMouseClick[1]])
        p2 = ps.screen_coords_to_world_position([currMousePos[0], currMousePos[1]])
        p3 = ps.screen_coords_to_world_position([self.lastMouseClick[0], currMousePos[1]])

        self.selection_rect.update_node_positions(
            np.array([p0, p1, p2, p3])
        )
        
        if self.continuous_selection:
            return x0, y0, x1, y1
        else:
            return None

    def _drag_end(self):
        window_w, window_h = ps.get_window_size()
        currMousePos = psim.GetMousePos()
        self.lastDragDelta = (
            currMousePos[0] - self.lastMouseClick[0],
            currMousePos[1] - self.lastMouseClick[1],
        )
        x0 = self.lastMouseClick[0] / window_w
        y0 = self.lastMouseClick[1] / window_h
        x1 = x0 + self.lastDragDelta[0] / window_w
        y1 = y0 + self.lastDragDelta[1] / window_h
        x0 = 2.0 * x0 - 1.0
        x1 = 2.0 * x1 - 1.0
        y0 = 2.0 * y0 - 1.0
        y1 = 2.0 * y1 - 1.0
        return x0, y0, x1, y1

    def handle_callback(self, payload: CallbackPayload):
        io = psim.GetIO()
        drag_bounds = None
        finalize_selection = False
        if psim.IsAnyMouseDown():
            if psim.IsMouseDown(self.assigned_mouse_button) and not self.is_dragging:
                self._drag_start()
            if self.is_dragging and self.lastMouseClick is not None:
                drag_bounds = self._drag_move()
        if psim.IsMouseReleased(self.assigned_mouse_button):
            if self.is_dragging:
                drag_bounds = self._drag_end()
                finalize_selection = True
                ps.set_navigation_style(self.navigation_style)
                self.selection_rect.set_enabled(False)
                self.is_dragging = False
            else:
                payload.last_selection.reset()
        if drag_bounds is not None:
            x0, y0, x1, y1 = drag_bounds
            x0, x1 = (x1, x0) if x0 > x1 else (x0, x1)
            y0, y1 = (y1, y0) if y0 > y1 else (y0, y1)
            payload.drag_bounds = x0, y0, x1, y1
            if io.KeyShift:
                payload.selection_preview = payload.last_selection.add(payload)
            elif io.KeyAlt:
                payload.selection_preview = payload.last_selection.remove(payload)
            elif io.KeyCtrl:
                payload.selection_preview = payload.last_selection.intersect(payload)
            else:
                payload.selection_preview = GSplatSelection().select(payload)
            if finalize_selection:
                payload.last_selection = payload.selection_preview

    def is_drag_in_progress(self):
        return self.is_dragging
