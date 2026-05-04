# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import polyscope as ps
import polyscope.imgui as psim
from typing import Tuple
from src.polyviewer.entities.payload import CallbackPayload
from src.polyviewer.entities.selection import GSplatSelection

_RECT_COLOR = (255 << 24) | (0 << 16) | (255 << 8) | 255  # ImU32 yellow, full alpha


class DragHandler:
    def __init__(self, continuous_selection=False):
        self.is_dragging: bool = False
        self.drag_just_finalized: bool = False
        self.lastMouseClick: Tuple[float, float] = None
        self.lastDragDelta = None
        self.navigation_style: Tuple[float, float] = None
        self.continuous_selection: bool = continuous_selection
        self.assigned_mouse_button = psim.ImGuiMouseButton_Middle
        self.m_left_drag: bool = False

    def _drag_start(self):
        self.lastMouseClick = psim.GetMousePos()
        self.navigation_style = ps.get_navigation_style()
        ps.set_navigation_style("none")
        self.is_dragging = True

    def _drag_move(self):
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
        y0 = 1.0 - 2.0 * y0
        y1 = 1.0 - 2.0 * y1
        if self.continuous_selection:
            return x0, y0, x1, y1
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
        self.drag_just_finalized = False
        io = psim.GetIO()
        drag_bounds = None
        finalize_selection = False
        if psim.IsAnyMouseDown():
            if psim.IsMouseDown(self.assigned_mouse_button) and not self.is_dragging:
                self._drag_start()
                self.m_left_drag = False
            elif psim.IsKeyDown(psim.ImGuiKey_M) and psim.IsMouseDown(psim.ImGuiMouseButton_Left) and not self.is_dragging:
                self._drag_start()
                self.m_left_drag = True
            if self.is_dragging and self.lastMouseClick is not None:
                drag_bounds = self._drag_move()
        release_button = psim.ImGuiMouseButton_Left if self.m_left_drag else self.assigned_mouse_button
        if psim.IsMouseReleased(release_button):
            if self.is_dragging:
                drag_bounds = self._drag_end()
                finalize_selection = True
                ps.set_navigation_style(self.navigation_style)
                self.is_dragging = False
                self.m_left_drag = False
            else:
                payload.last_selection.reset()

        if self.is_dragging and self.lastMouseClick is not None:
            curr = psim.GetMousePos()
            p_min = (min(self.lastMouseClick[0], curr[0]), min(self.lastMouseClick[1], curr[1]))
            p_max = (max(self.lastMouseClick[0], curr[0]), max(self.lastMouseClick[1], curr[1]))
            psim.GetBackgroundDrawList().AddRect(p_min, p_max, _RECT_COLOR, 6.0, 0, 2.0)

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
                self.drag_just_finalized = True
                payload.last_selection = payload.selection_preview

    def is_drag_in_progress(self):
        return self.is_dragging
