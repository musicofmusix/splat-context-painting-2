# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import polyscope as ps
from src.segment.entities import Segment


class GuiSegment(Segment):
    def __init__(self, mask, name=None):
        super().__init__(mask, name)
        self.properties = dict()
        self.point_cloud = ps.register_point_cloud(
            self.name, np.zeros([1, 3]), transparency=0.0
        )
        self.point_cloud.set_enabled(self.is_enabled)
        self.attributes = self.define_attributes()

    @classmethod
    def from_dict(cls, dict_val):
        res = GuiSegment(mask=dict_val["mask"], name=dict_val["name"])
        if "color" in dict_val:
            res.color = dict_val["color"]
        if "transform" in dict_val:
            res.transform = dict_val["transform"]
        if "properties" in dict_val:
            res.properties = dict_val["properties"]
        if "attributes" in dict_val:
            res.attributes = dict_val["attributes"]
        return res

    def as_dict(self):
        res = super().as_dict()
        res.update(
            {
                "properties": self.properties,
                "attributes": self.attributes,
                "color": self.color,
                "transform": self.transform,
            }
        )
        return res

    def define_attributes(self):
        return {
            "Young's Modulus [Pa]": 1e6,
            "Rho": 1e2,
            "Poisson Ratio": 0.45,
            "Volume Approximation": 1.0,
        }

    def _set_is_enabled(self, val):
        super()._set_is_enabled(val)
        self.point_cloud.set_enabled(val)

    @property
    def color(self):
        return self.point_cloud.get_color()

    @color.setter
    def color(self, val):
        self.point_cloud.set_color(val)

    @property
    def transform(self):
        return self.point_cloud.get_transform()

    @transform.setter
    def transform(self, mat):
        self.point_cloud.set_transform(mat)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["point_cloud"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.point_cloud = ps.register_point_cloud(
            self.name, np.zeros([0, 3]), transparency=0.0
        )
