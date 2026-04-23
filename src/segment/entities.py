# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import torch

logger = logging.getLogger(__name__)


class Segment:
    running_counter = 1

    def __init__(self, mask, name=None):
        self.mask = mask
        if name is None or name == f"Segment {Segment.running_counter}":
            name = f"Segment {Segment.running_counter}"
            Segment.running_counter += 1
        self.name = name
        self.__enabled = True

    @classmethod
    def from_dict(cls, dict_val):
        return Segment(mask=dict_val["mask"], name=dict_val["name"])

    @property
    def is_enabled(self):
        return self._get_is_enabled()

    @is_enabled.setter
    def is_enabled(self, val):
        self._set_is_enabled(val)

    def as_dict(self):
        return {"name": self.name, "mask": self.mask}

    def _get_is_enabled(self):
        return self.__enabled

    def _set_is_enabled(self, val):
        self.__enabled = val

    def is_empty(self) -> bool:
        return not self.mask.any()


class DisjointSegmentation:
    BACKGROUND_NAME = "background"

    def __init__(self, num_points, device, create_segment=None):
        self.create_segment = create_segment if create_segment is not None else Segment
        self.num_points = num_points
        self.device = device
        self.segments = {
            DisjointSegmentation.BACKGROUND_NAME: self.__create_background_segment()
        }

    @property
    def foreground_segments(self):
        return {
            k: v
            for k, v in self.segments.items()
            if k != DisjointSegmentation.BACKGROUND_NAME
        }

    def __create_background_segment(self, value=1):
        if value == 1:
            mask = torch.ones((self.num_points,), dtype=torch.bool, device=self.device)
        else:
            mask = torch.zeros((self.num_points,), dtype=torch.bool, device=self.device)
        return self.create_segment(mask, DisjointSegmentation.BACKGROUND_NAME)

    def save_segments_to_file(self, path):
        torch.save({k: v.as_dict() for k, v in self.segments.items()}, path)

    def load_segments_from_file(self, path):
        segment_dict = torch.load(path)
        if len(segment_dict) == 0:
            return False
        num_pts = -1
        for k, v in segment_dict.items():
            if num_pts < 0:
                num_pts = v["mask"].shape[0]
            else:
                if v["mask"].shape[0] != num_pts:
                    logger.error(f"Saved segments have inconsistent mask sizes")
                    return False
        self.num_points = num_pts
        self.segments = {
            k: self.create_segment.from_dict(v) for k, v in segment_dict.items()
        }
        for k, s in self.segments.items():
            s.mask = s.mask.to(self.device)
        if DisjointSegmentation.BACKGROUND_NAME not in segment_dict:
            logger.warning(
                f"Loaded segments have no {DisjointSegmentation.BACKGROUND_NAME}, adding dummy one"
            )
            self.segments[DisjointSegmentation.BACKGROUND_NAME] = (
                self.__create_background_segment(0)
            )
        return True

    def num_segments(self, enabled_only=False):
        if not enabled_only:
            return len(self.segments)
        return sum([1 for k, v in self.segments.items() if v.is_enabled])

    def has_segment(self, name):
        return name in self.segments

    def get_segment(self, name) -> Segment:
        return self.segments.get(name)

    def get_background_segment(self):
        return self.get_segment(DisjointSegmentation.BACKGROUND_NAME)

    def _check_segment_exists(self, name, raise_error=True):
        exists = self.has_segment(name)
        if not exists and raise_error:
            raise KeyError(
                f"Segment with name {name} not found; existing segments: {self.segments.keys()}"
            )
        return exists

    def num_segment_points(self, name):
        self._check_segment_exists(name)
        return torch.sum(self.segments[name].mask).item()

    def add_segment(self, name):
        if self.has_segment(name):
            logger.warning(
                f"Cannot create new segment: Segment {name} already exists with {self.num_segment_points(name)}"
            )
            return False
        self.segments[name] = self.create_segment(
            torch.zeros((self.num_points,), dtype=torch.bool, device=self.device), name
        )
        return True

    def update_segment_mask(self, name, mask):
        self._check_segment_exists(name)
        logger.info(f"Updating segment {name} with new mask")
        if name == DisjointSegmentation.BACKGROUND_NAME:
            logger.warning(f"Cannot manually set background mask.")
            return
        removed_mask = self.segments[name].mask & ~mask
        for seg_name, seg in self.segments.items():
            prev_pts = self.num_segment_points(seg_name)
            if seg_name == name:
                self.segments[seg_name].mask = mask
            elif seg_name == DisjointSegmentation.BACKGROUND_NAME:
                self.segments[seg_name].mask[removed_mask] = True
                seg.mask[mask] = False
            else:
                seg.mask[mask] = False
            logger.info(
                f"  > Updated segment {seg_name}: {prev_pts}pts --> {self.num_segment_points(seg_name)}"
            )

    def prune_segments(self, prune_mask):
        valid_points_mask = ~prune_mask
        for seg_name, seg in self.segments.items():
            prev_pts = self.num_segment_points(seg_name)
            seg.mask = seg.mask[valid_points_mask]
            logger.info(
                f"  > Updated segment {seg_name}: {prev_pts}pts --> {self.num_segment_points(seg_name)}"
            )

    def append_points(self, num_points, segment_name=BACKGROUND_NAME):
        for seg_name, seg in self.segments.items():
            if seg_name == segment_name:
                append_mask = torch.ones(num_points).to(seg.mask)
            else:
                append_mask = torch.zeros(num_points).to(seg.mask)
            seg.mask = torch.cat([seg.mask, append_mask], dim=0)

    def delete_segment(self, name):
        if not self.can_delete_segment(name):
            return False
        self.segments[DisjointSegmentation.BACKGROUND_NAME].mask |= self.segments[
            name
        ].mask
        del self.segments[name]
        return True

    def can_delete_segment(self, name):
        exists = self._check_segment_exists(name, raise_error=False)
        return exists and not name == DisjointSegmentation.BACKGROUND_NAME

    def get_enabled_segments_mask(self):
        mask = torch.zeros_like(
            self.segments[DisjointSegmentation.BACKGROUND_NAME].mask
        )
        for name, s in self.segments.items():
            if s.is_enabled:
                mask |= s.mask
        return mask
