"""Accumulates frame-to-frame camera motion into a running world offset.

The egomotion estimator gives the homography between *consecutive* frames. That
is enough to keep predictions honest while an object stays in view, but it is not
enough to answer "is the spot where I left the pillow back on screen now?" -- for
that the system needs to know where the current viewport sits in a fixed *room*
coordinate frame, accumulated across the whole pan.

This tracker integrates each frame's motion into that cumulative offset. The room
frame is anchored to the first frame, so a point at frame coordinate ``f`` maps to
world coordinate ``f + world_offset`` and back. Because the offset is integrated
from per-frame estimates it slowly drifts over a long pan (the known limitation):
this gives an approximate, lightweight room localization, not survey-grade SLAM.
"""

from __future__ import annotations

import numpy as np


class CameraPoseTracker:
    """Tracks the current viewport's translation within a fixed room frame."""

    def __init__(self) -> None:
        # World coordinate of the current frame's top-left corner. Starts at the
        # origin: the first frame *defines* the room coordinate frame.
        self._world_offset_x = 0.0
        self._world_offset_y = 0.0

    def update_with_frame_to_frame_homography(
        self, frame_to_frame_homography: np.ndarray | None, frame_shape: tuple[int, ...]
    ) -> None:
        """Fold one frame's egomotion into the cumulative world offset.

        ``frame_to_frame_homography`` maps *previous*-frame points to current-frame
        points (the convention the egomotion estimator returns). A world-fixed
        point therefore shifts within the frame by that homography; the viewport
        has moved through the room by the opposite amount, so the world offset
        grows by the negated content shift. ``None`` (motion could not be
        estimated) leaves the offset unchanged.
        """
        if frame_to_frame_homography is None:
            return
        content_shift_x, content_shift_y = self._content_shift_at_frame_center(
            frame_to_frame_homography, frame_shape
        )
        self._world_offset_x -= content_shift_x
        self._world_offset_y -= content_shift_y

    @staticmethod
    def _content_shift_at_frame_center(
        frame_to_frame_homography: np.ndarray, frame_shape: tuple[int, ...]
    ) -> tuple[float, float]:
        """How far a point at the frame centre moved between the two frames."""
        frame_height, frame_width = frame_shape[:2]
        frame_center = np.array(
            [frame_width / 2.0, frame_height / 2.0, 1.0], dtype=np.float64
        )
        mapped_center = frame_to_frame_homography @ frame_center
        if abs(mapped_center[2]) < 1e-8:
            return 0.0, 0.0
        mapped_center_x = mapped_center[0] / mapped_center[2]
        mapped_center_y = mapped_center[1] / mapped_center[2]
        return mapped_center_x - frame_center[0], mapped_center_y - frame_center[1]

    def world_position_of(self, frame_x: float, frame_y: float) -> tuple[float, float]:
        """Map a current-frame coordinate to its fixed room coordinate."""
        return frame_x + self._world_offset_x, frame_y + self._world_offset_y

    def frame_position_of(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Map a fixed room coordinate to where it currently sits in the frame."""
        return world_x - self._world_offset_x, world_y - self._world_offset_y

    @property
    def current_world_offset(self) -> tuple[float, float]:
        """World coordinate of the current frame's top-left corner."""
        return self._world_offset_x, self._world_offset_y
