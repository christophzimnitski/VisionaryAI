"""Detector abstraction.

A detector turns a single video frame into a list of candidate object detections.
Concrete detectors (template matching, motion differencing, or a future learned
detector such as YOLO) are interchangeable because they all satisfy the
``ObjectDetector`` protocol, so the tracker never depends on a specific one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from visionary.geometry import BoundingBox


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame."""

    bounding_box: BoundingBox
    detection_confidence: float = 1.0


class ObjectDetector(Protocol):
    """Anything that can find candidate objects in a frame."""

    def detect_objects_in_frame(self, frame: np.ndarray) -> list[Detection]:
        """Return all candidate detections found in ``frame``."""
        ...
