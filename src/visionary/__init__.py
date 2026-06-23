"""Visionary: a modular, tested multi-object tracking pipeline.

The public surface re-exports the pieces most callers need: the configuration,
the detectors, and the tracker itself.
"""

from __future__ import annotations

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.detection.template_matching_detector import TemplateMatchingDetector
from visionary.tracking.multi_object_tracker import (
    FrameProcessingResult,
    MultiObjectTracker,
    TrackResult,
)

__all__ = [
    "TrackerConfiguration",
    "MotionDifferenceDetector",
    "TemplateMatchingDetector",
    "MultiObjectTracker",
    "FrameProcessingResult",
    "TrackResult",
]

__version__ = "0.1.0"
