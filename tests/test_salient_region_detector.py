"""Unit tests for the salient-region detector (auto-discovery of resting objects)."""

from __future__ import annotations

import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.salient_region_detector import SalientRegionDetector
from visionary.geometry import intersection_over_union

_FRAME_WIDTH = 640
_FRAME_HEIGHT = 480


def _room_like_frame_with_two_distinct_objects() -> np.ndarray:
    # A near-uniform, faintly textured wall (the dominant background colour) with
    # two big, distinctly coloured objects and one tiny speck that is too small.
    random_number_generator = np.random.default_rng(0)
    frame = np.full((_FRAME_HEIGHT, _FRAME_WIDTH, 3), 90, dtype=np.uint8)
    frame += random_number_generator.integers(0, 6, frame.shape, dtype=np.uint8)
    frame[100:160, 100:180] = (40, 40, 205)  # a red object, 80 x 60
    frame[300:370, 400:470] = (40, 180, 40)  # a green object, 70 x 70
    frame[10:20, 10:20] = (205, 40, 40)  # a tiny blue speck, far below the size gate
    return frame


def test_two_big_distinct_objects_are_proposed_and_the_speck_is_ignored() -> None:
    detector = SalientRegionDetector(TrackerConfiguration())
    detections = detector.detect_objects_in_frame(_room_like_frame_with_two_distinct_objects())

    assert len(detections) == 2
    detected_boxes = [detection.bounding_box for detection in detections]
    red_object_box = (100, 100, 80, 60)
    green_object_box = (400, 300, 70, 70)
    for expected_box in (red_object_box, green_object_box):
        best_overlap = max(
            intersection_over_union(expected_box, detected_box) for detected_box in detected_boxes
        )
        assert best_overlap >= 0.6


def test_a_uniform_frame_proposes_nothing() -> None:
    detector = SalientRegionDetector(TrackerConfiguration())
    uniform_frame = np.full((_FRAME_HEIGHT, _FRAME_WIDTH, 3), 120, dtype=np.uint8)
    assert detector.detect_objects_in_frame(uniform_frame) == []
