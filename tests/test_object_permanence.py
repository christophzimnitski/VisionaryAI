"""End-to-end test of object permanence: leave the frame, then be re-identified.

A single distinct object cruises off the right edge, spends frames fully out of
view, and returns. The tracker must (a) keep reporting it while it is gone, with
an off-screen pointer aimed at the correct edge, and (b) restore its *original*
identity when it re-enters -- recognising it by its remembered features rather
than spawning a new track.
"""

from __future__ import annotations

import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.geometry import BoundingBox, intersection_over_union
from visionary.tracking.multi_object_tracker import MultiObjectTracker

_FRAME_WIDTH = 640
_FRAME_HEIGHT = 480
_OBJECT_SIZE = 48
_OBJECT_TOP_Y = 210


def _build_distinct_object_sprite() -> np.ndarray:
    sprite = np.zeros((_OBJECT_SIZE, _OBJECT_SIZE, 3), dtype=np.uint8)
    sprite[:] = (180, 60, 30)
    sprite[10:38, 10:38] = (255, 200, 120)
    return sprite


def _render_frame(
    random_number_generator: np.random.Generator, object_left_x: int | None, sprite: np.ndarray
) -> tuple[np.ndarray, BoundingBox | None]:
    frame = random_number_generator.integers(
        0, 5, (_FRAME_HEIGHT, _FRAME_WIDTH, 3), dtype=np.uint8
    )
    if object_left_x is None:
        return frame, None
    visible_left_x = max(0, object_left_x)
    visible_right_x = min(_FRAME_WIDTH, object_left_x + _OBJECT_SIZE)
    if visible_right_x <= visible_left_x:
        return frame, None
    frame[_OBJECT_TOP_Y : _OBJECT_TOP_Y + _OBJECT_SIZE, visible_left_x:visible_right_x] = sprite[
        :, visible_left_x - object_left_x : visible_right_x - object_left_x
    ]
    return frame, (visible_left_x, _OBJECT_TOP_Y, visible_right_x - visible_left_x, _OBJECT_SIZE)


def _best_overlapping_track_id(ground_truth_box, track_results) -> str | None:
    best_track_id, best_overlap = None, 0.0
    for track_result in track_results:
        overlap = intersection_over_union(ground_truth_box, track_result.bounding_box)
        if overlap > best_overlap and overlap >= 0.3:
            best_track_id, best_overlap = track_result.track_id, overlap
    return best_track_id


def test_object_is_re_identified_after_leaving_and_returning() -> None:
    configuration = TrackerConfiguration()
    tracker = MultiObjectTracker(MotionDifferenceDetector(configuration), configuration)
    random_number_generator = np.random.default_rng(0)
    sprite = _build_distinct_object_sprite()

    # Cruise right off the edge, stay gone for a stretch, then come back.
    object_positions = (
        [80 + 4 * step for step in range(130)]
        + [None] * 40
        + [600 - 4 * step for step in range(50)]
    )

    identity_before_leaving = None
    identity_after_returning = None
    off_screen_pointer_frame_count = 0
    most_recent_bearing = None
    has_left_view = False
    for object_left_x in object_positions:
        frame, ground_truth_box = _render_frame(random_number_generator, object_left_x, sprite)
        frame_processing_result = tracker.process_frame(frame)

        out_of_view_results = [
            track_result
            for track_result in frame_processing_result.track_results
            if track_result.is_out_of_view
        ]
        if out_of_view_results:
            off_screen_pointer_frame_count += 1
            most_recent_bearing = out_of_view_results[
                0
            ].off_screen_expectation.bearing_degrees_clockwise_from_up

        if ground_truth_box is not None and not has_left_view:
            identity_before_leaving = (
                _best_overlapping_track_id(ground_truth_box, frame_processing_result.track_results)
                or identity_before_leaving
            )
        if object_left_x is None:
            has_left_view = True
        if ground_truth_box is not None and has_left_view and identity_after_returning is None:
            identity_after_returning = _best_overlapping_track_id(
                ground_truth_box, frame_processing_result.track_results
            )

    assert identity_before_leaving is not None
    # While gone, the tracker kept pointing at it toward the right edge (~90 deg).
    assert off_screen_pointer_frame_count >= 10
    assert most_recent_bearing is not None
    assert abs(most_recent_bearing - 90.0) < 35.0
    # And it came back with the same identity, not a freshly spawned one.
    assert identity_after_returning == identity_before_leaving
