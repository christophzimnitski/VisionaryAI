"""Camera-stage test: a designated object tracked across a panning camera.

The camera pans across a static wall. A designated painting is registered on the
first frame and must (a) keep that identity while it is in view -- the prediction
being driven by estimated camera motion, not object motion -- and (b) once the
camera has panned past it, be reported out of view with a pointer aimed at the
edge it disappeared behind.
"""

from __future__ import annotations

import dataclasses

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import NullObjectDetector
from visionary.geometry import intersection_over_union
from visionary.io.video_sources import PanningCameraSceneSource
from visionary.tracking.multi_object_tracker import MultiObjectTracker


def _best_overlapping_track_id(ground_truth_box, track_results) -> str | None:
    best_track_id, best_overlap = None, 0.0
    for track_result in track_results:
        overlap = intersection_over_union(ground_truth_box, track_result.bounding_box)
        if overlap > best_overlap and overlap >= 0.3:
            best_track_id, best_overlap = track_result.track_id, overlap
    return best_track_id


def test_designated_painting_is_tracked_then_pointed_at_off_screen() -> None:
    configuration = dataclasses.replace(
        TrackerConfiguration(), enable_camera_motion_compensation=True
    )
    scene_source = PanningCameraSceneSource(total_frame_count=70, random_seed=0)
    tracker = MultiObjectTracker(NullObjectDetector(), configuration)

    registered_track_id = None
    in_view_frame_count = 0
    in_view_frames_matching_registered_id = 0
    off_screen_pointer_bearings: list[float] = []

    for frame, ground_truth_boxes in scene_source.iterate_frames_with_ground_truth():
        if scene_source._current_frame_index == 1:
            tracker.process_frame(frame)
            registered_track_id = tracker.register_object_of_interest(
                ground_truth_boxes["painting_red"], frame
            )
            continue

        frame_processing_result = tracker.process_frame(frame)
        painting_box = ground_truth_boxes.get("painting_red")
        if painting_box is not None:
            in_view_frame_count += 1
            if (
                _best_overlapping_track_id(painting_box, frame_processing_result.track_results)
                == registered_track_id
            ):
                in_view_frames_matching_registered_id += 1
        for track_result in frame_processing_result.track_results:
            if track_result.is_out_of_view and track_result.off_screen_expectation is not None:
                off_screen_pointer_bearings.append(
                    track_result.off_screen_expectation.bearing_degrees_clockwise_from_up
                )

    assert registered_track_id is not None
    # The painting keeps its registered identity for most of its time in view,
    # driven by camera-motion prediction.
    assert in_view_frames_matching_registered_id >= 0.5 * in_view_frame_count
    # Once panned past, it is pointed at toward the left edge it vanished behind.
    assert off_screen_pointer_bearings
    assert all(abs(bearing - 270.0) < 40.0 for bearing in off_screen_pointer_bearings)
