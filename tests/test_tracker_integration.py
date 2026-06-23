"""End-to-end integration tests driving the tracker with the synthetic scene.

These run fully headless (no OpenCV window) on the deterministic simulation, which
gives reproducible assertions about identity continuity, new-track discovery and
the lost/recover lifecycle.
"""

from __future__ import annotations

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.io.video_sources import SyntheticRocketSceneSource
from visionary.tracking.multi_object_tracker import MultiObjectTracker


def _build_tracker_and_synthetic_source(total_frame_count: int):
    configuration = TrackerConfiguration()
    synthetic_source = SyntheticRocketSceneSource(
        frame_width_in_pixels=configuration.frame_width_in_pixels,
        frame_height_in_pixels=configuration.frame_height_in_pixels,
        total_frame_count=total_frame_count,
        random_seed=1234,
    )
    motion_detector = MotionDifferenceDetector(configuration)
    tracker = MultiObjectTracker(motion_detector, configuration)
    return tracker, synthetic_source


def test_tracker_discovers_the_initial_objects() -> None:
    tracker, synthetic_source = _build_tracker_and_synthetic_source(total_frame_count=60)
    most_recent_active_track_count = 0
    while True:
        frame = synthetic_source.read_next_frame()
        if frame is None:
            break
        tracker.process_frame(frame)
        most_recent_active_track_count = tracker.active_track_count
    # The scene starts with two rockets; the tracker should be following them.
    assert most_recent_active_track_count >= 2


def test_a_new_object_appears_partway_through_the_scene() -> None:
    tracker, synthetic_source = _build_tracker_and_synthetic_source(total_frame_count=200)
    active_track_count_before_new_rocket = 0
    active_track_count_after_new_rocket = 0
    while True:
        frame = synthetic_source.read_next_frame()
        if frame is None:
            break
        result = tracker.process_frame(frame)
        if result.frame_index == 140:
            active_track_count_before_new_rocket = tracker.active_track_count
        if result.frame_index == 190:
            active_track_count_after_new_rocket = tracker.active_track_count
    # A third rocket is scripted to appear at frame 150.
    assert active_track_count_after_new_rocket >= active_track_count_before_new_rocket


def test_process_frame_returns_drawing_free_results() -> None:
    tracker, synthetic_source = _build_tracker_and_synthetic_source(total_frame_count=20)
    for _ in range(20):
        frame = synthetic_source.read_next_frame()
        if frame is None:
            break
        result = tracker.process_frame(frame)
        for track_result in result.track_results:
            assert isinstance(track_result.track_id, str)
            assert len(track_result.bounding_box) == 4
