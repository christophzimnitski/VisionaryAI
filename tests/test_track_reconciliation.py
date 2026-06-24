"""Tests for redundant-track reconciliation and long-run identity stability."""

from __future__ import annotations

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.evaluation.identity_consistency_evaluation import (
    evaluate_identity_consistency_on_synthetic_scene,
)
from visionary.geometry import BoundingBox
from visionary.tracking.multi_object_tracker import MultiObjectTracker
from visionary.tracking.track import TrackedObject


def _build_track_moving_horizontally(
    initial_left_x: int, velocity_x: int, frame_count: int
) -> TrackedObject:
    """Create a track and warm its Kalman filter to a steady horizontal velocity."""
    bounding_box: BoundingBox = (initial_left_x, 100, 40, 40)
    track = TrackedObject(bounding_box, current_frame_index=1)
    current_left_x = initial_left_x
    for frame_number in range(2, 2 + frame_count):
        track.predict_bounding_box()
        current_left_x += velocity_x
        track.update_with_matched_detection((current_left_x, 100, 40, 40), frame_number)
    return track


def _build_tracker() -> MultiObjectTracker:
    configuration = TrackerConfiguration()
    return MultiObjectTracker(MotionDifferenceDetector(configuration), configuration)


def test_co_moving_overlapping_tracks_are_treated_as_duplicates() -> None:
    tracker = _build_tracker()
    first_track = _build_track_moving_horizontally(0, velocity_x=5, frame_count=12)
    second_track = _build_track_moving_horizontally(6, velocity_x=5, frame_count=12)
    # Both end up at nearly the same place moving the same way.
    assert tracker._two_active_tracks_are_duplicates(first_track, second_track)


def test_crossing_tracks_are_not_merged_despite_overlapping() -> None:
    tracker = _build_tracker()
    rightward_track = _build_track_moving_horizontally(0, velocity_x=5, frame_count=12)
    leftward_track = _build_track_moving_horizontally(120, velocity_x=-5, frame_count=12)
    # Their boxes overlap at the crossing, but opposite velocities mean they are
    # two distinct objects and must never be merged.
    assert not tracker._two_active_tracks_are_duplicates(rightward_track, leftward_track)


def test_reconciliation_culls_the_younger_duplicate_track() -> None:
    tracker = _build_tracker()
    older_track = _build_track_moving_horizontally(0, velocity_x=5, frame_count=12)
    younger_track = _build_track_moving_horizontally(6, velocity_x=5, frame_count=12)
    younger_track.creation_frame_index = older_track.creation_frame_index + 5

    tracker._active_tracks_by_id = {
        older_track.track_id: older_track,
        younger_track.track_id: younger_track,
    }
    surviving_results = tracker._merge_redundant_coincident_active_tracks([])

    assert older_track.track_id in tracker._active_tracks_by_id
    assert younger_track.track_id not in tracker._active_tracks_by_id
    assert surviving_results == []


def test_a_fragment_inside_a_claimed_region_does_not_spawn_a_new_track() -> None:
    """A small motion fragment shed during a collision must count as already claimed.

    Such a fragment sits inside a tracked object's region but overlaps its box too
    little to clear the IoU gate, so without the containment check it would spawn a
    spurious, too-small track that then steals the real object.
    """
    tracker = _build_tracker()
    claimed_region_boxes = [(250, 260, 74, 72)]

    fragment_inside_claimed_region = (268, 288, 32, 31)
    assert tracker._detection_is_already_claimed(
        fragment_inside_claimed_region, claimed_region_boxes
    )

    detection_well_outside = (10, 10, 40, 40)
    assert not tracker._detection_is_already_claimed(
        detection_well_outside, claimed_region_boxes
    )


def test_identity_is_stable_while_objects_are_separated() -> None:
    """Over a multi-crossing run, identity should hold whenever objects are apart.

    This is the regression guard for the coalescence work: switches are only
    permitted at the irreducible moments of coincidence, never while the rockets
    are clearly separated, and recall must stay high throughout.
    """
    summary = evaluate_identity_consistency_on_synthetic_scene(total_frame_count=280)
    assert summary.detection_recall > 0.9
    assert summary.total_identity_switch_count_while_separated <= 2
