"""Unit tests for the anticipatory focus layer (per-track local search)."""

from __future__ import annotations

import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.tracking.anticipatory_focus_tracker import (
    AnticipatoryFocusTracker,
    PredictedTrackState,
)


def _build_frame_with_object_at(
    object_left_x: int, object_top_y: int, frame_size: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """Return a black frame with one textured object drawn in, plus its template.

    The object is half white / half black so it has the intensity variance that
    normalized cross-correlation needs (a uniform patch would be degenerate).
    """
    object_width, object_height = 24, 24
    object_patch = np.zeros((object_height, object_width, 3), dtype=np.uint8)
    object_patch[:, : object_width // 2] = 255
    frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
    frame[
        object_top_y : object_top_y + object_height,
        object_left_x : object_left_x + object_width,
    ] = object_patch
    return frame, object_patch


def test_focus_layer_relocates_an_object_near_its_prediction() -> None:
    focus_tracker = AnticipatoryFocusTracker(TrackerConfiguration())
    frame, template_patch = _build_frame_with_object_at(object_left_x=50, object_top_y=50)

    # The object is at (50, 50); the prediction is a few pixels off, as it would
    # be from a Kalman one-step-ahead estimate.
    predicted_track_state = PredictedTrackState(
        track_id="track-1",
        predicted_center_x=58.0,
        predicted_center_y=60.0,
        predicted_bounding_box=(46, 48, 24, 24),
        clean_appearance_template_patch=template_patch,
    )
    placements = focus_tracker.locate_tracks_within_search_regions(frame, [predicted_track_state])

    assert len(placements) == 1
    placement = placements[0]
    assert placement.is_confidently_located
    located_left_x, located_top_y, _, _ = placement.located_bounding_box
    assert abs(located_left_x - 50) <= 2
    assert abs(located_top_y - 50) <= 2


def test_focus_layer_abstains_when_it_has_no_template_yet() -> None:
    focus_tracker = AnticipatoryFocusTracker(TrackerConfiguration())
    frame, _ = _build_frame_with_object_at(object_left_x=50, object_top_y=50)

    predicted_track_state = PredictedTrackState(
        track_id="track-1",
        predicted_center_x=62.0,
        predicted_center_y=62.0,
        predicted_bounding_box=(50, 50, 24, 24),
        clean_appearance_template_patch=None,
    )
    placement = focus_tracker.locate_tracks_within_search_regions(frame, [predicted_track_state])[0]
    assert not placement.is_confidently_located


def test_occlusion_ordering_keeps_only_one_track_on_a_shared_blob() -> None:
    focus_tracker = AnticipatoryFocusTracker(TrackerConfiguration())
    frame, template_patch = _build_frame_with_object_at(object_left_x=50, object_top_y=50)

    # Two tracks predict almost the same place over the single visible object.
    first_track_state = PredictedTrackState(
        track_id="track-1",
        predicted_center_x=62.0,
        predicted_center_y=62.0,
        predicted_bounding_box=(50, 50, 24, 24),
        clean_appearance_template_patch=template_patch,
    )
    second_track_state = PredictedTrackState(
        track_id="track-2",
        predicted_center_x=66.0,
        predicted_center_y=62.0,
        predicted_bounding_box=(54, 50, 24, 24),
        clean_appearance_template_patch=template_patch,
    )
    placements = focus_tracker.locate_tracks_within_search_regions(
        frame, [first_track_state, second_track_state]
    )

    confidently_located = [p for p in placements if p.is_confidently_located]
    assert len(confidently_located) == 1
    # The track that keeps the blob is flagged as contesting it, so the tracker
    # protects its velocity from coalescence.
    assert confidently_located[0].is_contesting_a_shared_blob
