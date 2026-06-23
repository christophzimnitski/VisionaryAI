"""Unit tests for the Hungarian data-association step."""

from __future__ import annotations

import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.tracking.association import associate_detections_to_tracks

_CONFIGURATION = TrackerConfiguration()


def _associate(predicted_bounding_boxes, detection_bounding_boxes):
    """Call the association function with default gates/weights for two boxes."""
    predicted_centers = [
        (box[0] + box[2] / 2, box[1] + box[3] / 2) for box in predicted_bounding_boxes
    ]
    identity_covariances = [np.eye(2, dtype=np.float64) for _ in predicted_bounding_boxes]
    neutral_track_traits = [np.ones(9, dtype=np.float32) for _ in predicted_bounding_boxes]
    neutral_detection_traits = [
        np.ones(9, dtype=np.float32) for _ in detection_bounding_boxes
    ]
    return associate_detections_to_tracks(
        predicted_bounding_boxes,
        detection_bounding_boxes,
        predicted_centers,
        identity_covariances,
        neutral_track_traits,
        neutral_detection_traits,
        _CONFIGURATION.intersection_over_union_gate_threshold,
        _CONFIGURATION.mahalanobis_distance_gate_threshold,
        _CONFIGURATION.intersection_over_union_score_weight,
        _CONFIGURATION.mahalanobis_distance_score_weight,
        _CONFIGURATION.appearance_similarity_score_weight,
        _CONFIGURATION.mahalanobis_distance_score_normalization,
    )


def test_empty_inputs_produce_no_matches() -> None:
    matched_pairs, unmatched_detections, unmatched_tracks = _associate([], [])
    assert matched_pairs == []
    assert unmatched_detections == []
    assert unmatched_tracks == []


def test_single_overlapping_pair_is_matched() -> None:
    predicted_boxes = [(10, 10, 20, 20)]
    detection_boxes = [(11, 11, 20, 20)]
    matched_pairs, unmatched_detections, unmatched_tracks = _associate(
        predicted_boxes, detection_boxes
    )
    assert matched_pairs == [(0, 0)]
    assert unmatched_detections == []
    assert unmatched_tracks == []


def test_detection_far_outside_gate_stays_unmatched() -> None:
    predicted_boxes = [(10, 10, 20, 20)]
    detection_boxes = [(400, 400, 20, 20)]
    matched_pairs, unmatched_detections, unmatched_tracks = _associate(
        predicted_boxes, detection_boxes
    )
    assert matched_pairs == []
    assert unmatched_detections == [0]
    assert unmatched_tracks == [0]


def test_two_tracks_are_matched_to_their_nearest_detections() -> None:
    predicted_boxes = [(10, 10, 20, 20), (200, 200, 20, 20)]
    # Detections supplied in the opposite order to verify optimal assignment.
    detection_boxes = [(201, 201, 20, 20), (11, 11, 20, 20)]
    matched_pairs, unmatched_detections, unmatched_tracks = _associate(
        predicted_boxes, detection_boxes
    )
    assert set(matched_pairs) == {(0, 1), (1, 0)}
    assert unmatched_detections == []
    assert unmatched_tracks == []


def test_extra_detection_is_reported_as_unmatched() -> None:
    predicted_boxes = [(10, 10, 20, 20)]
    detection_boxes = [(11, 11, 20, 20), (300, 300, 20, 20)]
    matched_pairs, unmatched_detections, unmatched_tracks = _associate(
        predicted_boxes, detection_boxes
    )
    assert matched_pairs == [(0, 0)]
    assert unmatched_detections == [1]
    assert unmatched_tracks == []
