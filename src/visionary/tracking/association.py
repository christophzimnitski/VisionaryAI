"""Data association between predicted tracks and new detections.

Given the Kalman-predicted state of each existing track and the new detections in
the current frame, decide which detection (if any) belongs to which track.

This module replaces the original greedy, sort-by-IoU matching with an *optimal*
assignment computed by the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
over a combined cost that fuses three cues:

  * spatial overlap (Intersection-over-Union),
  * motion consistency (Mahalanobis distance under the Kalman innovation covariance),
  * appearance similarity (cosine similarity of trait vectors).

Pairs that fail either the IoU gate or the Mahalanobis gate are forbidden, so the
configured gate thresholds are genuinely honoured (the original code accepted an
``iou_threshold`` argument and then ignored it).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from visionary.features.appearance_traits import (
    compute_appearance_trait_cosine_similarity,
)
from visionary.geometry import BoundingBox, intersection_over_union

# Cost assigned to a forbidden (gated-out) track/detection pairing. Any matched
# pair whose cost is at least this value is rejected after the assignment solve.
_FORBIDDEN_PAIR_COST = 1.0e6


def associate_detections_to_tracks(
    predicted_track_bounding_boxes: list[BoundingBox],
    detection_bounding_boxes: list[BoundingBox],
    predicted_track_centers: list[tuple[float, float]],
    track_innovation_covariances: list[np.ndarray],
    track_appearance_trait_vectors: list[np.ndarray],
    detection_appearance_trait_vectors: list[np.ndarray],
    intersection_over_union_gate_threshold: float,
    mahalanobis_distance_gate_threshold: float,
    intersection_over_union_score_weight: float,
    mahalanobis_distance_score_weight: float,
    appearance_similarity_score_weight: float,
    mahalanobis_distance_score_normalization: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimally match tracks to detections.

    Returns a triple of:
      * ``matched_track_detection_index_pairs`` — list of ``(track_index, detection_index)``,
      * ``unmatched_detection_indices``,
      * ``unmatched_track_indices``.
    """
    track_count = len(predicted_track_bounding_boxes)
    detection_count = len(detection_bounding_boxes)

    if track_count == 0 or detection_count == 0:
        return [], list(range(detection_count)), list(range(track_count))

    pairwise_cost_matrix = np.full(
        (track_count, detection_count), _FORBIDDEN_PAIR_COST, dtype=np.float64
    )

    for track_index in range(track_count):
        predicted_center_x, predicted_center_y = predicted_track_centers[track_index]
        innovation_covariance = track_innovation_covariances[track_index]
        try:
            inverse_innovation_covariance = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            inverse_innovation_covariance = np.eye(2)

        for detection_index in range(detection_count):
            detection_bounding_box = detection_bounding_boxes[detection_index]

            overlap_ratio = intersection_over_union(
                predicted_track_bounding_boxes[track_index], detection_bounding_box
            )
            if overlap_ratio < intersection_over_union_gate_threshold:
                continue

            detection_center_x = detection_bounding_box[0] + detection_bounding_box[2] / 2
            detection_center_y = detection_bounding_box[1] + detection_bounding_box[3] / 2
            center_offset = np.array(
                [
                    detection_center_x - predicted_center_x,
                    detection_center_y - predicted_center_y,
                ]
            )
            mahalanobis_distance = float(
                np.sqrt(center_offset @ inverse_innovation_covariance @ center_offset)
            )
            if mahalanobis_distance >= mahalanobis_distance_gate_threshold:
                continue

            appearance_similarity = compute_appearance_trait_cosine_similarity(
                track_appearance_trait_vectors[track_index],
                detection_appearance_trait_vectors[detection_index],
            )

            combined_match_score = (
                intersection_over_union_score_weight * overlap_ratio
                + mahalanobis_distance_score_weight
                * (1.0 - mahalanobis_distance / mahalanobis_distance_score_normalization)
                + appearance_similarity_score_weight * appearance_similarity
            )
            # The solver minimizes cost, so convert the "higher is better" score
            # into a cost. Higher score => lower cost.
            pairwise_cost_matrix[track_index, detection_index] = -combined_match_score

    matched_track_rows, matched_detection_columns = linear_sum_assignment(
        pairwise_cost_matrix
    )

    matched_track_detection_index_pairs: list[tuple[int, int]] = []
    matched_track_indices: set[int] = set()
    matched_detection_indices: set[int] = set()
    for track_index, detection_index in zip(matched_track_rows, matched_detection_columns):
        if pairwise_cost_matrix[track_index, detection_index] >= _FORBIDDEN_PAIR_COST:
            # The Hungarian solver had to pick a forbidden pair to complete the
            # assignment; treat it as no match.
            continue
        matched_track_detection_index_pairs.append((track_index, detection_index))
        matched_track_indices.add(track_index)
        matched_detection_indices.add(detection_index)

    unmatched_detection_indices = [
        detection_index
        for detection_index in range(detection_count)
        if detection_index not in matched_detection_indices
    ]
    unmatched_track_indices = [
        track_index
        for track_index in range(track_count)
        if track_index not in matched_track_indices
    ]
    return (
        matched_track_detection_index_pairs,
        unmatched_detection_indices,
        unmatched_track_indices,
    )
