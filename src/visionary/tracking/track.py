"""A single tracked object and its lifecycle.

A :class:`TrackedObject` owns a Kalman filter for motion prediction, an appearance
trait vector for re-identification, and the bookkeeping needed to move through the
lifecycle states: ACTIVE (recently matched) and LOST (temporarily unmatched but
still recoverable).
"""

from __future__ import annotations

import uuid
from collections import Counter, deque
from enum import Enum

import numpy as np

from visionary.features.appearance_traits import (
    APPEARANCE_TRAIT_VECTOR_LENGTH,
    assign_object_type_token,
)
from visionary.geometry import BoundingBox
from visionary.tracking.kalman_filter import ConstantVelocityKalmanFilter

_TYPE_TOKEN_HISTORY_LENGTH = 5
_BOUNDING_BOX_HISTORY_LENGTH = 10


class TrackLifecycleState(str, Enum):
    """The two lifecycle states a track can be in."""

    ACTIVE = "active"
    LOST = "lost"


class TrackedObject:
    """One persistent object identity across frames."""

    def __init__(
        self,
        initial_bounding_box: BoundingBox,
        current_frame_index: int,
        appearance_trait_vector: np.ndarray | None = None,
    ) -> None:
        self.track_id = str(uuid.uuid4())[:8]
        self.bounding_box_width = initial_bounding_box[2]
        self.bounding_box_height = initial_bounding_box[3]

        if appearance_trait_vector is None:
            appearance_trait_vector = np.zeros(
                APPEARANCE_TRAIT_VECTOR_LENGTH, dtype=np.float32
            )
        self.appearance_trait_vector = appearance_trait_vector

        initial_type_token = assign_object_type_token(appearance_trait_vector)
        self.object_type_token = initial_type_token
        self._recent_type_token_history: deque[str] = deque(
            [initial_type_token], maxlen=_TYPE_TOKEN_HISTORY_LENGTH
        )

        self.recent_bounding_box_history: deque[BoundingBox] = deque(
            [initial_bounding_box], maxlen=_BOUNDING_BOX_HISTORY_LENGTH
        )
        self.consecutive_missed_frame_count = 0
        self.last_seen_frame_index = current_frame_index
        self.lifecycle_state = TrackLifecycleState.ACTIVE

        center_x = initial_bounding_box[0] + initial_bounding_box[2] / 2
        center_y = initial_bounding_box[1] + initial_bounding_box[3] / 2
        self._kalman_filter = ConstantVelocityKalmanFilter(center_x, center_y)

    def predict_bounding_box(self) -> tuple[BoundingBox, float, float, np.ndarray]:
        """Advance the Kalman filter and return the predicted bounding box, the
        predicted centre, and the innovation covariance for gating."""
        predicted_center_x, predicted_center_y, innovation_covariance = (
            self._kalman_filter.predict_next_center()
        )
        predicted_left_x = int(predicted_center_x - self.bounding_box_width / 2)
        predicted_top_y = int(predicted_center_y - self.bounding_box_height / 2)
        predicted_bounding_box = (
            predicted_left_x,
            predicted_top_y,
            self.bounding_box_width,
            self.bounding_box_height,
        )
        return (
            predicted_bounding_box,
            predicted_center_x,
            predicted_center_y,
            innovation_covariance,
        )

    def update_with_matched_detection(
        self,
        matched_bounding_box: BoundingBox,
        current_frame_index: int,
        matched_appearance_trait_vector: np.ndarray | None = None,
    ) -> None:
        """Correct the filter with a matched detection and refresh lifecycle state."""
        center_x = matched_bounding_box[0] + matched_bounding_box[2] / 2
        center_y = matched_bounding_box[1] + matched_bounding_box[3] / 2
        self._kalman_filter.correct_with_measured_center(center_x, center_y)

        self.recent_bounding_box_history.append(matched_bounding_box)
        if matched_appearance_trait_vector is not None:
            self.appearance_trait_vector = matched_appearance_trait_vector
            self._update_object_type_token(matched_appearance_trait_vector)

        self.consecutive_missed_frame_count = 0
        self.last_seen_frame_index = current_frame_index
        self.lifecycle_state = TrackLifecycleState.ACTIVE

    def _update_object_type_token(self, appearance_trait_vector: np.ndarray) -> None:
        """Stabilize the type token via majority vote over recent observations."""
        most_recent_type_token = assign_object_type_token(appearance_trait_vector)
        self._recent_type_token_history.append(most_recent_type_token)
        self.object_type_token = Counter(self._recent_type_token_history).most_common(1)[0][0]

    def coast_through_missed_frame(self) -> None:
        """Advance the prediction without a measurement and count the miss."""
        self._kalman_filter.coast_without_measurement()
        self.consecutive_missed_frame_count += 1

    def apply_camera_motion_warp(self, homography_matrix: np.ndarray) -> None:
        """Warp the estimated centre by a frame-to-frame homography so that camera
        motion does not look like object motion during association."""
        current_center_x, current_center_y = self._kalman_filter.get_current_center()
        homogeneous_point = np.array(
            [current_center_x, current_center_y, 1.0], dtype=np.float64
        )
        warped_point = homography_matrix @ homogeneous_point
        if abs(warped_point[2]) < 1e-8:
            return
        self._kalman_filter.set_current_center(
            warped_point[0] / warped_point[2], warped_point[1] / warped_point[2]
        )

    @property
    def most_recent_bounding_box(self) -> BoundingBox:
        return self.recent_bounding_box_history[-1]
