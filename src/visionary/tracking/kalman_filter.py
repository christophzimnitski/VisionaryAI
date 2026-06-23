"""Constant-velocity Kalman filter wrapper.

Wraps ``cv2.KalmanFilter`` behind descriptive methods so the rest of the code
never touches raw OpenCV matrices. The filter state is the object centre and its
velocity: ``[center_x, center_y, velocity_x, velocity_y]``; the measurement is the
observed centre ``[center_x, center_y]``.
"""

from __future__ import annotations

import cv2
import numpy as np


class ConstantVelocityKalmanFilter:
    """A 4-state / 2-measurement constant-velocity Kalman filter for a centroid."""

    def __init__(self, initial_center_x: float, initial_center_y: float) -> None:
        self._kalman_filter = cv2.KalmanFilter(4, 2)
        self._kalman_filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self._kalman_filter.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kalman_filter.processNoiseCov = (
            np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]],
                dtype=np.float32,
            )
            * 1.5
        )
        self._kalman_filter.measurementNoiseCov = (
            np.array([[1, 0], [0, 1]], dtype=np.float32) * 8
        )
        self._kalman_filter.errorCovPost = 1.0 * np.eye(4, dtype=np.float32)
        self._kalman_filter.statePost = np.array(
            [[initial_center_x], [initial_center_y], [0.0], [0.0]], dtype=np.float32
        )

    def predict_next_center(self) -> tuple[float, float, np.ndarray]:
        """Advance the filter one step and return the predicted centre and the
        2x2 innovation covariance used for Mahalanobis gating."""
        predicted_state = self._kalman_filter.predict()
        predicted_center_x = float(predicted_state[0, 0])
        predicted_center_y = float(predicted_state[1, 0])
        innovation_covariance = (
            self._kalman_filter.measurementMatrix
            @ self._kalman_filter.errorCovPre
            @ self._kalman_filter.measurementMatrix.T
            + self._kalman_filter.measurementNoiseCov
        )
        return predicted_center_x, predicted_center_y, innovation_covariance

    def correct_with_measured_center(
        self, measured_center_x: float, measured_center_y: float
    ) -> None:
        """Fuse a new centre measurement into the filter state."""
        self._kalman_filter.correct(
            np.array([[measured_center_x], [measured_center_y]], dtype=np.float32)
        )

    def correct_position_preserving_velocity(
        self, measured_center_x: float, measured_center_y: float
    ) -> None:
        """Pull the position estimate toward a measurement while keeping velocity.

        This is the antidote to *track coalescence*: when two identical objects
        overlap, the measured blob centre barely moves, so a normal correction
        would drag the filter's velocity toward zero and the track would forget
        which way its object was heading. By restoring the predicted (pre-update)
        velocity after the correction, the box follows the visible blob but the
        established motion -- the only thing that can later tell the two objects
        apart -- is preserved intact.
        """
        predicted_velocity_x = float(self._kalman_filter.statePre[2, 0])
        predicted_velocity_y = float(self._kalman_filter.statePre[3, 0])
        self._kalman_filter.correct(
            np.array([[measured_center_x], [measured_center_y]], dtype=np.float32)
        )
        self._kalman_filter.statePost[2, 0] = predicted_velocity_x
        self._kalman_filter.statePost[3, 0] = predicted_velocity_y

    def coast_without_measurement(self) -> None:
        """Advance the state without a measurement (used while a track is missed)."""
        self._kalman_filter.statePost = self._kalman_filter.statePre.copy()
        self._kalman_filter.errorCovPost = self._kalman_filter.errorCovPre.copy()

    def get_current_center(self) -> tuple[float, float]:
        """Return the current best estimate of the object centre."""
        return (
            float(self._kalman_filter.statePost[0, 0]),
            float(self._kalman_filter.statePost[1, 0]),
        )

    def get_current_velocity(self) -> tuple[float, float]:
        """Return the current best estimate of the object velocity (pixels/frame)."""
        return (
            float(self._kalman_filter.statePost[2, 0]),
            float(self._kalman_filter.statePost[3, 0]),
        )

    def set_current_center(self, new_center_x: float, new_center_y: float) -> None:
        """Overwrite the current centre estimate (used by camera-motion warping)."""
        self._kalman_filter.statePost[0, 0] = new_center_x
        self._kalman_filter.statePost[1, 0] = new_center_y
