"""Unit tests for the constant-velocity Kalman filter wrapper."""

from __future__ import annotations

from visionary.tracking.kalman_filter import ConstantVelocityKalmanFilter


def test_prediction_converges_to_constant_velocity_motion() -> None:
    """Feeding a steadily moving centre should make the prediction error shrink."""
    kalman_filter = ConstantVelocityKalmanFilter(initial_center_x=0.0, initial_center_y=0.0)

    constant_velocity_per_frame = 5.0
    true_center_x = 0.0
    final_prediction_error = None

    for frame_number in range(1, 31):
        predicted_center_x, _, _ = kalman_filter.predict_next_center()
        true_center_x = frame_number * constant_velocity_per_frame
        final_prediction_error = abs(predicted_center_x - true_center_x)
        kalman_filter.correct_with_measured_center(true_center_x, 0.0)

    assert final_prediction_error is not None
    # After warm-up the one-step-ahead prediction should track within a few pixels.
    assert final_prediction_error < 2.0


def test_innovation_covariance_is_two_by_two() -> None:
    kalman_filter = ConstantVelocityKalmanFilter(0.0, 0.0)
    _, _, innovation_covariance = kalman_filter.predict_next_center()
    assert innovation_covariance.shape == (2, 2)


def test_set_current_center_overrides_estimate() -> None:
    kalman_filter = ConstantVelocityKalmanFilter(0.0, 0.0)
    kalman_filter.set_current_center(123.0, 456.0)
    center_x, center_y = kalman_filter.get_current_center()
    assert (center_x, center_y) == (123.0, 456.0)
