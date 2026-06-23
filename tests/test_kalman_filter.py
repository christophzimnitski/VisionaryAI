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


def _warm_up_with_constant_rightward_velocity(
    kalman_filter: ConstantVelocityKalmanFilter, velocity_per_frame: float, frame_count: int
) -> None:
    for frame_number in range(1, frame_count + 1):
        kalman_filter.predict_next_center()
        kalman_filter.correct_with_measured_center(velocity_per_frame * frame_number, 0.0)


def test_get_current_velocity_recovers_the_established_motion() -> None:
    kalman_filter = ConstantVelocityKalmanFilter(0.0, 0.0)
    _warm_up_with_constant_rightward_velocity(kalman_filter, velocity_per_frame=5.0, frame_count=25)
    velocity_x, velocity_y = kalman_filter.get_current_velocity()
    assert abs(velocity_x - 5.0) < 1.0
    assert abs(velocity_y) < 1.0


def test_position_preserving_correction_keeps_velocity_through_a_stalled_blob() -> None:
    """A correction at a stalled position must not corrupt the established velocity.

    This is the coalescence defence: while two identical objects overlap the
    measured blob barely moves, and a normal correction would drag velocity to
    zero. The position-preserving correction must leave velocity untouched.
    """
    kalman_filter = ConstantVelocityKalmanFilter(0.0, 0.0)
    _warm_up_with_constant_rightward_velocity(kalman_filter, velocity_per_frame=5.0, frame_count=25)

    velocity_x_before, _ = kalman_filter.get_current_velocity()
    last_position_x, _ = kalman_filter.get_current_center()

    kalman_filter.predict_next_center()
    # Measure the object as essentially stationary (as during a full overlap).
    kalman_filter.correct_position_preserving_velocity(last_position_x, 0.0)

    velocity_x_after, _ = kalman_filter.get_current_velocity()
    corrected_position_x, _ = kalman_filter.get_current_center()
    assert abs(velocity_x_after - velocity_x_before) < 1e-5
    # The position still responds to the measurement (it is pulled back behind the
    # one-step prediction toward the stalled measurement).
    assert corrected_position_x < last_position_x + velocity_x_before


def test_normal_correction_does_corrupt_velocity_through_a_stalled_blob() -> None:
    """Contrast: the ordinary correction *does* lose velocity at a stalled blob."""
    kalman_filter = ConstantVelocityKalmanFilter(0.0, 0.0)
    _warm_up_with_constant_rightward_velocity(kalman_filter, velocity_per_frame=5.0, frame_count=25)

    velocity_x_before, _ = kalman_filter.get_current_velocity()
    last_position_x, _ = kalman_filter.get_current_center()

    kalman_filter.predict_next_center()
    kalman_filter.correct_with_measured_center(last_position_x, 0.0)

    velocity_x_after, _ = kalman_filter.get_current_velocity()
    assert velocity_x_after < velocity_x_before
