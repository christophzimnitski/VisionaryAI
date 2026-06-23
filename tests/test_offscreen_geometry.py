"""Unit tests for the off-screen expectation geometry (the arrow + bearing)."""

from __future__ import annotations

from visionary.geometry import compute_off_screen_expectation

_FRAME_WIDTH = 640
_FRAME_HEIGHT = 480


def test_a_point_inside_the_frame_has_no_off_screen_expectation() -> None:
    assert compute_off_screen_expectation(320, 240, _FRAME_WIDTH, _FRAME_HEIGHT) is None


def test_bearing_points_up_for_an_object_above_the_frame() -> None:
    expectation = compute_off_screen_expectation(320, -80, _FRAME_WIDTH, _FRAME_HEIGHT)
    assert expectation is not None
    assert abs(expectation.bearing_degrees_clockwise_from_up - 0.0) < 5.0
    assert expectation.edge_indicator_y <= 1


def test_bearing_points_right_for_an_object_past_the_right_edge() -> None:
    expectation = compute_off_screen_expectation(760, 240, _FRAME_WIDTH, _FRAME_HEIGHT)
    assert expectation is not None
    assert abs(expectation.bearing_degrees_clockwise_from_up - 90.0) < 5.0
    assert expectation.edge_indicator_x >= _FRAME_WIDTH - 1


def test_bearing_points_down_for_an_object_below_the_frame() -> None:
    expectation = compute_off_screen_expectation(320, 560, _FRAME_WIDTH, _FRAME_HEIGHT)
    assert expectation is not None
    assert abs(expectation.bearing_degrees_clockwise_from_up - 180.0) < 5.0


def test_bearing_points_left_for_an_object_past_the_left_edge() -> None:
    expectation = compute_off_screen_expectation(-80, 240, _FRAME_WIDTH, _FRAME_HEIGHT)
    assert expectation is not None
    assert abs(expectation.bearing_degrees_clockwise_from_up - 270.0) < 5.0
    assert expectation.edge_indicator_x <= 1


def test_edge_indicator_stays_on_the_frame_border() -> None:
    expectation = compute_off_screen_expectation(900, -200, _FRAME_WIDTH, _FRAME_HEIGHT)
    assert expectation is not None
    assert 0 <= expectation.edge_indicator_x <= _FRAME_WIDTH
    assert 0 <= expectation.edge_indicator_y <= _FRAME_HEIGHT
