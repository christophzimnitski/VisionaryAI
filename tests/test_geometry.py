"""Unit tests for the pure geometric helpers."""

from __future__ import annotations

import numpy as np

from visionary.geometry import intersection_over_union, non_maximum_suppression


def test_identical_boxes_have_intersection_over_union_of_one() -> None:
    bounding_box = (10, 20, 30, 40)
    assert intersection_over_union(bounding_box, bounding_box) == 1.0


def test_disjoint_boxes_have_zero_intersection_over_union() -> None:
    left_box = (0, 0, 10, 10)
    far_away_box = (100, 100, 10, 10)
    assert intersection_over_union(left_box, far_away_box) == 0.0


def test_half_overlapping_boxes_have_expected_intersection_over_union() -> None:
    first_box = (0, 0, 10, 10)
    horizontally_shifted_box = (5, 0, 10, 10)
    # Intersection area = 5 * 10 = 50; union area = 100 + 100 - 50 = 150.
    assert intersection_over_union(first_box, horizontally_shifted_box) == 50 / 150


def test_intersection_over_union_is_symmetric() -> None:
    first_box = (3, 7, 12, 9)
    second_box = (8, 10, 6, 14)
    assert intersection_over_union(first_box, second_box) == intersection_over_union(
        second_box, first_box
    )


def test_zero_area_box_does_not_divide_by_zero() -> None:
    degenerate_box = (5, 5, 0, 0)
    other_box = (5, 5, 10, 10)
    assert intersection_over_union(degenerate_box, other_box) == 0.0


def test_non_maximum_suppression_collapses_overlapping_cluster() -> None:
    overlapping_boxes = np.array(
        [[10, 10, 20, 20], [12, 12, 20, 20], [11, 11, 20, 20]]
    )
    kept_boxes = non_maximum_suppression(overlapping_boxes, overlap_threshold=0.3)
    assert len(kept_boxes) == 1


def test_non_maximum_suppression_keeps_disjoint_boxes() -> None:
    disjoint_boxes = np.array([[0, 0, 10, 10], [100, 100, 10, 10], [200, 200, 10, 10]])
    kept_boxes = non_maximum_suppression(disjoint_boxes, overlap_threshold=0.3)
    assert len(kept_boxes) == 3


def test_non_maximum_suppression_on_empty_input_returns_empty() -> None:
    kept_boxes = non_maximum_suppression(np.empty((0, 4)), overlap_threshold=0.3)
    assert len(kept_boxes) == 0
