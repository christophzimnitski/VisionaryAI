"""Pure geometric helpers shared across the pipeline.

These functions deliberately contain no OpenCV GUI calls and no global state so
they can be unit-tested exhaustively. Bounding boxes are represented throughout
the project as a ``(top_left_x, top_left_y, width, height)`` tuple in pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class OffScreenExpectation:
    """Where an out-of-view object is expected to be, relative to the frame.

    This is the "point an arrow at the painting behind you" output. It is frame-
    agnostic on purpose: the object's expected centre may come from object motion
    (a rocket that flew off the edge) or, later, from camera motion (a static
    painting the camera panned away from) -- the geometry is identical either way.
    """

    expected_center_x: float
    expected_center_y: float
    direction_unit_x: float
    direction_unit_y: float
    # Compass-style bearing measured clockwise from straight up (0 = up, 90 =
    # right, 180 = down, 270 = left), which is how a person would read the arrow.
    bearing_degrees_clockwise_from_up: float
    # A point clamped to the frame border, on the ray from the frame centre toward
    # the expected centre -- where the on-screen arrow should be drawn.
    edge_indicator_x: int
    edge_indicator_y: int


def compute_off_screen_expectation(
    expected_center_x: float,
    expected_center_y: float,
    frame_width_in_pixels: int,
    frame_height_in_pixels: int,
) -> OffScreenExpectation | None:
    """Return where an off-frame expected centre points, or ``None`` if on-frame.

    Given where the tracker expects an object's centre to be -- possibly outside
    the frame -- return the direction and compass bearing from the frame centre to
    it, plus the border point at which to draw the indicator arrow.
    """
    is_inside_frame = (
        0.0 <= expected_center_x <= frame_width_in_pixels
        and 0.0 <= expected_center_y <= frame_height_in_pixels
    )
    if is_inside_frame:
        return None

    frame_center_x = frame_width_in_pixels / 2.0
    frame_center_y = frame_height_in_pixels / 2.0
    offset_x = expected_center_x - frame_center_x
    offset_y = expected_center_y - frame_center_y
    offset_magnitude = math.hypot(offset_x, offset_y)
    if offset_magnitude < 1e-9:
        return None
    direction_unit_x = offset_x / offset_magnitude
    direction_unit_y = offset_y / offset_magnitude

    # Bearing clockwise from up: up is -y, so atan2(dx, -dy) puts 0 at the top.
    bearing_degrees = math.degrees(math.atan2(offset_x, -offset_y)) % 360.0

    edge_indicator_x, edge_indicator_y = _intersect_ray_with_frame_border(
        frame_center_x,
        frame_center_y,
        direction_unit_x,
        direction_unit_y,
        frame_width_in_pixels,
        frame_height_in_pixels,
    )
    return OffScreenExpectation(
        expected_center_x=expected_center_x,
        expected_center_y=expected_center_y,
        direction_unit_x=direction_unit_x,
        direction_unit_y=direction_unit_y,
        bearing_degrees_clockwise_from_up=bearing_degrees,
        edge_indicator_x=int(round(edge_indicator_x)),
        edge_indicator_y=int(round(edge_indicator_y)),
    )


def _intersect_ray_with_frame_border(
    origin_x: float,
    origin_y: float,
    direction_unit_x: float,
    direction_unit_y: float,
    frame_width_in_pixels: int,
    frame_height_in_pixels: int,
) -> tuple[float, float]:
    """Find where a ray from the frame centre first crosses the frame border."""
    smallest_positive_distance = None
    for border_numerator, border_denominator in (
        (0.0 - origin_x, direction_unit_x),
        (frame_width_in_pixels - origin_x, direction_unit_x),
        (0.0 - origin_y, direction_unit_y),
        (frame_height_in_pixels - origin_y, direction_unit_y),
    ):
        if abs(border_denominator) < 1e-9:
            continue
        distance_to_border = border_numerator / border_denominator
        if distance_to_border <= 0:
            continue
        hit_x = origin_x + distance_to_border * direction_unit_x
        hit_y = origin_y + distance_to_border * direction_unit_y
        hit_is_on_the_frame_border = (
            -1.0 <= hit_x <= frame_width_in_pixels + 1.0
            and -1.0 <= hit_y <= frame_height_in_pixels + 1.0
        )
        if hit_is_on_the_frame_border and (
            smallest_positive_distance is None
            or distance_to_border < smallest_positive_distance
        ):
            smallest_positive_distance = distance_to_border

    if smallest_positive_distance is None:
        return origin_x, origin_y
    border_x = origin_x + smallest_positive_distance * direction_unit_x
    border_y = origin_y + smallest_positive_distance * direction_unit_y
    border_x = min(max(border_x, 0.0), float(frame_width_in_pixels))
    border_y = min(max(border_y, 0.0), float(frame_height_in_pixels))
    return border_x, border_y


def intersection_over_union(
    first_bounding_box: BoundingBox, second_bounding_box: BoundingBox
) -> float:
    """Return the Intersection-over-Union overlap ratio of two bounding boxes.

    The result is ``0.0`` for disjoint boxes and ``1.0`` for identical boxes.
    """
    first_left_x, first_top_y, first_width, first_height = first_bounding_box
    second_left_x, second_top_y, second_width, second_height = second_bounding_box

    intersection_left_x = max(first_left_x, second_left_x)
    intersection_top_y = max(first_top_y, second_top_y)
    intersection_right_x = min(first_left_x + first_width, second_left_x + second_width)
    intersection_bottom_y = min(first_top_y + first_height, second_top_y + second_height)

    intersection_width = max(0, intersection_right_x - intersection_left_x)
    intersection_height = max(0, intersection_bottom_y - intersection_top_y)
    intersection_area = intersection_width * intersection_height

    first_box_area = first_width * first_height
    second_box_area = second_width * second_height
    union_area = first_box_area + second_box_area - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / float(union_area)


def non_maximum_suppression(
    candidate_bounding_boxes: np.ndarray, overlap_threshold: float
) -> np.ndarray:
    """Greedily suppress overlapping detections, keeping the strongest boxes.

    ``candidate_bounding_boxes`` is an ``(N, 4)`` array in
    ``(top_left_x, top_left_y, width, height)`` form. Returns the kept boxes in
    the same form.
    """
    if len(candidate_bounding_boxes) == 0:
        return np.empty((0, 4), dtype=int)

    boxes_as_float = candidate_bounding_boxes.astype("float")
    left_x = boxes_as_float[:, 0]
    top_y = boxes_as_float[:, 1]
    right_x = boxes_as_float[:, 0] + boxes_as_float[:, 2]
    bottom_y = boxes_as_float[:, 1] + boxes_as_float[:, 3]

    box_areas = (right_x - left_x + 1) * (bottom_y - top_y + 1)
    indices_sorted_by_bottom_edge = np.argsort(bottom_y)

    kept_box_indices: list[int] = []
    while len(indices_sorted_by_bottom_edge) > 0:
        last_position = len(indices_sorted_by_bottom_edge) - 1
        current_index = indices_sorted_by_bottom_edge[last_position]
        kept_box_indices.append(current_index)

        remaining_indices = indices_sorted_by_bottom_edge[:last_position]
        overlap_left_x = np.maximum(left_x[current_index], left_x[remaining_indices])
        overlap_top_y = np.maximum(top_y[current_index], top_y[remaining_indices])
        overlap_right_x = np.minimum(right_x[current_index], right_x[remaining_indices])
        overlap_bottom_y = np.minimum(bottom_y[current_index], bottom_y[remaining_indices])

        overlap_width = np.maximum(0, overlap_right_x - overlap_left_x + 1)
        overlap_height = np.maximum(0, overlap_bottom_y - overlap_top_y + 1)
        overlap_ratio = (overlap_width * overlap_height) / box_areas[remaining_indices]

        indices_sorted_by_bottom_edge = np.delete(
            indices_sorted_by_bottom_edge,
            np.concatenate(([last_position], np.where(overlap_ratio > overlap_threshold)[0])),
        )

    return candidate_bounding_boxes[kept_box_indices].astype("int")
