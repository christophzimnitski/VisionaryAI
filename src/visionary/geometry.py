"""Pure geometric helpers shared across the pipeline.

These functions deliberately contain no OpenCV GUI calls and no global state so
they can be unit-tested exhaustively. Bounding boxes are represented throughout
the project as a ``(top_left_x, top_left_y, width, height)`` tuple in pixels.
"""

from __future__ import annotations

import numpy as np

BoundingBox = tuple[int, int, int, int]


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
