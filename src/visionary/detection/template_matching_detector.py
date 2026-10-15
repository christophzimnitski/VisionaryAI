"""Template-matching detector.

Locates instances of a known grayscale template image inside each frame using
normalized cross-correlation, then removes overlapping hits with non-maximum
suppression. Useful when the target object's appearance is known in advance; it
is an alternative to :class:`MotionDifferenceDetector`.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import Detection
from visionary.geometry import BoundingBox, non_maximum_suppression


class TemplateMatchingDetector:
    """Detects a known template via ``cv2.matchTemplate`` plus NMS."""

    def __init__(
        self, template_image_path: str, configuration: TrackerConfiguration
    ) -> None:
        self._configuration = configuration
        self._template_grayscale_image = cv2.imread(
            template_image_path, cv2.IMREAD_GRAYSCALE
        )
        if self._template_grayscale_image is None:
            raise FileNotFoundError(
                f"Template image not found at {template_image_path}"
            )
        self._template_width, self._template_height = (
            self._template_grayscale_image.shape[::-1]
        )

    def detect_objects_in_frame(
        self, frame: np.ndarray, region_of_interest: BoundingBox | None = None
    ) -> list[Detection]:
        target_image = frame
        horizontal_offset, vertical_offset = 0, 0

        if region_of_interest is not None:
            region_left_x, region_top_y, region_width, region_height = region_of_interest
            region_left_x = max(0, region_left_x)
            region_top_y = max(0, region_top_y)
            region_width = min(frame.shape[1] - region_left_x, region_width)
            region_height = min(frame.shape[0] - region_top_y, region_height)
            if region_width <= 0 or region_height <= 0:
                return []
            target_image = frame[
                region_top_y : region_top_y + region_height,
                region_left_x : region_left_x + region_width,
            ]
            horizontal_offset, vertical_offset = region_left_x, region_top_y

        if (
            target_image.shape[0] < self._template_height
            or target_image.shape[1] < self._template_width
        ):
            return []

        target_grayscale_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        correlation_response_map = cv2.matchTemplate(
            target_grayscale_image, self._template_grayscale_image, cv2.TM_CCOEFF_NORMED
        )
        matching_locations = np.where(
            correlation_response_map >= self._configuration.template_match_confidence_threshold
        )

        raw_candidate_boxes = [
            [top_left_x, top_left_y, self._template_width, self._template_height]
            for top_left_x, top_left_y in zip(*matching_locations[::-1], strict=True)
        ]
        if not raw_candidate_boxes:
            return []

        suppressed_boxes = non_maximum_suppression(
            np.array(raw_candidate_boxes),
            self._configuration.non_maximum_suppression_overlap_threshold,
        )

        detections: list[Detection] = []
        for left_x, top_y, width, height in suppressed_boxes:
            detections.append(
                Detection(
                    bounding_box=(
                        int(left_x + horizontal_offset),
                        int(top_y + vertical_offset),
                        int(width),
                        int(height),
                    )
                )
            )
        return detections
