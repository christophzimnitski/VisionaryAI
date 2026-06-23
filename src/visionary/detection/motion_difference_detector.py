"""Motion-difference detector.

Detects moving objects by thresholding the absolute difference between the
current frame and the previous frame, then extracting bounding boxes around the
resulting motion blobs. This is the project's canonical default detector because
it requires no per-object template and works on arbitrary moving objects.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import Detection


class MotionDifferenceDetector:
    """Frame-differencing detector that emits one detection per motion blob."""

    def __init__(self, configuration: TrackerConfiguration) -> None:
        self._configuration = configuration
        self._previous_blurred_grayscale_frame: np.ndarray | None = None

    def _convert_to_blurred_grayscale(self, frame: np.ndarray) -> np.ndarray:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(
            grayscale_frame, self._configuration.gaussian_blur_kernel_size, 0
        )

    def detect_objects_in_frame(self, frame: np.ndarray) -> list[Detection]:
        current_blurred_grayscale_frame = self._convert_to_blurred_grayscale(frame)

        if self._previous_blurred_grayscale_frame is None:
            self._previous_blurred_grayscale_frame = current_blurred_grayscale_frame
            return []

        absolute_frame_difference = cv2.absdiff(
            self._previous_blurred_grayscale_frame, current_blurred_grayscale_frame
        )
        _, binarized_motion_mask = cv2.threshold(
            absolute_frame_difference,
            self._configuration.motion_difference_binarization_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        dilated_motion_mask = cv2.dilate(
            binarized_motion_mask,
            np.ones(self._configuration.motion_dilation_kernel_size, np.uint8),
            iterations=self._configuration.motion_dilation_iteration_count,
        )
        motion_contours, _ = cv2.findContours(
            dilated_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[Detection] = []
        for motion_contour in motion_contours:
            motion_contour_area = cv2.contourArea(motion_contour)
            is_within_expected_object_size = (
                self._configuration.minimum_motion_contour_area_in_pixels
                <= motion_contour_area
                <= self._configuration.maximum_motion_contour_area_in_pixels
            )
            if not is_within_expected_object_size:
                continue
            bounding_box = cv2.boundingRect(motion_contour)
            detections.append(Detection(bounding_box=bounding_box))

        self._previous_blurred_grayscale_frame = current_blurred_grayscale_frame
        return detections
