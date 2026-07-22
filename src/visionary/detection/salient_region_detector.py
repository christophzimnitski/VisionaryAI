"""Detector that proposes big, distinct resting objects -- no motion required.

A panning camera over a still room sees no motion, so the motion-difference
detector finds nothing and there is nothing to anchor. This detector takes the
other route to "objectness": it proposes regions that stand out from the room's
dominant surfaces (the wall, the floor) by colour and are large enough to be a
real object -- "laptop-sized or bigger". It deliberately does *not* try to label
what the object is; it only says "there is a distinct big thing here", which is
all the rest of the pipeline needs to anchor it and recognise it later.

The method is intentionally simple and cheap: smooth the frame so fine texture
washes out, quantize it to a few colours, treat the colours that cover large
parts of the frame as background, and emit a detection for every sufficiently
large connected blob of a non-background colour. This is the honestly fragile
part on real footage -- gradients, shadows and clutter all fight it -- so its
gates live in the configuration to be tuned per scene.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import Detection
from visionary.geometry import intersection_over_union

# Two proposals this similar (by mutual IoU) are treated as the same object and
# de-duplicated. The threshold is high on purpose so that a *contained* object --
# a pillow lying on a much larger blanket -- is kept rather than swallowed, since
# objects resting on bigger objects are exactly what this scene is full of.
_NEAR_DUPLICATE_OVERLAP_THRESHOLD = 0.6

# Median-blur window used to wash out fine wall/floor texture before quantizing,
# so low-contrast texture collapses into the background colours while large
# objects survive. Odd by requirement of ``cv2.medianBlur``.
_TEXTURE_SMOOTHING_KERNEL_SIZE = 9


class SalientRegionDetector:
    """Proposes large, colour-distinct regions as candidate resting objects."""

    def __init__(self, configuration: TrackerConfiguration) -> None:
        self._configuration = configuration

    def detect_objects_in_frame(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            return []
        frame_height, frame_width = frame.shape[:2]
        total_pixel_count = float(frame_height * frame_width)

        quantized_color_label_image = self._quantize_into_color_labels(frame)
        background_color_labels = self._identify_background_color_labels(
            quantized_color_label_image, total_pixel_count
        )

        minimum_area = self._configuration.salient_region_minimum_area_in_pixels
        maximum_area = (
            self._configuration.salient_region_maximum_area_fraction_of_frame
            * total_pixel_count
        )
        morphological_kernel = np.ones(
            self._configuration.salient_region_morphological_kernel_size, np.uint8
        )

        candidate_boxes: list[tuple[int, int, int, int]] = []
        distinct_color_labels, label_pixel_counts = np.unique(
            quantized_color_label_image, return_counts=True
        )
        for color_label, label_pixel_count in zip(distinct_color_labels, label_pixel_counts):
            if color_label in background_color_labels:
                continue
            if label_pixel_count < minimum_area:
                # The colour does not cover enough of the frame to hold even one
                # big object, so it cannot yield a valid proposal.
                continue
            candidate_boxes.extend(
                self._extract_large_blobs_of_one_color(
                    quantized_color_label_image,
                    color_label,
                    morphological_kernel,
                    minimum_area,
                    maximum_area,
                )
            )

        if not candidate_boxes:
            return []
        deduplicated_boxes = self._remove_near_duplicate_boxes(candidate_boxes)
        return [
            Detection(bounding_box=(int(left_x), int(top_y), int(width), int(height)))
            for left_x, top_y, width, height in deduplicated_boxes
        ]

    @staticmethod
    def _remove_near_duplicate_boxes(
        candidate_boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Drop proposals that nearly coincide, keeping larger and nested boxes.

        Considering boxes largest-first and rejecting only those that strongly
        overlap an already-kept box removes the duplicate a coloured border can
        cast around an object, while a small object contained in a big one (low
        mutual IoU) survives.
        """
        boxes_largest_first = sorted(
            candidate_boxes, key=lambda box: box[2] * box[3], reverse=True
        )
        kept_boxes: list[tuple[int, int, int, int]] = []
        for candidate_box in boxes_largest_first:
            if any(
                intersection_over_union(candidate_box, kept_box)
                > _NEAR_DUPLICATE_OVERLAP_THRESHOLD
                for kept_box in kept_boxes
            ):
                continue
            kept_boxes.append(candidate_box)
        return kept_boxes

    def _quantize_into_color_labels(self, frame: np.ndarray) -> np.ndarray:
        """Smooth then quantize the frame into a single integer colour-label image."""
        smoothed_frame = cv2.medianBlur(frame, _TEXTURE_SMOOTHING_KERNEL_SIZE)
        levels = self._configuration.salient_region_color_quantization_levels
        quantized_channels = (smoothed_frame.astype(np.int32) * levels) // 256
        return (
            quantized_channels[:, :, 0] * levels * levels
            + quantized_channels[:, :, 1] * levels
            + quantized_channels[:, :, 2]
        )

    def _identify_background_color_labels(
        self, quantized_color_label_image: np.ndarray, total_pixel_count: float
    ) -> set[int]:
        """Colours covering a large share of the frame are the room's surfaces."""
        distinct_color_labels, label_pixel_counts = np.unique(
            quantized_color_label_image, return_counts=True
        )
        share_threshold = self._configuration.salient_region_background_color_share_threshold
        return {
            int(color_label)
            for color_label, label_pixel_count in zip(distinct_color_labels, label_pixel_counts)
            if label_pixel_count / total_pixel_count >= share_threshold
        }

    @staticmethod
    def _extract_large_blobs_of_one_color(
        quantized_color_label_image: np.ndarray,
        color_label: int,
        morphological_kernel: np.ndarray,
        minimum_area: float,
        maximum_area: float,
    ) -> list[tuple[int, int, int, int]]:
        """Return size-gated bounding boxes of every blob of one colour label."""
        color_mask = np.where(
            quantized_color_label_image == color_label, np.uint8(255), np.uint8(0)
        )
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, morphological_kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, morphological_kernel)

        component_count, _labels, component_stats, _centroids = (
            cv2.connectedComponentsWithStats(color_mask, connectivity=8)
        )
        boxes: list[tuple[int, int, int, int]] = []
        # Component 0 is the background of this mask; skip it.
        for component_index in range(1, component_count):
            component_area = component_stats[component_index, cv2.CC_STAT_AREA]
            if not (minimum_area <= component_area <= maximum_area):
                continue
            left_x = int(component_stats[component_index, cv2.CC_STAT_LEFT])
            top_y = int(component_stats[component_index, cv2.CC_STAT_TOP])
            width = int(component_stats[component_index, cv2.CC_STAT_WIDTH])
            height = int(component_stats[component_index, cv2.CC_STAT_HEIGHT])
            boxes.append((left_x, top_y, width, height))
        return boxes
