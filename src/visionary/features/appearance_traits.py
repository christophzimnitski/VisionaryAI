"""Hand-crafted appearance descriptors used for re-identification.

For each detection we compute a small fixed-length "trait vector" summarizing its
colour, shape and texture. Trait vectors let the tracker (a) score how similar a
detection looks to an existing track and (b) assign a stable, human-readable
"object type token" so that objects which look alike share a label.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from visionary.geometry import BoundingBox

APPEARANCE_TRAIT_VECTOR_LENGTH = 9

# Indices into the trait vector, named so downstream code never uses bare numbers.
_HUE_MEAN_INDEX = 0
_SATURATION_MEAN_INDEX = 1
_VALUE_MEAN_INDEX = 2
_ASPECT_RATIO_INDEX = 3
_EDGE_DENSITY_INDEX = 4
_INTERNAL_FEATURE_COUNT_INDEX = 5
_NORMALIZED_SIZE_INDEX = 8


def extract_appearance_trait_vector(
    frame: np.ndarray, bounding_box: BoundingBox
) -> np.ndarray:
    """Compute the fixed-length appearance descriptor for one bounding box."""
    left_x, top_y, width, height = bounding_box
    image_patch = frame[top_y : top_y + height, left_x : left_x + width]
    if image_patch.size == 0:
        return np.zeros(APPEARANCE_TRAIT_VECTOR_LENGTH, dtype=np.float32)

    hue_saturation_value_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)
    normalized_mean_hue_saturation_value = np.array(
        cv2.mean(hue_saturation_value_patch)[:3]
    ) / [179.0, 255.0, 255.0]

    grayscale_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    detected_edges = cv2.Canny(grayscale_patch, 50, 150)
    edge_density = np.sum(detected_edges > 0) / (width * height + 1e-8)

    aspect_ratio = width / max(height, 1.0)

    _, bright_region_mask = cv2.threshold(grayscale_patch, 180, 255, cv2.THRESH_BINARY)
    bright_region_contours, _ = cv2.findContours(
        bright_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    internal_feature_count = len(
        [contour for contour in bright_region_contours if cv2.contourArea(contour) > 5]
    )

    frame_height, frame_width = frame.shape[:2]
    normalized_size = (width * height) / (frame_width * frame_height)

    appearance_trait_vector = np.zeros(APPEARANCE_TRAIT_VECTOR_LENGTH, dtype=np.float32)
    appearance_trait_vector[_HUE_MEAN_INDEX] = normalized_mean_hue_saturation_value[0]
    appearance_trait_vector[_SATURATION_MEAN_INDEX] = normalized_mean_hue_saturation_value[1]
    appearance_trait_vector[_VALUE_MEAN_INDEX] = normalized_mean_hue_saturation_value[2]
    appearance_trait_vector[_ASPECT_RATIO_INDEX] = aspect_ratio
    appearance_trait_vector[_EDGE_DENSITY_INDEX] = edge_density
    appearance_trait_vector[_INTERNAL_FEATURE_COUNT_INDEX] = float(internal_feature_count)
    appearance_trait_vector[_NORMALIZED_SIZE_INDEX] = normalized_size
    return appearance_trait_vector


def assign_object_type_token(appearance_trait_vector: np.ndarray | None) -> str:
    """Map a trait vector to a stable token; look-alike objects share a token.

    The trait vector is coarsely quantized into wide bins so that small frame-to-
    frame variations do not produce a new token, then hashed into a short label.
    """
    if appearance_trait_vector is None:
        return "OBJ-0000"

    hue_bin = int(appearance_trait_vector[_HUE_MEAN_INDEX] * 6) % 6
    saturation_bin = int(appearance_trait_vector[_SATURATION_MEAN_INDEX] * 3) % 3
    value_bin = int(appearance_trait_vector[_VALUE_MEAN_INDEX] * 3) % 3
    shape_bin = (
        1
        if (
            appearance_trait_vector[_ASPECT_RATIO_INDEX] > 1.35
            or appearance_trait_vector[_EDGE_DENSITY_INDEX] > 0.19
        )
        else 0
    )
    internal_feature_bin = min(int(appearance_trait_vector[_INTERNAL_FEATURE_COUNT_INDEX]), 3)
    normalized_size = appearance_trait_vector[_NORMALIZED_SIZE_INDEX]
    if normalized_size > 0.017:
        size_bin = 2
    elif normalized_size < 0.0028:
        size_bin = 0
    else:
        size_bin = 1

    quantized_trait_key = (
        hue_bin,
        saturation_bin,
        value_bin,
        shape_bin,
        internal_feature_bin,
        size_bin,
    )
    type_token = hashlib.md5(str(quantized_trait_key).encode()).hexdigest()[:4].upper()
    return f"OBJ-{type_token}"


def compute_appearance_trait_cosine_similarity(
    first_trait_vector: np.ndarray | None, second_trait_vector: np.ndarray | None
) -> float:
    """Return cosine similarity in ``[0, 1]`` between two trait vectors."""
    if first_trait_vector is None or second_trait_vector is None:
        return 0.0
    dot_product = float(np.dot(first_trait_vector, second_trait_vector))
    first_vector_magnitude = float(np.linalg.norm(first_trait_vector))
    second_vector_magnitude = float(np.linalg.norm(second_trait_vector))
    return dot_product / (first_vector_magnitude * second_vector_magnitude + 1e-8)
