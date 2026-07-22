"""A small, incremental memory of an object's *intrinsic* features.

This is the "object file" / active-memory idea, kept deliberately lightweight. An
object is remembered not by storing its whole picture but by a handful of named,
individually-comparable traits -- mean colour, relative size, aspect ratio -- each
updated as a running average so the memory is refined, never recomputed from
scratch.

Recognition is a *partial* match: two observations are compared on the traits they
share, weighted by how discriminating each trait is (colour dominates, because it
separates "the blue rocket" or "the black sedan" from everything else). This is
what lets an object be re-identified when it returns to view even though its
position is no longer where prediction said it would be -- the rocket that flew
off the top, or, later, the painting the camera panned away from.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionary.geometry import BoundingBox

# Colour is compared as *normalized chromaticity* (each channel over the channel
# sum), not raw BGR, so it is invariant to brightness -- a patch that happens to
# include more dark background is dimmer but the same hue, and must still match.
_CHROMATICITY_DISTANCE_AT_ZERO_SIMILARITY = 0.2
_COLOR_SIMILARITY_WEIGHT = 0.6
_SIZE_SIMILARITY_WEIGHT = 0.25
_ASPECT_SIMILARITY_WEIGHT = 0.15


@dataclass(frozen=True)
class ObservedObjectFeatures:
    """A single observation's intrinsic traits, extracted from one image patch."""

    mean_color_bgr: np.ndarray
    normalized_size: float
    aspect_ratio: float


def compare_observed_object_features(
    first_features: ObservedObjectFeatures, second_features: ObservedObjectFeatures
) -> float:
    """Return a ``[0, 1]`` partial-match score between two intrinsic snapshots.

    This is the same brightness-invariant comparison the active memory uses to
    recognise a returning object, exposed as a free function so that two *fresh
    observations* can be compared -- not only a remembered memory against one
    observation. The displacement reasoner needs exactly this: deciding that the
    thing that vanished from one place is the thing that appeared in another is a
    comparison between two observations, neither of which is the running memory.
    """
    chromaticity_distance = float(
        np.linalg.norm(
            _normalized_chromaticity(first_features.mean_color_bgr)
            - _normalized_chromaticity(second_features.mean_color_bgr)
        )
    )
    color_similarity = max(
        0.0, 1.0 - chromaticity_distance / _CHROMATICITY_DISTANCE_AT_ZERO_SIMILARITY
    )
    size_similarity = _ratio_similarity(
        first_features.normalized_size, second_features.normalized_size
    )
    aspect_similarity = _ratio_similarity(
        first_features.aspect_ratio, second_features.aspect_ratio
    )
    return (
        _COLOR_SIMILARITY_WEIGHT * color_similarity
        + _SIZE_SIMILARITY_WEIGHT * size_similarity
        + _ASPECT_SIMILARITY_WEIGHT * aspect_similarity
    )


def describe_observed_object(
    image_patch: np.ndarray, bounding_box: BoundingBox, frame_shape: tuple[int, ...]
) -> ObservedObjectFeatures | None:
    """Extract the intrinsic feature snapshot of one detection, or ``None`` if empty."""
    if image_patch.size == 0:
        return None
    _left_x, _top_y, width, height = bounding_box
    frame_height, frame_width = frame_shape[:2]
    mean_color_bgr = np.array(cv2.mean(image_patch)[:3], dtype=np.float32)
    normalized_size = (width * height) / float(frame_width * frame_height + 1e-9)
    aspect_ratio = width / max(float(height), 1.0)
    return ObservedObjectFeatures(
        mean_color_bgr=mean_color_bgr,
        normalized_size=normalized_size,
        aspect_ratio=aspect_ratio,
    )


@dataclass
class ObjectFeatureMemory:
    """The remembered, slowly-updated intrinsic traits of one tracked object."""

    mean_color_bgr: np.ndarray | None = None
    normalized_size: float | None = None
    aspect_ratio: float | None = None
    observation_count: int = 0

    @property
    def has_been_observed(self) -> bool:
        return self.observation_count > 0

    def update_with_observation(self, observed_features: ObservedObjectFeatures) -> None:
        """Fold a fresh clean observation into the running feature averages."""
        self.observation_count += 1
        blend_toward_new = 1.0 / min(self.observation_count, 20)
        if self.mean_color_bgr is None:
            self.mean_color_bgr = observed_features.mean_color_bgr.astype(np.float32)
            self.normalized_size = observed_features.normalized_size
            self.aspect_ratio = observed_features.aspect_ratio
            return
        self.mean_color_bgr = (
            (1.0 - blend_toward_new) * self.mean_color_bgr
            + blend_toward_new * observed_features.mean_color_bgr
        )
        self.normalized_size = (
            (1.0 - blend_toward_new) * self.normalized_size
            + blend_toward_new * observed_features.normalized_size
        )
        self.aspect_ratio = (
            (1.0 - blend_toward_new) * self.aspect_ratio
            + blend_toward_new * observed_features.aspect_ratio
        )

    def recognition_similarity_to(self, observed_features: ObservedObjectFeatures) -> float:
        """Return a ``[0, 1]`` partial-match score against a fresh observation.

        Only traits that have been remembered contribute, each weighted by how
        discriminating it is. Colour dominates because it is what separates one
        object from another of the same kind.
        """
        if not self.has_been_observed:
            return 0.0
        remembered_features = ObservedObjectFeatures(
            mean_color_bgr=self.mean_color_bgr,
            normalized_size=self.normalized_size,
            aspect_ratio=self.aspect_ratio,
        )
        return compare_observed_object_features(remembered_features, observed_features)


def _normalized_chromaticity(color_bgr: np.ndarray) -> np.ndarray:
    """Return the brightness-independent chromaticity (each channel over the sum)."""
    channel_sum = float(np.sum(color_bgr)) + 1e-6
    return color_bgr / channel_sum


def _ratio_similarity(first_value: float, second_value: float) -> float:
    """Return ``min/max`` of two positive values: 1 when equal, →0 when very different."""
    larger_value = max(abs(first_value), abs(second_value), 1e-9)
    smaller_value = min(abs(first_value), abs(second_value))
    return smaller_value / larger_value
