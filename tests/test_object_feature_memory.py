"""Unit tests for the intrinsic object feature memory (active-memory recognition)."""

from __future__ import annotations

import numpy as np

from visionary.features.object_feature_memory import (
    ObjectFeatureMemory,
    ObservedObjectFeatures,
)


def _features(mean_color_bgr: list[float]) -> ObservedObjectFeatures:
    return ObservedObjectFeatures(
        mean_color_bgr=np.array(mean_color_bgr, dtype=np.float32),
        normalized_size=0.01,
        aspect_ratio=1.0,
    )


def test_recognition_is_high_for_the_same_colour_at_a_different_brightness() -> None:
    memory = ObjectFeatureMemory()
    memory.update_with_observation(_features([200.0, 80.0, 60.0]))
    # The same hue, dimmer because more dark background was in the box.
    dimmer_same_hue = _features([100.0, 40.0, 30.0])
    assert memory.recognition_similarity_to(dimmer_same_hue) > 0.8


def test_recognition_is_low_for_a_clearly_different_colour() -> None:
    memory = ObjectFeatureMemory()
    memory.update_with_observation(_features([200.0, 80.0, 60.0]))  # blue-ish
    orange_object = _features([60.0, 140.0, 200.0])
    assert memory.recognition_similarity_to(orange_object) < 0.5


def test_an_unobserved_memory_matches_nothing() -> None:
    memory = ObjectFeatureMemory()
    assert not memory.has_been_observed
    assert memory.recognition_similarity_to(_features([200.0, 80.0, 60.0])) == 0.0
