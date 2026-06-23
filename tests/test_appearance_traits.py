"""Unit tests for appearance trait extraction, type tokens and similarity."""

from __future__ import annotations

import numpy as np

from visionary.features.appearance_traits import (
    APPEARANCE_TRAIT_VECTOR_LENGTH,
    assign_object_type_token,
    compute_appearance_trait_cosine_similarity,
    extract_appearance_trait_vector,
)


def _build_solid_color_frame(
    width: int = 100, height: int = 100, blue: int = 0, green: int = 0, red: int = 255
) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (blue, green, red)
    return frame


def test_trait_vector_has_expected_length() -> None:
    frame = _build_solid_color_frame()
    trait_vector = extract_appearance_trait_vector(frame, (10, 10, 30, 30))
    assert trait_vector.shape == (APPEARANCE_TRAIT_VECTOR_LENGTH,)


def test_empty_patch_returns_zero_trait_vector() -> None:
    frame = _build_solid_color_frame()
    trait_vector = extract_appearance_trait_vector(frame, (10, 10, 0, 0))
    assert np.count_nonzero(trait_vector) == 0


def test_identical_patches_receive_the_same_type_token() -> None:
    frame = _build_solid_color_frame()
    first_trait_vector = extract_appearance_trait_vector(frame, (10, 10, 30, 30))
    second_trait_vector = extract_appearance_trait_vector(frame, (40, 40, 30, 30))
    assert assign_object_type_token(first_trait_vector) == assign_object_type_token(
        second_trait_vector
    )


def test_none_trait_vector_yields_default_token() -> None:
    assert assign_object_type_token(None) == "OBJ-0000"


def test_cosine_similarity_of_a_vector_with_itself_is_one() -> None:
    trait_vector = np.array([0.1, 0.2, 0.3, 1.0, 0.5, 2.0, 0.0, 0.0, 0.01], dtype=np.float32)
    similarity = compute_appearance_trait_cosine_similarity(trait_vector, trait_vector)
    assert abs(similarity - 1.0) < 1e-5


def test_cosine_similarity_with_none_is_zero() -> None:
    trait_vector = np.ones(APPEARANCE_TRAIT_VECTOR_LENGTH, dtype=np.float32)
    assert compute_appearance_trait_cosine_similarity(trait_vector, None) == 0.0
