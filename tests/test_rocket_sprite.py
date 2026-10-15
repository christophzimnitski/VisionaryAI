"""Unit tests for loading and normalizing the synthetic scene's rocket sprite."""

from __future__ import annotations

import numpy as np
import pytest

from visionary.io.video_sources import (
    SyntheticRocketSceneSource,
    load_bundled_rocket_sprite_image,
    normalize_rocket_sprite_image,
)

_EXPECTED_LONGEST_SIDE = 60


def _assert_is_usable_sprite(sprite_image: np.ndarray | None) -> None:
    assert sprite_image is not None
    assert sprite_image.dtype == np.uint8
    assert sprite_image.ndim == 3 and sprite_image.shape[2] == 3
    assert max(sprite_image.shape[:2]) == _EXPECTED_LONGEST_SIDE


def test_bundled_sprite_loads_and_is_normalized() -> None:
    _assert_is_usable_sprite(load_bundled_rocket_sprite_image())


@pytest.mark.parametrize(
    "raw_sprite_image",
    [
        np.full((72, 72, 4), 200, dtype=np.uint8),  # RGBA, like the bundled PNG
        np.full((57, 60), 128, dtype=np.uint8),  # grayscale
        np.full((10, 10, 1), 128, dtype=np.uint8),  # single-channel, tiny
        np.full((2000, 1000, 3), 90, dtype=np.uint8),  # far larger than the frame
        np.full((40, 80, 3), 60000, dtype=np.uint16),  # 16-bit
    ],
)
def test_awkward_images_become_usable_sprites(raw_sprite_image: np.ndarray) -> None:
    _assert_is_usable_sprite(normalize_rocket_sprite_image(raw_sprite_image))


def test_aspect_ratio_is_preserved() -> None:
    sprite_image = normalize_rocket_sprite_image(np.zeros((40, 80, 3), dtype=np.uint8))
    assert sprite_image is not None
    assert sprite_image.shape[:2] == (30, 60)


def test_transparent_pixels_become_white() -> None:
    fully_transparent = np.zeros((60, 60, 4), dtype=np.uint8)
    sprite_image = normalize_rocket_sprite_image(fully_transparent)
    assert sprite_image is not None
    assert np.all(sprite_image == 255)


@pytest.mark.parametrize(
    "unusable_sprite_image",
    [None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((10, 10, 2), dtype=np.uint8)],
)
def test_unusable_images_are_rejected(unusable_sprite_image: np.ndarray | None) -> None:
    assert normalize_rocket_sprite_image(unusable_sprite_image) is None


def test_scene_runs_with_an_oversized_caller_supplied_sprite() -> None:
    synthetic_source = SyntheticRocketSceneSource(
        rocket_sprite_image=np.full((900, 900, 4), 255, dtype=np.uint8),
        total_frame_count=200,
    )
    rendered_frame_count = 0
    while (frame := synthetic_source.read_next_frame()) is not None:
        assert frame.shape == (480, 640, 3)
        rendered_frame_count += 1
    assert rendered_frame_count == 200
