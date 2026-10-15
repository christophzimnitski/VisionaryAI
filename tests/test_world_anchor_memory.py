"""Unit tests for the world anchor memory (the lightweight room map)."""

from __future__ import annotations

import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.features.object_feature_memory import ObservedObjectFeatures
from visionary.mapping.world_anchor_memory import WorldAnchorMemory


def _signature() -> ObservedObjectFeatures:
    return ObservedObjectFeatures(
        mean_color_bgr=np.array([230.0, 230.0, 230.0], dtype=np.float32),
        normalized_size=0.02,
        aspect_ratio=1.0,
    )


def test_an_anchor_is_in_view_only_when_its_spot_is_on_screen() -> None:
    memory = WorldAnchorMemory(TrackerConfiguration())
    memory.register_anchor(
        "pillow", _signature(), world_center=(900.0, 300.0), box_size=(80, 60), frame_index=1
    )

    # Camera looking at world x in [820, 1460]: the anchor at x=900 is on screen.
    in_view = memory.anchors_expected_in_view(
        world_offset=(820.0, 100.0), frame_width=640, frame_height=480
    )
    assert [anchor.object_identity for anchor in in_view] == ["pillow"]

    # Camera panned far right (world x in [1200, 1840]): the anchor is behind us.
    out_of_view = memory.anchors_expected_in_view(
        world_offset=(1200.0, 100.0), frame_width=640, frame_height=480
    )
    assert out_of_view == []


def test_refuting_an_anchor_declares_it_missing_only_after_enough_frames() -> None:
    configuration = TrackerConfiguration()
    memory = WorldAnchorMemory(configuration)
    memory.register_anchor(
        "pillow", _signature(), world_center=(900.0, 300.0), box_size=(80, 60), frame_index=1
    )

    refutations_needed = configuration.anchor_consecutive_refutations_before_declared_missing
    declared_missing_on = [memory.refute_anchor("pillow") for _ in range(refutations_needed)]

    # Only the final refutation crosses the patience threshold.
    assert declared_missing_on[:-1] == [False] * (refutations_needed - 1)
    assert declared_missing_on[-1] is True


def test_confirming_an_anchor_clears_refutations_and_updates_last_seen() -> None:
    memory = WorldAnchorMemory(TrackerConfiguration())
    memory.register_anchor(
        "pillow", _signature(), world_center=(900.0, 300.0), box_size=(80, 60), frame_index=1
    )
    memory.refute_anchor("pillow")
    memory.refute_anchor("pillow")

    memory.confirm_anchor(
        "pillow", world_center=(905.0, 305.0), feature_signature=_signature(), frame_index=50
    )

    anchor = memory.get_anchor("pillow")
    assert anchor.consecutive_refutation_count == 0
    assert anchor.last_confirmed_frame_index == 50
    # The world centre was nudged toward, but not snapped onto, the new observation.
    assert 900.0 < anchor.world_center_x < 905.0


def test_discarding_an_anchor_removes_it() -> None:
    memory = WorldAnchorMemory(TrackerConfiguration())
    memory.register_anchor(
        "pillow", _signature(), world_center=(900.0, 300.0), box_size=(80, 60), frame_index=1
    )
    assert memory.has_anchor("pillow")
    memory.discard_anchor("pillow")
    assert not memory.has_anchor("pillow")
    assert memory.anchor_count == 0
