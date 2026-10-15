"""Unit tests for the displacement reasoner (the "missing pillow is on the floor" core).

These exercise the pure event-pairing logic with hand-built feature signatures,
no images: a disappearance and a matching appearance far apart should fuse into
one displacement conclusion; non-matches, in-place re-acquisitions, and stale
changes should not.
"""

from __future__ import annotations

import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.features.object_feature_memory import ObservedObjectFeatures
from visionary.tracking.object_displacement_reasoner import ObjectDisplacementReasoner


def _signature(
    mean_color_bgr: list[float], normalized_size: float = 0.02
) -> ObservedObjectFeatures:
    return ObservedObjectFeatures(
        mean_color_bgr=np.array(mean_color_bgr, dtype=np.float32),
        normalized_size=normalized_size,
        aspect_ratio=1.0,
    )


def test_a_vanished_object_and_a_matching_appearance_fuse_into_one_displacement() -> None:
    reasoner = ObjectDisplacementReasoner(TrackerConfiguration())
    pillow_signature = _signature([230.0, 230.0, 230.0])

    reasoner.record_object_disappeared("pillow_on_bed", pillow_signature, (150.0, 120.0), 40)
    reasoner.record_object_appeared("unknown_floor_object", pillow_signature, (520.0, 400.0), 95)

    conclusions = reasoner.reconcile_into_displacement_conclusions(current_frame_index=96)

    assert len(conclusions) == 1
    conclusion = conclusions[0]
    assert conclusion.disappeared_object_identity == "pillow_on_bed"
    assert conclusion.appeared_object_identity == "unknown_floor_object"
    assert conclusion.from_world_position == (150.0, 120.0)
    assert conclusion.to_world_position == (520.0, 400.0)
    assert conclusion.frames_between_observations == 55
    # Both pending changes are consumed once explained.
    assert reasoner.pending_disappearance_count == 0
    assert reasoner.pending_appearance_count == 0


def test_two_identical_pillows_are_matched_to_the_right_slots() -> None:
    # Two look-alike pillows go missing; two look-alike pillows appear elsewhere.
    # Slot matching should pair each disappearance with one appearance (any valid
    # pairing), consuming all four changes -- without inventing extra conclusions.
    reasoner = ObjectDisplacementReasoner(TrackerConfiguration())
    pillow = _signature([225.0, 225.0, 225.0])

    reasoner.record_object_disappeared("bed_slot_left", pillow, (140.0, 120.0), 10)
    reasoner.record_object_disappeared("bed_slot_right", pillow, (220.0, 120.0), 10)
    reasoner.record_object_appeared("floor_object_one", pillow, (500.0, 410.0), 60)
    reasoner.record_object_appeared("floor_object_two", pillow, (560.0, 300.0), 65)

    conclusions = reasoner.reconcile_into_displacement_conclusions(current_frame_index=66)

    assert len(conclusions) == 2
    disappeared = {conclusion.disappeared_object_identity for conclusion in conclusions}
    appeared = {conclusion.appeared_object_identity for conclusion in conclusions}
    assert disappeared == {"bed_slot_left", "bed_slot_right"}
    assert appeared == {"floor_object_one", "floor_object_two"}
    assert reasoner.pending_disappearance_count == 0
    assert reasoner.pending_appearance_count == 0


def test_a_clearly_different_appearance_is_not_fused() -> None:
    reasoner = ObjectDisplacementReasoner(TrackerConfiguration())
    reasoner.record_object_disappeared(
        "white_pillow", _signature([235.0, 235.0, 235.0]), (150.0, 120.0), 10
    )
    reasoner.record_object_appeared(
        "orange_thing", _signature([20.0, 120.0, 235.0]), (520.0, 400.0), 40
    )

    conclusions = reasoner.reconcile_into_displacement_conclusions(current_frame_index=41)

    assert conclusions == []
    # The unmatched changes are retained for a possible later partner.
    assert reasoner.pending_disappearance_count == 1
    assert reasoner.pending_appearance_count == 1


def test_an_appearance_in_the_same_place_is_a_re_acquisition_not_a_displacement() -> None:
    reasoner = ObjectDisplacementReasoner(TrackerConfiguration())
    pillow = _signature([230.0, 230.0, 230.0])
    reasoner.record_object_disappeared("pillow", pillow, (150.0, 120.0), 10)
    # Reappears essentially where it vanished -- below the minimum move distance.
    reasoner.record_object_appeared("pillow_again", pillow, (158.0, 126.0), 30)

    conclusions = reasoner.reconcile_into_displacement_conclusions(current_frame_index=31)

    assert conclusions == []


def test_stale_changes_are_forgotten() -> None:
    configuration = TrackerConfiguration()
    reasoner = ObjectDisplacementReasoner(configuration)
    pillow = _signature([230.0, 230.0, 230.0])
    reasoner.record_object_disappeared("pillow", pillow, (150.0, 120.0), 10)

    # An appearance arriving far beyond the retention window cannot pair with the
    # long-forgotten disappearance.
    very_late_frame = 10 + configuration.displacement_reasoning_maximum_event_age_in_frames + 50
    reasoner.record_object_appeared("late_object", pillow, (520.0, 400.0), very_late_frame)
    conclusions = reasoner.reconcile_into_displacement_conclusions(
        current_frame_index=very_late_frame
    )

    assert conclusions == []
    assert reasoner.pending_disappearance_count == 0
    assert reasoner.pending_appearance_count == 1
