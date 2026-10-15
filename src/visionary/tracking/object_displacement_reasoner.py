"""Reasoning that links a vanished object to a re-appeared one.

This is the heart of the "the missing pillow is on the floor" deduction. The
system never watched the pillow being thrown -- it only knows two *changes*:

  * a **disappearance**: a big object that was anchored in the room is no longer
    where it was left (its spot came back into view, but the object did not), and
  * an **appearance**: a big object turned up somewhere it was not before.

Neither change, on its own, says anything moved. This reasoner holds the
unmatched changes of each kind in a short memory and periodically asks: *is the
thing that vanished here the thing that appeared there?* The answer is a partial
feature match (same chromaticity, size, aspect), gated so that two changes are
only fused when they look alike, are far enough apart to be a genuine move rather
than a re-acquisition in place, and are close enough in time to plausibly be one
event. The optimal pairing across all pending changes is solved once with the
Hungarian algorithm, so two identical pillows that both moved are matched to the
right slots rather than greedily mis-paired.

Crucially this works at the *slot* level, which sidesteps the unsolvable case:
when two pillows look identical you cannot say *which* one is now on the floor,
but you do not need to -- a pillow-shaped hole on the bed paired with a
pillow-shaped object on the floor is enough to conclude "a pillow moved there".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from visionary.configuration import TrackerConfiguration
from visionary.features.object_feature_memory import (
    ObservedObjectFeatures,
    compare_observed_object_features,
)
from visionary.logging_configuration import get_logger_for_module

_logger = get_logger_for_module(__name__)

# Cost filled in for a pairing forbidden by the gates, mirroring the convention
# used by the track/detection association so the solver never prefers a bad pair.
_FORBIDDEN_PAIR_COST = 1.0e6


@dataclass(frozen=True)
class RememberedObjectChange:
    """One pending change of identity -- something that vanished or appeared.

    ``object_identity`` identifies the object the change came from (the room
    monitor mints these itself rather than reusing a tracker track id); the same
    dataclass describes both a disappearance and an appearance because the
    reasoner only cares about *where*, *what it looked like*, and *when*.
    """

    object_identity: str
    feature_signature: ObservedObjectFeatures
    world_position_x: float
    world_position_y: float
    frame_index: int


@dataclass(frozen=True)
class DisplacementConclusion:
    """A fused pair of changes: an object that was here is now there."""

    disappeared_object_identity: str
    appeared_object_identity: str
    from_world_position: tuple[float, float]
    to_world_position: tuple[float, float]
    feature_similarity: float
    frames_between_observations: int
    concluded_at_frame_index: int


class ObjectDisplacementReasoner:
    """Pairs disappearance changes with appearance changes into displacements."""

    def __init__(self, configuration: TrackerConfiguration) -> None:
        self._configuration = configuration
        self._unmatched_disappearances: list[RememberedObjectChange] = []
        self._unmatched_appearances: list[RememberedObjectChange] = []
        self._displacement_conclusions: list[DisplacementConclusion] = []

    # -- recording changes ----------------------------------------------------

    def record_object_disappeared(
        self,
        object_identity: str,
        feature_signature: ObservedObjectFeatures,
        world_position: tuple[float, float],
        frame_index: int,
    ) -> None:
        """Remember that an anchored object is no longer where it was left."""
        self._unmatched_disappearances.append(
            RememberedObjectChange(
                object_identity=object_identity,
                feature_signature=feature_signature,
                world_position_x=world_position[0],
                world_position_y=world_position[1],
                frame_index=frame_index,
            )
        )

    def record_object_appeared(
        self,
        object_identity: str,
        feature_signature: ObservedObjectFeatures,
        world_position: tuple[float, float],
        frame_index: int,
    ) -> None:
        """Remember that a big object turned up where there was none before."""
        self._unmatched_appearances.append(
            RememberedObjectChange(
                object_identity=object_identity,
                feature_signature=feature_signature,
                world_position_x=world_position[0],
                world_position_y=world_position[1],
                frame_index=frame_index,
            )
        )

    # -- reconciliation -------------------------------------------------------

    def reconcile_into_displacement_conclusions(
        self, current_frame_index: int
    ) -> list[DisplacementConclusion]:
        """Match pending disappearances to appearances; return the new conclusions.

        Each accepted pairing consumes both changes (they are explained), so a
        change is never used in two conclusions. Conclusions are also kept in a
        cumulative list for callers that want the running history.
        """
        self._forget_stale_changes(current_frame_index)
        if not self._unmatched_disappearances or not self._unmatched_appearances:
            return []

        pairwise_cost_matrix, similarity_matrix = self._build_pairing_cost_matrix()
        matched_disappearance_rows, matched_appearance_columns = linear_sum_assignment(
            pairwise_cost_matrix
        )

        newly_concluded: list[DisplacementConclusion] = []
        consumed_disappearance_indices: set[int] = set()
        consumed_appearance_indices: set[int] = set()
        for disappearance_index, appearance_index in zip(
            matched_disappearance_rows, matched_appearance_columns, strict=True
        ):
            if pairwise_cost_matrix[disappearance_index, appearance_index] >= _FORBIDDEN_PAIR_COST:
                continue
            disappearance = self._unmatched_disappearances[disappearance_index]
            appearance = self._unmatched_appearances[appearance_index]
            conclusion = DisplacementConclusion(
                disappeared_object_identity=disappearance.object_identity,
                appeared_object_identity=appearance.object_identity,
                from_world_position=(
                    disappearance.world_position_x,
                    disappearance.world_position_y,
                ),
                to_world_position=(appearance.world_position_x, appearance.world_position_y),
                feature_similarity=float(similarity_matrix[disappearance_index, appearance_index]),
                frames_between_observations=abs(appearance.frame_index - disappearance.frame_index),
                concluded_at_frame_index=current_frame_index,
            )
            newly_concluded.append(conclusion)
            consumed_disappearance_indices.add(disappearance_index)
            consumed_appearance_indices.add(appearance_index)
            _logger.info(
                "Displacement concluded at frame %d: object %s that vanished near "
                "(%.0f, %.0f) is the object %s now near (%.0f, %.0f) (similarity %.2f)",
                current_frame_index,
                conclusion.disappeared_object_identity,
                conclusion.from_world_position[0],
                conclusion.from_world_position[1],
                conclusion.appeared_object_identity,
                conclusion.to_world_position[0],
                conclusion.to_world_position[1],
                conclusion.feature_similarity,
            )

        self._unmatched_disappearances = [
            change
            for index, change in enumerate(self._unmatched_disappearances)
            if index not in consumed_disappearance_indices
        ]
        self._unmatched_appearances = [
            change
            for index, change in enumerate(self._unmatched_appearances)
            if index not in consumed_appearance_indices
        ]
        self._displacement_conclusions.extend(newly_concluded)
        return newly_concluded

    def _build_pairing_cost_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the (disappearance x appearance) cost and similarity matrices.

        A pairing is forbidden unless it clears every gate; an allowed pairing's
        cost is the negated feature similarity so the minimizing solver maximizes
        total similarity over the chosen pairs.
        """
        disappearance_count = len(self._unmatched_disappearances)
        appearance_count = len(self._unmatched_appearances)
        pairwise_cost_matrix = np.full(
            (disappearance_count, appearance_count), _FORBIDDEN_PAIR_COST, dtype=np.float64
        )
        similarity_matrix = np.zeros((disappearance_count, appearance_count), dtype=np.float64)

        for disappearance_index, disappearance in enumerate(self._unmatched_disappearances):
            for appearance_index, appearance in enumerate(self._unmatched_appearances):
                feature_similarity = compare_observed_object_features(
                    disappearance.feature_signature, appearance.feature_signature
                )
                similarity_matrix[disappearance_index, appearance_index] = feature_similarity
                if not self._pairing_clears_every_gate(
                    disappearance, appearance, feature_similarity
                ):
                    continue
                pairwise_cost_matrix[disappearance_index, appearance_index] = -feature_similarity
        return pairwise_cost_matrix, similarity_matrix

    def _pairing_clears_every_gate(
        self,
        disappearance: RememberedObjectChange,
        appearance: RememberedObjectChange,
        feature_similarity: float,
    ) -> bool:
        if (
            feature_similarity
            < self._configuration.displacement_reasoning_minimum_feature_similarity
        ):
            return False
        world_distance = float(
            np.hypot(
                appearance.world_position_x - disappearance.world_position_x,
                appearance.world_position_y - disappearance.world_position_y,
            )
        )
        if (
            world_distance
            < self._configuration.displacement_reasoning_minimum_world_distance_in_pixels
        ):
            # Too close to be a move: this is a re-acquisition in place, which the
            # tracker's own re-identification already handles -- not a displacement.
            return False
        return True

    def _forget_stale_changes(self, current_frame_index: int) -> None:
        maximum_age = self._configuration.displacement_reasoning_maximum_event_age_in_frames
        self._unmatched_disappearances = [
            change
            for change in self._unmatched_disappearances
            if current_frame_index - change.frame_index <= maximum_age
        ]
        self._unmatched_appearances = [
            change
            for change in self._unmatched_appearances
            if current_frame_index - change.frame_index <= maximum_age
        ]

    # -- introspection --------------------------------------------------------

    @property
    def displacement_conclusions(self) -> list[DisplacementConclusion]:
        """Every displacement concluded so far, in the order they were drawn."""
        return list(self._displacement_conclusions)

    @property
    def pending_disappearance_count(self) -> int:
        return len(self._unmatched_disappearances)

    @property
    def pending_appearance_count(self) -> int:
        return len(self._unmatched_appearances)
