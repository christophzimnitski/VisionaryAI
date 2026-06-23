"""Dependency-free identity-consistency evaluation against the synthetic scene.

The standard MOT metrics (MOTA / IDF1) need the optional ``motmetrics`` package.
This module instead computes, with no third-party dependency, the single number
that matters most for the collision-recovery problem: **how often does a tracked
object's identity switch while the object stays continuously visible?**

For every frame it greedily matches each ground-truth object to the predicted
track it overlaps most (above an IoU threshold). It then counts an *identity
switch* whenever a ground-truth object becomes matched to a different track id
than the one it was last matched to -- the classic MOT "IDSW" definition. A
perfect tracker that holds identity straight through a head-on collision scores
zero identity switches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.geometry import BoundingBox, intersection_over_union
from visionary.io.video_sources import SyntheticRocketSceneSource
from visionary.tracking.multi_object_tracker import MultiObjectTracker


@dataclass
class _PerGroundTruthObjectStatistics:
    """Running statistics accumulated for one ground-truth object over the run."""

    total_visible_frame_count: int = 0
    matched_frame_count: int = 0
    identity_switch_count: int = 0
    # Switches split by whether the object was (recently) overlapping another at
    # the moment its identity changed. Switches while well separated are genuine
    # tracker failures; switches during an overlap are the information-theoretic
    # irreducible case (two identical objects coinciding).
    identity_switch_count_while_separated: int = 0
    identity_switch_count_near_overlap: int = 0
    fragmentation_count: int = 0
    distinct_assigned_track_ids: set[str] = field(default_factory=set)
    most_recently_assigned_track_id: str | None = None
    was_matched_in_previous_frame: bool = False
    most_recent_overlap_frame_index: int = -(10**9)


@dataclass(frozen=True)
class IdentityConsistencySummary:
    """Headline identity-consistency numbers produced by an evaluation run."""

    evaluated_frame_count: int
    total_ground_truth_visible_frames: int
    total_matched_frames: int
    total_identity_switch_count: int
    total_identity_switch_count_while_separated: int
    total_identity_switch_count_near_overlap: int
    total_fragmentation_count: int
    per_ground_truth_object_statistics: dict[str, _PerGroundTruthObjectStatistics]

    @property
    def detection_recall(self) -> float:
        if self.total_ground_truth_visible_frames == 0:
            return 0.0
        return self.total_matched_frames / self.total_ground_truth_visible_frames

    @property
    def formatted_report(self) -> str:
        lines = [
            "Identity-consistency report (synthetic collision scene)",
            f"  frames evaluated .............. {self.evaluated_frame_count}",
            f"  detection recall .............. {self.detection_recall:6.1%}",
            f"  total identity switches ....... {self.total_identity_switch_count}",
            f"    - while well separated ...... {self.total_identity_switch_count_while_separated}"
            "   (genuine tracker failures)",
            f"    - during an overlap ......... {self.total_identity_switch_count_near_overlap}"
            "   (irreducible: identical objects coinciding)",
            f"  total fragmentations .......... {self.total_fragmentation_count}",
            "  per ground-truth object:",
        ]
        for ground_truth_identifier, statistics in sorted(
            self.per_ground_truth_object_statistics.items()
        ):
            per_object_recall = (
                statistics.matched_frame_count / statistics.total_visible_frame_count
                if statistics.total_visible_frame_count
                else 0.0
            )
            lines.append(
                f"    {ground_truth_identifier:<10} "
                f"recall {per_object_recall:6.1%}  "
                f"id-switches {statistics.identity_switch_count} "
                f"(separated {statistics.identity_switch_count_while_separated})  "
                f"distinct-ids {len(statistics.distinct_assigned_track_ids)}  "
                f"fragmentations {statistics.fragmentation_count}"
            )
        return "\n".join(lines)


def _match_ground_truth_objects_to_predicted_tracks(
    ground_truth_bounding_boxes: dict[str, BoundingBox],
    predicted_track_bounding_boxes: dict[str, BoundingBox],
    intersection_over_union_match_threshold: float,
) -> dict[str, str]:
    """Greedily pair each ground-truth object with its best-overlapping track.

    Returns a mapping ``ground_truth_identifier -> predicted_track_id`` for every
    pair whose overlap clears the threshold. The assignment is one-to-one: each
    track is awarded to at most one ground-truth object.
    """
    candidate_overlaps: list[tuple[float, str, str]] = []
    for ground_truth_identifier, ground_truth_box in ground_truth_bounding_boxes.items():
        for predicted_track_id, predicted_box in predicted_track_bounding_boxes.items():
            overlap_ratio = intersection_over_union(ground_truth_box, predicted_box)
            if overlap_ratio >= intersection_over_union_match_threshold:
                candidate_overlaps.append(
                    (overlap_ratio, ground_truth_identifier, predicted_track_id)
                )

    candidate_overlaps.sort(reverse=True)
    ground_truth_to_track_assignment: dict[str, str] = {}
    already_assigned_track_ids: set[str] = set()
    for _, ground_truth_identifier, predicted_track_id in candidate_overlaps:
        if ground_truth_identifier in ground_truth_to_track_assignment:
            continue
        if predicted_track_id in already_assigned_track_ids:
            continue
        ground_truth_to_track_assignment[ground_truth_identifier] = predicted_track_id
        already_assigned_track_ids.add(predicted_track_id)
    return ground_truth_to_track_assignment


def _ground_truth_objects_currently_overlapping(
    ground_truth_bounding_boxes: dict[str, BoundingBox],
    overlap_ambiguity_threshold: float,
) -> set[str]:
    """Return the identifiers of ground-truth objects overlapping another object.

    These are the frames where two objects coincide, so an identity change there
    is the irreducible ambiguity rather than a tracker mistake.
    """
    overlapping_identifiers: set[str] = set()
    ground_truth_items = list(ground_truth_bounding_boxes.items())
    for first_index in range(len(ground_truth_items)):
        first_identifier, first_box = ground_truth_items[first_index]
        for second_index in range(first_index + 1, len(ground_truth_items)):
            second_identifier, second_box = ground_truth_items[second_index]
            if intersection_over_union(first_box, second_box) >= overlap_ambiguity_threshold:
                overlapping_identifiers.add(first_identifier)
                overlapping_identifiers.add(second_identifier)
    return overlapping_identifiers


def evaluate_identity_consistency_on_synthetic_scene(
    configuration: TrackerConfiguration | None = None,
    total_frame_count: int = 300,
    intersection_over_union_match_threshold: float = 0.3,
    ground_truth_overlap_ambiguity_threshold: float = 0.15,
    near_overlap_frame_window: int = 4,
    random_seed: int = 0,
) -> IdentityConsistencySummary:
    """Run the tracker over the synthetic collision scene and score identity stability."""
    if configuration is None:
        configuration = TrackerConfiguration()

    synthetic_scene_source = SyntheticRocketSceneSource(
        frame_width_in_pixels=configuration.frame_width_in_pixels,
        frame_height_in_pixels=configuration.frame_height_in_pixels,
        total_frame_count=total_frame_count,
        random_seed=random_seed,
    )
    motion_detector = MotionDifferenceDetector(configuration)
    tracker = MultiObjectTracker(motion_detector, configuration)

    per_object_statistics: dict[str, _PerGroundTruthObjectStatistics] = {}
    evaluated_frame_count = 0

    for frame, ground_truth_bounding_boxes in (
        synthetic_scene_source.iterate_frames_with_ground_truth()
    ):
        evaluated_frame_count += 1
        frame_processing_result = tracker.process_frame(frame)

        predicted_track_bounding_boxes = {
            track_result.track_id: track_result.bounding_box
            for track_result in frame_processing_result.track_results
        }
        ground_truth_to_track_assignment = _match_ground_truth_objects_to_predicted_tracks(
            ground_truth_bounding_boxes,
            predicted_track_bounding_boxes,
            intersection_over_union_match_threshold,
        )
        currently_overlapping_identifiers = _ground_truth_objects_currently_overlapping(
            ground_truth_bounding_boxes, ground_truth_overlap_ambiguity_threshold
        )

        for ground_truth_identifier in ground_truth_bounding_boxes:
            statistics = per_object_statistics.setdefault(
                ground_truth_identifier, _PerGroundTruthObjectStatistics()
            )
            statistics.total_visible_frame_count += 1
            if ground_truth_identifier in currently_overlapping_identifiers:
                statistics.most_recent_overlap_frame_index = evaluated_frame_count

            assigned_track_id = ground_truth_to_track_assignment.get(ground_truth_identifier)
            if assigned_track_id is None:
                statistics.was_matched_in_previous_frame = False
                continue

            statistics.matched_frame_count += 1
            statistics.distinct_assigned_track_ids.add(assigned_track_id)
            if (
                statistics.most_recently_assigned_track_id is not None
                and statistics.most_recently_assigned_track_id != assigned_track_id
            ):
                statistics.identity_switch_count += 1
                frames_since_overlap = (
                    evaluated_frame_count - statistics.most_recent_overlap_frame_index
                )
                if frames_since_overlap <= near_overlap_frame_window:
                    statistics.identity_switch_count_near_overlap += 1
                else:
                    statistics.identity_switch_count_while_separated += 1
            if not statistics.was_matched_in_previous_frame and statistics.matched_frame_count > 1:
                statistics.fragmentation_count += 1
            statistics.most_recently_assigned_track_id = assigned_track_id
            statistics.was_matched_in_previous_frame = True

    return IdentityConsistencySummary(
        evaluated_frame_count=evaluated_frame_count,
        total_ground_truth_visible_frames=sum(
            statistics.total_visible_frame_count
            for statistics in per_object_statistics.values()
        ),
        total_matched_frames=sum(
            statistics.matched_frame_count for statistics in per_object_statistics.values()
        ),
        total_identity_switch_count=sum(
            statistics.identity_switch_count for statistics in per_object_statistics.values()
        ),
        total_identity_switch_count_while_separated=sum(
            statistics.identity_switch_count_while_separated
            for statistics in per_object_statistics.values()
        ),
        total_identity_switch_count_near_overlap=sum(
            statistics.identity_switch_count_near_overlap
            for statistics in per_object_statistics.values()
        ),
        total_fragmentation_count=sum(
            statistics.fragmentation_count for statistics in per_object_statistics.values()
        ),
        per_ground_truth_object_statistics=per_object_statistics,
    )


def main() -> None:
    """CLI: ``python -m visionary.evaluation.identity_consistency_evaluation``."""
    evaluation_summary = evaluate_identity_consistency_on_synthetic_scene()
    print(evaluation_summary.formatted_report)


if __name__ == "__main__":
    main()
