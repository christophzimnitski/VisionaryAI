"""Quantitative tracking evaluation against the synthetic ground truth.

The synthetic scene knows exactly where every object is in every frame, so we can
score the tracker objectively instead of eyeballing it. This harness feeds the
ground truth and the tracker's predictions into ``motmetrics`` and reports the
standard multi-object-tracking metrics (MOTA, IDF1, ID switches, ...).

``motmetrics`` is an optional dependency; install it with ``pip install motmetrics``.
Turning a "did my change help?" question into a single number is what lets every
future change be measured rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.geometry import BoundingBox
from visionary.io.video_sources import SyntheticRocketSceneSource
from visionary.tracking.multi_object_tracker import MultiObjectTracker


@dataclass(frozen=True)
class TrackingEvaluationSummary:
    """The headline metrics produced by an evaluation run."""

    multi_object_tracking_accuracy: float
    identity_f1_score: float
    identity_switch_count: int
    mostly_tracked_object_count: int
    mostly_lost_object_count: int
    formatted_report: str


def _convert_bounding_box_to_left_top_width_height(
    bounding_box: BoundingBox,
) -> tuple[float, float, float, float]:
    left_x, top_y, width, height = bounding_box
    return float(left_x), float(top_y), float(width), float(height)


def evaluate_tracker_on_synthetic_scene(
    configuration: TrackerConfiguration | None = None,
    total_frame_count: int = 500,
    intersection_over_union_match_threshold: float = 0.5,
) -> TrackingEvaluationSummary:
    """Run the tracker over the synthetic scene and return MOT metrics.

    Raises ``ImportError`` with an actionable message if ``motmetrics`` is missing.
    """
    try:
        import motmetrics
    except ImportError as missing_dependency_error:  # pragma: no cover - trivial guard
        raise ImportError(
            "The evaluation harness requires the 'motmetrics' package. "
            "Install it with: pip install motmetrics"
        ) from missing_dependency_error

    if configuration is None:
        configuration = TrackerConfiguration()

    synthetic_scene_source = SyntheticRocketSceneSource(
        frame_width_in_pixels=configuration.frame_width_in_pixels,
        frame_height_in_pixels=configuration.frame_height_in_pixels,
        total_frame_count=total_frame_count,
    )
    motion_detector = MotionDifferenceDetector(configuration)
    tracker = MultiObjectTracker(motion_detector, configuration)

    accumulator = motmetrics.MOTAccumulator(auto_id=True)

    for frame, ground_truth_bounding_boxes in (
        synthetic_scene_source.iterate_frames_with_ground_truth()
    ):
        frame_processing_result = tracker.process_frame(frame)

        ground_truth_object_identifiers = list(ground_truth_bounding_boxes.keys())
        ground_truth_box_list = [
            _convert_bounding_box_to_left_top_width_height(bounding_box)
            for bounding_box in ground_truth_bounding_boxes.values()
        ]

        predicted_track_identifiers = [
            track_result.track_id
            for track_result in frame_processing_result.track_results
        ]
        predicted_box_list = [
            _convert_bounding_box_to_left_top_width_height(track_result.bounding_box)
            for track_result in frame_processing_result.track_results
        ]

        distance_matrix = motmetrics.distances.iou_matrix(
            ground_truth_box_list,
            predicted_box_list,
            max_iou=1.0 - intersection_over_union_match_threshold,
        )
        accumulator.update(
            ground_truth_object_identifiers,
            predicted_track_identifiers,
            distance_matrix,
        )

    metrics_host = motmetrics.metrics.create()
    computed_metrics = metrics_host.compute(
        accumulator,
        metrics=["mota", "idf1", "num_switches", "mostly_tracked", "mostly_lost"],
        name="synthetic_scene",
    )
    formatted_report = motmetrics.io.render_summary(
        computed_metrics,
        namemap=motmetrics.io.motchallenge_metric_names,
    )

    return TrackingEvaluationSummary(
        multi_object_tracking_accuracy=float(computed_metrics["mota"].iloc[0]),
        identity_f1_score=float(computed_metrics["idf1"].iloc[0]),
        identity_switch_count=int(computed_metrics["num_switches"].iloc[0]),
        mostly_tracked_object_count=int(computed_metrics["mostly_tracked"].iloc[0]),
        mostly_lost_object_count=int(computed_metrics["mostly_lost"].iloc[0]),
        formatted_report=formatted_report,
    )


def main() -> None:
    """Command-line entry point: ``python -m visionary.evaluation.mot_metrics_evaluation``."""
    evaluation_summary = evaluate_tracker_on_synthetic_scene()
    print(evaluation_summary.formatted_report)


if __name__ == "__main__":
    main()
