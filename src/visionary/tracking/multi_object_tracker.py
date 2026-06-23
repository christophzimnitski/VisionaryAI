"""The multi-object tracking orchestrator.

This is the heart of the system. Each call to :meth:`process_frame` runs the full
tracking-by-detection cycle for one frame:

  1. (optional) estimate camera motion and warp predictions,
  2. ask the injected detector for detections and describe their appearance once,
  3. predict every existing track's new position with its Kalman filter,
  4. optimally associate active tracks to detections (Hungarian assignment),
  5. update matched tracks; coast unmatched ones and demote them to LOST,
  6. try to recover LOST tracks from leftover detections with looser gates,
  7. spawn new tracks for genuinely new detections,
  8. retire LOST tracks that have not reappeared for too long.

Crucially, this class performs **no drawing** — it returns a structured
:class:`FrameProcessingResult`, which the visualization layer consumes separately.
That separation is what makes the tracker runnable headless and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import ObjectDetector
from visionary.features.appearance_traits import extract_appearance_trait_vector
from visionary.geometry import BoundingBox, intersection_over_union
from visionary.logging_configuration import get_logger_for_module
from visionary.tracking.association import associate_detections_to_tracks
from visionary.tracking.camera_motion_compensation import (
    estimate_camera_motion_homography,
)
from visionary.tracking.track import TrackedObject, TrackLifecycleState

_logger = get_logger_for_module(__name__)


@dataclass(frozen=True)
class TrackResult:
    """The per-frame, drawing-free description of one track for downstream use."""

    track_id: str
    object_type_token: str
    bounding_box: BoundingBox
    lifecycle_state: TrackLifecycleState
    consecutive_missed_frame_count: int
    was_created_this_frame: bool = False
    was_recovered_this_frame: bool = False


@dataclass
class FrameProcessingResult:
    """Everything the tracker produces for a single processed frame."""

    frame_index: int
    track_results: list[TrackResult] = field(default_factory=list)
    detection_bounding_boxes: list[BoundingBox] = field(default_factory=list)
    collision_or_occlusion_detected: bool = False


class MultiObjectTracker:
    """Tracks an arbitrary number of objects across frames using a pluggable detector."""

    def __init__(
        self, object_detector: ObjectDetector, configuration: TrackerConfiguration
    ) -> None:
        self._object_detector = object_detector
        self._configuration = configuration
        self._active_tracks_by_id: dict[str, TrackedObject] = {}
        self._lost_tracks_by_id: dict[str, TrackedObject] = {}
        self._previous_grayscale_frame: np.ndarray | None = None
        self._frame_index = 0

    # -- public API -----------------------------------------------------------

    @property
    def active_track_count(self) -> int:
        return len(self._active_tracks_by_id)

    def process_frame(self, frame: np.ndarray) -> FrameProcessingResult:
        self._frame_index += 1
        frame = self._resize_frame_if_needed(frame)
        current_grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._previous_grayscale_frame is None:
            self._previous_grayscale_frame = current_grayscale_frame
            return FrameProcessingResult(frame_index=self._frame_index)

        camera_motion_homography = self._estimate_camera_motion_if_enabled(
            current_grayscale_frame
        )

        detections = self._object_detector.detect_objects_in_frame(frame)
        detection_bounding_boxes = [detection.bounding_box for detection in detections]
        # Describe every detection's appearance exactly once and reuse it for
        # association, track updates, and new-track creation (the original code
        # recomputed this descriptor several times per detection).
        detection_appearance_trait_vectors = [
            extract_appearance_trait_vector(frame, bounding_box)
            for bounding_box in detection_bounding_boxes
        ]

        active_tracks = list(self._active_tracks_by_id.values())
        lost_tracks = list(self._lost_tracks_by_id.values())

        active_prediction = self._predict_tracks(active_tracks, camera_motion_homography)
        lost_prediction = self._predict_tracks(lost_tracks, camera_motion_homography)

        # --- Stage 1: associate ACTIVE tracks to detections (priority) --------
        matched_active_pairs, unmatched_detection_indices, _ = (
            associate_detections_to_tracks(
                active_prediction.predicted_bounding_boxes,
                detection_bounding_boxes,
                active_prediction.predicted_centers,
                active_prediction.innovation_covariances,
                active_prediction.appearance_trait_vectors,
                detection_appearance_trait_vectors,
                self._configuration.intersection_over_union_gate_threshold,
                self._configuration.mahalanobis_distance_gate_threshold,
                self._configuration.intersection_over_union_score_weight,
                self._configuration.mahalanobis_distance_score_weight,
                self._configuration.appearance_similarity_score_weight,
                self._configuration.mahalanobis_distance_score_normalization,
            )
        )

        track_results: list[TrackResult] = []
        matched_active_track_indices = {track_index for track_index, _ in matched_active_pairs}

        for track_index, detection_index in matched_active_pairs:
            matched_track = active_tracks[track_index]
            matched_track.update_with_matched_detection(
                detection_bounding_boxes[detection_index],
                self._frame_index,
                detection_appearance_trait_vectors[detection_index],
            )
            track_results.append(self._build_track_result(matched_track))

        # --- Stage 2: coast unmatched ACTIVE tracks, demote to LOST if needed --
        for track_index, active_track in enumerate(active_tracks):
            if track_index in matched_active_track_indices:
                continue
            self._coast_and_maybe_demote_track(active_track)
            if active_track.lifecycle_state == TrackLifecycleState.ACTIVE:
                track_results.append(self._build_track_result(active_track))

        # --- Stage 3: try to recover LOST tracks from leftover detections ------
        unmatched_detection_indices = self._attempt_lost_track_recovery(
            lost_tracks,
            lost_prediction,
            detection_bounding_boxes,
            detection_appearance_trait_vectors,
            unmatched_detection_indices,
            track_results,
        )

        # --- Stage 4: spawn new tracks for genuinely new detections ------------
        self._spawn_new_tracks_for_unmatched_detections(
            detection_bounding_boxes,
            detection_appearance_trait_vectors,
            unmatched_detection_indices,
            track_results,
        )

        # --- Stage 5: retire LOST tracks that have been gone too long ----------
        self._retire_expired_lost_tracks()

        collision_or_occlusion_detected = self._detect_collision_or_occlusion(
            detection_count=len(detection_bounding_boxes)
        )

        self._previous_grayscale_frame = current_grayscale_frame
        return FrameProcessingResult(
            frame_index=self._frame_index,
            track_results=track_results,
            detection_bounding_boxes=detection_bounding_boxes,
            collision_or_occlusion_detected=collision_or_occlusion_detected,
        )

    # -- internal helpers -----------------------------------------------------

    @dataclass
    class _TrackPredictionBundle:
        predicted_bounding_boxes: list[BoundingBox] = field(default_factory=list)
        predicted_centers: list[tuple[float, float]] = field(default_factory=list)
        innovation_covariances: list[np.ndarray] = field(default_factory=list)
        appearance_trait_vectors: list[np.ndarray] = field(default_factory=list)

    def _resize_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if (
            frame.shape[0] != self._configuration.frame_height_in_pixels
            or frame.shape[1] != self._configuration.frame_width_in_pixels
        ):
            return cv2.resize(
                frame,
                (
                    self._configuration.frame_width_in_pixels,
                    self._configuration.frame_height_in_pixels,
                ),
            )
        return frame

    def _estimate_camera_motion_if_enabled(
        self, current_grayscale_frame: np.ndarray
    ) -> np.ndarray | None:
        if not self._configuration.enable_camera_motion_compensation:
            return None
        return estimate_camera_motion_homography(
            self._previous_grayscale_frame, current_grayscale_frame, self._configuration
        )

    def _predict_tracks(
        self,
        tracks: list[TrackedObject],
        camera_motion_homography: np.ndarray | None,
    ) -> "MultiObjectTracker._TrackPredictionBundle":
        prediction_bundle = self._TrackPredictionBundle()
        for track in tracks:
            if camera_motion_homography is not None:
                track.apply_camera_motion_warp(camera_motion_homography)
            (
                predicted_bounding_box,
                predicted_center_x,
                predicted_center_y,
                innovation_covariance,
            ) = track.predict_bounding_box()
            prediction_bundle.predicted_bounding_boxes.append(predicted_bounding_box)
            prediction_bundle.predicted_centers.append(
                (predicted_center_x, predicted_center_y)
            )
            prediction_bundle.innovation_covariances.append(innovation_covariance)
            prediction_bundle.appearance_trait_vectors.append(track.appearance_trait_vector)
        return prediction_bundle

    def _coast_and_maybe_demote_track(self, track: TrackedObject) -> None:
        previously_missed_frame_count = track.consecutive_missed_frame_count
        track.coast_through_missed_frame()
        if previously_missed_frame_count == 0 and track.consecutive_missed_frame_count == 1:
            _logger.info(
                "Track %s lost detection (coasting) at frame %d",
                track.track_id,
                self._frame_index,
            )
        if (
            track.consecutive_missed_frame_count
            > self._configuration.maximum_consecutive_missed_frames_before_lost
        ):
            track.lifecycle_state = TrackLifecycleState.LOST
            self._lost_tracks_by_id[track.track_id] = self._active_tracks_by_id.pop(
                track.track_id
            )
            _logger.info(
                "Track %s entered LOST state at frame %d",
                track.track_id,
                self._frame_index,
            )

    def _attempt_lost_track_recovery(
        self,
        lost_tracks: list[TrackedObject],
        lost_prediction: "MultiObjectTracker._TrackPredictionBundle",
        detection_bounding_boxes: list[BoundingBox],
        detection_appearance_trait_vectors: list[np.ndarray],
        unmatched_detection_indices: list[int],
        track_results: list[TrackResult],
    ) -> list[int]:
        if not unmatched_detection_indices or not lost_tracks:
            return unmatched_detection_indices

        leftover_detection_bounding_boxes = [
            detection_bounding_boxes[detection_index]
            for detection_index in unmatched_detection_indices
        ]
        leftover_detection_appearance_trait_vectors = [
            detection_appearance_trait_vectors[detection_index]
            for detection_index in unmatched_detection_indices
        ]

        matched_lost_pairs, remaining_leftover_indices, _ = associate_detections_to_tracks(
            lost_prediction.predicted_bounding_boxes,
            leftover_detection_bounding_boxes,
            lost_prediction.predicted_centers,
            lost_prediction.innovation_covariances,
            lost_prediction.appearance_trait_vectors,
            leftover_detection_appearance_trait_vectors,
            self._configuration.lost_track_intersection_over_union_gate_threshold,
            self._configuration.lost_track_mahalanobis_distance_gate_threshold,
            self._configuration.intersection_over_union_score_weight,
            self._configuration.mahalanobis_distance_score_weight,
            self._configuration.appearance_similarity_score_weight,
            self._configuration.mahalanobis_distance_score_normalization,
        )

        for lost_track_index, leftover_detection_index in matched_lost_pairs:
            original_detection_index = unmatched_detection_indices[leftover_detection_index]
            recovered_track = lost_tracks[lost_track_index]
            recovered_track.update_with_matched_detection(
                detection_bounding_boxes[original_detection_index],
                self._frame_index,
                detection_appearance_trait_vectors[original_detection_index],
            )
            self._active_tracks_by_id[recovered_track.track_id] = (
                self._lost_tracks_by_id.pop(recovered_track.track_id)
            )
            _logger.info(
                "Track %s recovered from LOST state at frame %d",
                recovered_track.track_id,
                self._frame_index,
            )
            track_results.append(
                self._build_track_result(recovered_track, was_recovered_this_frame=True)
            )

        return [
            unmatched_detection_indices[leftover_index]
            for leftover_index in remaining_leftover_indices
        ]

    def _spawn_new_tracks_for_unmatched_detections(
        self,
        detection_bounding_boxes: list[BoundingBox],
        detection_appearance_trait_vectors: list[np.ndarray],
        unmatched_detection_indices: list[int],
        track_results: list[TrackResult],
    ) -> None:
        for detection_index in unmatched_detection_indices:
            detection_bounding_box = detection_bounding_boxes[detection_index]
            is_already_covered_by_existing_track = any(
                intersection_over_union(
                    existing_track.most_recent_bounding_box, detection_bounding_box
                )
                > self._configuration.intersection_over_union_gate_threshold
                for existing_track in self._active_tracks_by_id.values()
            )
            if is_already_covered_by_existing_track:
                continue

            newly_created_track = TrackedObject(
                detection_bounding_box,
                self._frame_index,
                detection_appearance_trait_vectors[detection_index],
            )
            self._active_tracks_by_id[newly_created_track.track_id] = newly_created_track
            _logger.info(
                "New track %s (type %s) created at frame %d",
                newly_created_track.track_id,
                newly_created_track.object_type_token,
                self._frame_index,
            )
            track_results.append(
                self._build_track_result(newly_created_track, was_created_this_frame=True)
            )

    def _retire_expired_lost_tracks(self) -> None:
        retirement_threshold = (
            self._configuration.maximum_consecutive_missed_frames_before_lost
            + self._configuration.maximum_frames_to_retain_lost_track
        )
        for lost_track_id in list(self._lost_tracks_by_id.keys()):
            lost_track = self._lost_tracks_by_id[lost_track_id]
            if lost_track.consecutive_missed_frame_count > retirement_threshold:
                _logger.info(
                    "Track %s permanently retired at frame %d",
                    lost_track_id,
                    self._frame_index,
                )
                del self._lost_tracks_by_id[lost_track_id]

    def _detect_collision_or_occlusion(self, detection_count: int) -> bool:
        # Heuristic: fewer detections than active tracks while several objects are
        # present usually means two objects have merged into one blob.
        active_track_count = len(self._active_tracks_by_id)
        if detection_count < active_track_count and active_track_count > 1:
            _logger.info(
                "Collision or occlusion detected at frame %d", self._frame_index
            )
            return True
        return False

    def _build_track_result(
        self,
        track: TrackedObject,
        was_created_this_frame: bool = False,
        was_recovered_this_frame: bool = False,
    ) -> TrackResult:
        return TrackResult(
            track_id=track.track_id,
            object_type_token=track.object_type_token,
            bounding_box=track.most_recent_bounding_box,
            lifecycle_state=track.lifecycle_state,
            consecutive_missed_frame_count=track.consecutive_missed_frame_count,
            was_created_this_frame=was_created_this_frame,
            was_recovered_this_frame=was_recovered_this_frame,
        )
