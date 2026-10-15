"""The multi-object tracking orchestrator (two-layer design).

Each call to :meth:`process_frame` runs two complementary layers:

  * **Focus layer (first).** For every object already being tracked, the
    anticipatory focus tracker looks *only* inside a small window around the
    Kalman-predicted position and re-locates the object there by appearance,
    biased toward its predicted motion. This is what preserves identity straight
    through a collision: each track follows the blob continuing in *its* own
    direction, and when two tracks would land on the same spot an occlusion
    ordering lets the stronger keep the visible blob while the other coasts on
    prediction and re-acquires its object once they separate. Every located or
    coasting track also *claims* the region it occupies.

  * **Filter layer (second).** The injected detector cheaply scans the whole
    frame for big changes. Any change *not* already claimed by a focus-tracked
    object is inspected as a candidate new object: it is first offered to the
    lost tracks for re-identification, and otherwise spawns a new track.

Crucially, this class performs **no drawing** -- it returns a structured
:class:`FrameProcessingResult`, which the visualization layer consumes
separately. That separation is what makes the tracker runnable headless and
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import ObjectDetector
from visionary.features.appearance_traits import extract_appearance_trait_vector
from visionary.features.object_feature_memory import describe_observed_object
from visionary.geometry import (
    BoundingBox,
    OffScreenExpectation,
    compute_off_screen_expectation,
    intersection_over_union,
)
from visionary.logging_configuration import get_logger_for_module
from visionary.tracking.anticipatory_focus_tracker import (
    AnticipatoryFocusTracker,
    PredictedTrackState,
)
from visionary.tracking.association import associate_detections_to_tracks
from visionary.tracking.camera_motion_compensation import (
    estimate_camera_motion_homography,
)
from visionary.tracking.track import TrackedObject, TrackLifecycleState

_logger = get_logger_for_module(__name__)

# A filter-layer change is considered already explained by a focus-tracked object
# when this much of a claimed region falls inside the change (covers the case of
# two objects merging into one big motion blob during a collision).
_CLAIMED_REGION_CONTAINMENT_THRESHOLD = 0.5


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
    # Object permanence: when the object's expected position is outside the frame
    # the track is still reported, flagged out-of-view, with a pointer to where it
    # is expected to be (the arrow + bearing the visualization draws at the edge).
    is_out_of_view: bool = False
    off_screen_expectation: OffScreenExpectation | None = None


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
        self._focus_layer = AnticipatoryFocusTracker(configuration)
        self._active_tracks_by_id: dict[str, TrackedObject] = {}
        self._lost_tracks_by_id: dict[str, TrackedObject] = {}
        self._previous_grayscale_frame: np.ndarray | None = None
        self._frame_index = 0
        # The most recent frame-to-frame egomotion homography this tracker
        # estimated (``None`` when compensation is disabled or estimation failed).
        # Exposed so a higher layer can reuse it -- e.g. to integrate camera pose
        # for a room map -- without paying for a second ORB pass.
        self._most_recent_camera_motion_homography: np.ndarray | None = None

    # -- public API -----------------------------------------------------------

    @property
    def active_track_count(self) -> int:
        return len(self._active_tracks_by_id)

    @property
    def most_recent_camera_motion_homography(self) -> np.ndarray | None:
        """The last frame-to-frame egomotion homography, for reuse by a room map."""
        return self._most_recent_camera_motion_homography

    def register_object_of_interest(
        self, bounding_box: BoundingBox, frame: np.ndarray
    ) -> str:
        """Designate an object to track from now on, seeding its identity and memory.

        Used when objects are pointed out rather than discovered (e.g. a static
        painting a panning camera should keep track of). The new track is seeded
        with the object's appearance template and intrinsic feature memory so the
        focus layer can hold it and so it can be re-recognised after it leaves view.
        """
        frame = self._resize_frame_if_needed(frame)
        appearance_trait_vector = extract_appearance_trait_vector(frame, bounding_box)
        registered_track = TrackedObject(bounding_box, self._frame_index, appearance_trait_vector)
        registered_track.refresh_clean_appearance_template(
            self._extract_image_patch(frame, bounding_box)
        )
        self._record_clean_feature_observation(registered_track, frame, bounding_box)
        self._active_tracks_by_id[registered_track.track_id] = registered_track
        _logger.info(
            "Registered object of interest as track %s at frame %d",
            registered_track.track_id,
            self._frame_index,
        )
        return registered_track.track_id

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

        track_results: list[TrackResult] = []
        claimed_region_bounding_boxes: list[BoundingBox] = []

        # Snapshot the lost tracks *before* the focus layer runs, so a track that
        # is demoted to LOST this frame is not also predicted again for recovery.
        lost_tracks_at_frame_start = list(self._lost_tracks_by_id.values())

        # --- Focus layer: re-locate active tracks in their predicted windows ---
        self._run_focus_layer_on_active_tracks(
            frame,
            camera_motion_homography,
            track_results,
            claimed_region_bounding_boxes,
        )

        # Predict the pre-existing lost tracks so the filter layer can re-identify
        # them by position as well as by appearance.
        lost_track_prediction = self._predict_tracks(
            lost_tracks_at_frame_start, camera_motion_homography
        )

        # --- Filter layer: cheap global change-catch for new / re-entering ----
        detections = self._object_detector.detect_objects_in_frame(frame)
        detection_bounding_boxes = [detection.bounding_box for detection in detections]
        unclaimed_detection_indices = [
            detection_index
            for detection_index, detection_bounding_box in enumerate(detection_bounding_boxes)
            if not self._detection_is_already_claimed(
                detection_bounding_box, claimed_region_bounding_boxes
            )
        ]
        # Describe each unclaimed change once and reuse it for recovery and spawning.
        detection_appearance_trait_vector_by_index = {
            detection_index: extract_appearance_trait_vector(
                frame, detection_bounding_boxes[detection_index]
            )
            for detection_index in unclaimed_detection_indices
        }

        unclaimed_detection_indices = self._attempt_lost_track_recovery(
            frame,
            lost_tracks_at_frame_start,
            lost_track_prediction,
            detection_bounding_boxes,
            detection_appearance_trait_vector_by_index,
            unclaimed_detection_indices,
            track_results,
        )
        # Recognise a returning object by its remembered features even when its
        # predicted position is no longer where it really is.
        unclaimed_detection_indices = self._attempt_appearance_reidentification(
            frame,
            detection_bounding_boxes,
            detection_appearance_trait_vector_by_index,
            unclaimed_detection_indices,
            track_results,
        )
        self._spawn_new_tracks_for_unmatched_detections(
            frame,
            detection_bounding_boxes,
            detection_appearance_trait_vector_by_index,
            unclaimed_detection_indices,
            track_results,
        )
        # Collapse any tracks that have converged onto the same object so the
        # active-track count cannot drift above the number of real objects.
        track_results = self._merge_redundant_coincident_active_tracks(track_results)
        # Age the lost tracks that were not recovered this frame so their
        # prediction advances and they eventually retire instead of lingering.
        self._coast_unrecovered_lost_tracks(lost_tracks_at_frame_start)
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

    # -- focus layer ----------------------------------------------------------

    def _run_focus_layer_on_active_tracks(
        self,
        frame: np.ndarray,
        camera_motion_homography: np.ndarray | None,
        track_results: list[TrackResult],
        claimed_region_bounding_boxes: list[BoundingBox],
    ) -> None:
        active_tracks = list(self._active_tracks_by_id.values())
        if not active_tracks:
            return

        predicted_track_states: list[PredictedTrackState] = []
        anticipated_bounding_box_by_track_id: dict[str, BoundingBox] = {}
        predicted_center_by_track_id: dict[str, tuple[float, float]] = {}
        for active_track in active_tracks:
            if camera_motion_homography is not None:
                active_track.apply_camera_motion_warp(camera_motion_homography)
            (
                predicted_bounding_box,
                predicted_center_x,
                predicted_center_y,
                _innovation_covariance,
            ) = active_track.predict_bounding_box()
            anticipated_bounding_box_by_track_id[active_track.track_id] = (
                self._clamp_bounding_box_to_frame(predicted_bounding_box, frame.shape)
            )
            predicted_center_by_track_id[active_track.track_id] = (
                predicted_center_x,
                predicted_center_y,
            )
            predicted_track_states.append(
                PredictedTrackState(
                    track_id=active_track.track_id,
                    predicted_center_x=predicted_center_x,
                    predicted_center_y=predicted_center_y,
                    predicted_bounding_box=predicted_bounding_box,
                    clean_appearance_template_patch=active_track.clean_appearance_template_patch,
                    # Widen the search the longer the track has been out of view,
                    # to absorb the prediction drift accumulated with no measurements.
                    search_region_expansion_in_pixels=min(
                        active_track.consecutive_off_frame_count, 140
                    ),
                )
            )

        focus_placements = self._focus_layer.locate_tracks_within_search_regions(
            frame, predicted_track_states
        )
        focus_placement_by_track_id = {
            focus_placement.track_id: focus_placement
            for focus_placement in focus_placements
        }

        for active_track in active_tracks:
            focus_placement = focus_placement_by_track_id[active_track.track_id]
            missed_frame_count_before_update = active_track.consecutive_missed_frame_count

            if focus_placement.is_confidently_located:
                located_bounding_box = focus_placement.located_bounding_box
                located_appearance_trait_vector = extract_appearance_trait_vector(
                    frame, located_bounding_box
                )
                active_track.update_with_matched_detection(
                    located_bounding_box,
                    self._frame_index,
                    located_appearance_trait_vector,
                    preserve_velocity_through_contention=(
                        focus_placement.is_contesting_a_shared_blob
                    ),
                )
                if (
                    focus_placement.is_clean_unoccluded_view
                    and not focus_placement.is_contesting_a_shared_blob
                    and focus_placement.located_appearance_patch is not None
                ):
                    active_track.refresh_clean_appearance_template(
                        focus_placement.located_appearance_patch
                    )
                    self._record_clean_feature_observation(
                        active_track, frame, located_bounding_box
                    )
                claimed_region_bounding_boxes.append(located_bounding_box)
                track_results.append(
                    self._build_track_result(
                        active_track,
                        was_recovered_this_frame=missed_frame_count_before_update > 0,
                    )
                )
            else:
                self._coast_active_track_that_was_not_located(
                    active_track,
                    predicted_center_by_track_id[active_track.track_id],
                    anticipated_bounding_box_by_track_id[active_track.track_id],
                    frame.shape,
                    track_results,
                    claimed_region_bounding_boxes,
                )

    def _coast_active_track_that_was_not_located(
        self,
        active_track: TrackedObject,
        predicted_center: tuple[float, float],
        anticipated_bounding_box: BoundingBox,
        frame_shape: tuple[int, ...],
        track_results: list[TrackResult],
        claimed_region_bounding_boxes: list[BoundingBox],
    ) -> None:
        """Coast an unlocated active track, treating off-frame ones as out-of-view.

        An object whose expected position is outside the frame has not been *lost*
        -- it is exactly where a moving object (or, later, a panned-away camera)
        would put it. Such a track is held by object permanence and reported with a
        pointer to where it is expected, instead of being retired like an in-view
        miss.
        """
        frame_height, frame_width = frame_shape[:2]
        predicted_center_x, predicted_center_y = predicted_center
        raw_predicted_bounding_box = (
            int(predicted_center_x - active_track.bounding_box_width / 2),
            int(predicted_center_y - active_track.bounding_box_height / 2),
            active_track.bounding_box_width,
            active_track.bounding_box_height,
        )
        fraction_inside_frame = self._fraction_of_box_inside_frame(
            raw_predicted_bounding_box, frame_shape
        )
        track_is_leaving_the_frame = fraction_inside_frame < 0.5

        if not track_is_leaving_the_frame:
            # Genuinely in view but not located: a normal in-frame miss.
            self._coast_and_maybe_demote_track(active_track, anticipated_bounding_box)
            if active_track.lifecycle_state == TrackLifecycleState.ACTIVE:
                claimed_region_bounding_boxes.append(active_track.most_recent_bounding_box)
                track_results.append(self._build_track_result(active_track))
            return

        # The object is mostly or entirely outside the frame. It has not been lost
        # -- it is leaving -- so hold it by permanence rather than retiring it.
        active_track.coast_through_missed_frame(anticipated_bounding_box)
        active_track.consecutive_off_frame_count += 1
        if (
            active_track.consecutive_off_frame_count
            > self._configuration.maximum_frames_to_retain_out_of_view_track
        ):
            active_track.lifecycle_state = TrackLifecycleState.LOST
            self._lost_tracks_by_id[active_track.track_id] = self._active_tracks_by_id.pop(
                active_track.track_id
            )
            _logger.info(
                "Out-of-view track %s given up at frame %d",
                active_track.track_id,
                self._frame_index,
            )
            return

        off_screen_expectation = compute_off_screen_expectation(
            predicted_center_x, predicted_center_y, frame_width, frame_height
        )
        if off_screen_expectation is None:
            # Still clipping the edge (partly visible): keep it, no pointer yet.
            claimed_region_bounding_boxes.append(active_track.most_recent_bounding_box)
            track_results.append(self._build_track_result(active_track))
            return
        track_results.append(
            self._build_track_result(
                active_track,
                is_out_of_view=True,
                off_screen_expectation=off_screen_expectation,
            )
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
            self._most_recent_camera_motion_homography = None
            return None
        self._most_recent_camera_motion_homography = estimate_camera_motion_homography(
            self._previous_grayscale_frame, current_grayscale_frame, self._configuration
        )
        return self._most_recent_camera_motion_homography

    def _predict_tracks(
        self,
        tracks: list[TrackedObject],
        camera_motion_homography: np.ndarray | None,
    ) -> MultiObjectTracker._TrackPredictionBundle:
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

    def _coast_and_maybe_demote_track(
        self, track: TrackedObject, anticipated_bounding_box: BoundingBox | None = None
    ) -> None:
        previously_missed_frame_count = track.consecutive_missed_frame_count
        track.coast_through_missed_frame(anticipated_bounding_box)
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
        frame: np.ndarray,
        lost_tracks: list[TrackedObject],
        lost_track_prediction: MultiObjectTracker._TrackPredictionBundle,
        detection_bounding_boxes: list[BoundingBox],
        detection_appearance_trait_vector_by_index: dict[int, np.ndarray],
        unclaimed_detection_indices: list[int],
        track_results: list[TrackResult],
    ) -> list[int]:
        if not unclaimed_detection_indices or not lost_tracks:
            return unclaimed_detection_indices

        leftover_detection_bounding_boxes = [
            detection_bounding_boxes[detection_index]
            for detection_index in unclaimed_detection_indices
        ]
        leftover_detection_appearance_trait_vectors = [
            detection_appearance_trait_vector_by_index[detection_index]
            for detection_index in unclaimed_detection_indices
        ]

        matched_lost_pairs, remaining_leftover_indices, _ = associate_detections_to_tracks(
            lost_track_prediction.predicted_bounding_boxes,
            leftover_detection_bounding_boxes,
            lost_track_prediction.predicted_centers,
            lost_track_prediction.innovation_covariances,
            lost_track_prediction.appearance_trait_vectors,
            leftover_detection_appearance_trait_vectors,
            self._configuration.lost_track_intersection_over_union_gate_threshold,
            self._configuration.lost_track_mahalanobis_distance_gate_threshold,
            self._configuration.intersection_over_union_score_weight,
            self._configuration.mahalanobis_distance_score_weight,
            self._configuration.appearance_similarity_score_weight,
            self._configuration.mahalanobis_distance_score_normalization,
        )

        for lost_track_index, leftover_detection_index in matched_lost_pairs:
            original_detection_index = unclaimed_detection_indices[leftover_detection_index]
            recovered_track = lost_tracks[lost_track_index]
            recovered_bounding_box = detection_bounding_boxes[original_detection_index]
            recovered_track.update_with_matched_detection(
                recovered_bounding_box,
                self._frame_index,
                detection_appearance_trait_vector_by_index[original_detection_index],
            )
            recovered_track.refresh_clean_appearance_template(
                self._extract_image_patch(frame, recovered_bounding_box)
            )
            self._record_clean_feature_observation(
                recovered_track, frame, recovered_bounding_box
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
            unclaimed_detection_indices[leftover_index]
            for leftover_index in remaining_leftover_indices
        ]

    def _spawn_new_tracks_for_unmatched_detections(
        self,
        frame: np.ndarray,
        detection_bounding_boxes: list[BoundingBox],
        detection_appearance_trait_vector_by_index: dict[int, np.ndarray],
        unclaimed_detection_indices: list[int],
        track_results: list[TrackResult],
    ) -> None:
        for detection_index in unclaimed_detection_indices:
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
                detection_appearance_trait_vector_by_index[detection_index],
            )
            newly_created_track.refresh_clean_appearance_template(
                self._extract_image_patch(frame, detection_bounding_box)
            )
            self._record_clean_feature_observation(
                newly_created_track, frame, detection_bounding_box
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

    def _merge_redundant_coincident_active_tracks(
        self, track_results: list[TrackResult]
    ) -> list[TrackResult]:
        """Cull active tracks that have converged onto the same object.

        Two active tracks are duplicates when their boxes overlap heavily *and*
        they are moving together; the younger track is removed. Because velocity
        agreement is required, two distinct objects crossing (which move in
        opposite directions) are never merged however much their boxes overlap.
        Returns the track results with culled tracks removed.
        """
        active_tracks = list(self._active_tracks_by_id.values())
        culled_track_ids: set[str] = set()
        for first_position in range(len(active_tracks)):
            first_track = active_tracks[first_position]
            if first_track.track_id in culled_track_ids:
                continue
            for second_position in range(first_position + 1, len(active_tracks)):
                second_track = active_tracks[second_position]
                if second_track.track_id in culled_track_ids:
                    continue
                if not self._two_active_tracks_are_duplicates(first_track, second_track):
                    continue
                younger_track, older_track = self._order_tracks_by_seniority(
                    first_track, second_track
                )
                culled_track_ids.add(younger_track.track_id)
                _logger.info(
                    "Track %s merged into %s at frame %d (duplicate of one object)",
                    younger_track.track_id,
                    older_track.track_id,
                    self._frame_index,
                )
                if younger_track.track_id == first_track.track_id:
                    # The surviving partner changed, so stop comparing against the
                    # now-culled first track and move on to the next one.
                    break

        for culled_track_id in culled_track_ids:
            self._active_tracks_by_id.pop(culled_track_id, None)
        return [
            track_result
            for track_result in track_results
            if track_result.track_id not in culled_track_ids
        ]

    def _two_active_tracks_are_duplicates(
        self, first_track: TrackedObject, second_track: TrackedObject
    ) -> bool:
        bounding_box_overlap = intersection_over_union(
            first_track.most_recent_bounding_box, second_track.most_recent_bounding_box
        )
        if bounding_box_overlap < self._configuration.merge_coincident_track_minimum_overlap:
            return False

        first_velocity_x, first_velocity_y = first_track.current_velocity
        second_velocity_x, second_velocity_y = second_track.current_velocity
        first_speed = float(np.hypot(first_velocity_x, first_velocity_y))
        second_speed = float(np.hypot(second_velocity_x, second_velocity_y))
        if (
            min(first_speed, second_speed)
            < self._configuration.merge_coincident_track_minimum_speed
        ):
            # Direction is meaningless for a near-stationary track; require a very
            # high overlap before declaring a merge.
            return (
                bounding_box_overlap
                >= self._configuration.merge_coincident_track_high_overlap_fallback
            )

        velocity_cosine_similarity = (
            first_velocity_x * second_velocity_x + first_velocity_y * second_velocity_y
        ) / (first_speed * second_speed)
        return (
            velocity_cosine_similarity
            >= self._configuration.merge_coincident_track_velocity_agreement
        )

    @staticmethod
    def _order_tracks_by_seniority(
        first_track: TrackedObject, second_track: TrackedObject
    ) -> tuple[TrackedObject, TrackedObject]:
        """Return ``(younger_track, older_track)`` -- the younger one is culled.

        Seniority prefers the earlier-created track, breaking ties by the longer
        observation history.
        """
        first_is_younger = (
            first_track.creation_frame_index,
            -first_track.observed_frame_count,
        ) > (
            second_track.creation_frame_index,
            -second_track.observed_frame_count,
        )
        if first_is_younger:
            return first_track, second_track
        return second_track, first_track

    def _coast_unrecovered_lost_tracks(
        self, lost_tracks_at_frame_start: list[TrackedObject]
    ) -> None:
        """Advance and age every lost track that was not recovered this frame.

        Lost tracks are predicted each frame so they can be re-identified by
        position, but unless they are also coasted their missed-frame counter
        never grows -- so they would never advance their estimate nor retire.
        """
        for lost_track in lost_tracks_at_frame_start:
            if lost_track.track_id in self._lost_tracks_by_id:
                lost_track.coast_through_missed_frame()

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
            #_logger.info(
            #    "Collision or occlusion detected at frame %d", self._frame_index
            #)
            return True
        return False

    def _detection_is_already_claimed(
        self,
        detection_bounding_box: BoundingBox,
        claimed_region_bounding_boxes: list[BoundingBox],
    ) -> bool:
        """True if a focus-tracked object already explains this filter-layer change.

        A change counts as claimed if it overlaps a claimed region above the
        association IoU gate, or if either box is largely contained inside the
        other. The two containment directions matter for different collision
        artefacts: a claimed box inside the change is the merged-blob case, while
        the change inside a claimed box is a small fragment shed during a collision
        -- which must not be allowed to spawn a spurious (too-small) new track.
        """
        for claimed_region_bounding_box in claimed_region_bounding_boxes:
            if (
                intersection_over_union(detection_bounding_box, claimed_region_bounding_box)
                > self._configuration.intersection_over_union_gate_threshold
            ):
                return True
            claimed_region_inside_detection = self._fraction_of_first_box_inside_second(
                claimed_region_bounding_box, detection_bounding_box
            )
            detection_inside_claimed_region = self._fraction_of_first_box_inside_second(
                detection_bounding_box, claimed_region_bounding_box
            )
            if (
                claimed_region_inside_detection > _CLAIMED_REGION_CONTAINMENT_THRESHOLD
                or detection_inside_claimed_region > _CLAIMED_REGION_CONTAINMENT_THRESHOLD
            ):
                return True
        return False

    @staticmethod
    def _fraction_of_first_box_inside_second(
        first_bounding_box: BoundingBox, second_bounding_box: BoundingBox
    ) -> float:
        first_left_x, first_top_y, first_width, first_height = first_bounding_box
        second_left_x, second_top_y, second_width, second_height = second_bounding_box
        intersection_left_x = max(first_left_x, second_left_x)
        intersection_top_y = max(first_top_y, second_top_y)
        intersection_right_x = min(
            first_left_x + first_width, second_left_x + second_width
        )
        intersection_bottom_y = min(
            first_top_y + first_height, second_top_y + second_height
        )
        intersection_width = max(0, intersection_right_x - intersection_left_x)
        intersection_height = max(0, intersection_bottom_y - intersection_top_y)
        intersection_area = intersection_width * intersection_height
        first_box_area = first_width * first_height
        if first_box_area <= 0:
            return 0.0
        return intersection_area / float(first_box_area)

    def _fraction_of_box_inside_frame(
        self, bounding_box: BoundingBox, frame_shape: tuple[int, ...]
    ) -> float:
        frame_height, frame_width = frame_shape[:2]
        frame_bounding_box = (0, 0, frame_width, frame_height)
        return self._fraction_of_first_box_inside_second(bounding_box, frame_bounding_box)

    @staticmethod
    def _clamp_bounding_box_to_frame(
        bounding_box: BoundingBox, frame_shape: tuple[int, ...]
    ) -> BoundingBox:
        frame_height, frame_width = frame_shape[:2]
        left_x, top_y, width, height = bounding_box
        clamped_left_x = max(0, min(left_x, frame_width - 1))
        clamped_top_y = max(0, min(top_y, frame_height - 1))
        clamped_width = max(1, min(width, frame_width - clamped_left_x))
        clamped_height = max(1, min(height, frame_height - clamped_top_y))
        return (clamped_left_x, clamped_top_y, clamped_width, clamped_height)

    @staticmethod
    def _extract_image_patch(
        frame: np.ndarray, bounding_box: BoundingBox
    ) -> np.ndarray:
        left_x, top_y, width, height = bounding_box
        return frame[top_y : top_y + height, left_x : left_x + width]

    def _build_track_result(
        self,
        track: TrackedObject,
        was_created_this_frame: bool = False,
        was_recovered_this_frame: bool = False,
        is_out_of_view: bool = False,
        off_screen_expectation: OffScreenExpectation | None = None,
    ) -> TrackResult:
        return TrackResult(
            track_id=track.track_id,
            object_type_token=track.object_type_token,
            bounding_box=track.most_recent_bounding_box,
            lifecycle_state=track.lifecycle_state,
            consecutive_missed_frame_count=track.consecutive_missed_frame_count,
            was_created_this_frame=was_created_this_frame,
            was_recovered_this_frame=was_recovered_this_frame,
            is_out_of_view=is_out_of_view,
            off_screen_expectation=off_screen_expectation,
        )

    def _record_clean_feature_observation(
        self, track: TrackedObject, frame: np.ndarray, bounding_box: BoundingBox
    ) -> None:
        """Fold a clean in-view observation into the track's intrinsic feature memory."""
        observed_features = describe_observed_object(
            self._extract_image_patch(frame, bounding_box), bounding_box, frame.shape
        )
        if observed_features is not None:
            track.update_feature_memory(observed_features)

    def _attempt_appearance_reidentification(
        self,
        frame: np.ndarray,
        detection_bounding_boxes: list[BoundingBox],
        detection_appearance_trait_vector_by_index: dict[int, np.ndarray],
        unclaimed_detection_indices: list[int],
        track_results: list[TrackResult],
    ) -> list[int]:
        """Re-bind a returning object to its old id by its remembered features.

        Candidates are objects currently out of view or lost -- the ones whose
        predicted position can no longer be trusted. A detection that no in-view
        track explains is compared to each candidate's feature memory; a strong
        enough partial match restores that object's identity instead of spawning a
        new track. This is what carries the same id across a disappearance.
        """
        if not unclaimed_detection_indices:
            return unclaimed_detection_indices

        candidate_tracks = [
            track
            for track in self._active_tracks_by_id.values()
            if track.consecutive_off_frame_count > 0 and track.feature_memory.has_been_observed
        ] + [
            track
            for track in self._lost_tracks_by_id.values()
            if track.feature_memory.has_been_observed
        ]
        if not candidate_tracks:
            return unclaimed_detection_indices

        remaining_detection_indices: list[int] = []
        already_reidentified_track_ids: set[str] = set()
        for detection_index in unclaimed_detection_indices:
            detection_bounding_box = detection_bounding_boxes[detection_index]
            observed_features = describe_observed_object(
                self._extract_image_patch(frame, detection_bounding_box),
                detection_bounding_box,
                frame.shape,
            )
            best_matching_track = None
            best_similarity = self._configuration.appearance_reidentification_minimum_similarity
            if observed_features is not None:
                for candidate_track in candidate_tracks:
                    if candidate_track.track_id in already_reidentified_track_ids:
                        continue
                    similarity = candidate_track.feature_memory.recognition_similarity_to(
                        observed_features
                    )
                    if similarity >= best_similarity:
                        best_similarity = similarity
                        best_matching_track = candidate_track

            if best_matching_track is None:
                remaining_detection_indices.append(detection_index)
                continue

            best_matching_track.update_with_matched_detection(
                detection_bounding_box,
                self._frame_index,
                detection_appearance_trait_vector_by_index[detection_index],
            )
            best_matching_track.refresh_clean_appearance_template(
                self._extract_image_patch(frame, detection_bounding_box)
            )
            self._record_clean_feature_observation(
                best_matching_track, frame, detection_bounding_box
            )
            if best_matching_track.track_id in self._lost_tracks_by_id:
                self._active_tracks_by_id[best_matching_track.track_id] = (
                    self._lost_tracks_by_id.pop(best_matching_track.track_id)
                )
            already_reidentified_track_ids.add(best_matching_track.track_id)
            _logger.info(
                "Track %s re-identified by appearance at frame %d (similarity %.2f)",
                best_matching_track.track_id,
                self._frame_index,
                best_similarity,
            )
            # Replace any earlier (out-of-view) result for this track with the
            # in-view re-identification so each track appears once per frame.
            track_results[:] = [
                track_result
                for track_result in track_results
                if track_result.track_id != best_matching_track.track_id
            ]
            track_results.append(
                self._build_track_result(best_matching_track, was_recovered_this_frame=True)
            )

        return remaining_detection_indices
