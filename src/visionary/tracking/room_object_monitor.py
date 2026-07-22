"""Coordinator that turns tracking into room-level object reasoning.

The :class:`MultiObjectTracker` is excellent at the *seen* story -- it follows
objects that move in view, holds them by permanence while they are off screen, and
re-identifies them when they return. This coordinator adds the *unseen* story on
top of it, which is what the room use case needs: panning away, an object being
moved while you were not looking, and you later realizing where it went.

It does so by composing four pieces this package already provides:

  * the tracker, fed a :class:`SalientRegionDetector` so resting big objects are
    discovered without motion,
  * a :class:`CameraPoseTracker`, integrating the tracker's own egomotion estimate
    into a room (world) coordinate frame,
  * a :class:`WorldAnchorMemory`, dropping a persistent anchor for each big object
    in world coordinates and judging, when an anchor's spot returns to view,
    whether the object is still there, and
  * an :class:`ObjectDisplacementReasoner`, pairing an object that *disappeared*
    from its anchor with one that *appeared* elsewhere to conclude it moved.

Anchors are confirmed by *where they are and what they look like*, not by track
id, so an object that is still in its place keeps its anchor even if the tracker
re-numbered it across a long pan -- which stops a re-identification slip from
fabricating a phantom "it moved".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import ObjectDetector
from visionary.features.object_feature_memory import (
    ObservedObjectFeatures,
    compare_observed_object_features,
    describe_observed_object,
)
from visionary.geometry import BoundingBox
from visionary.logging_configuration import get_logger_for_module
from visionary.mapping.world_anchor_memory import WorldAnchorMemory, WorldObjectAnchor
from visionary.tracking.camera_pose_tracker import CameraPoseTracker
from visionary.tracking.multi_object_tracker import (
    FrameProcessingResult,
    MultiObjectTracker,
)
from visionary.tracking.object_displacement_reasoner import (
    DisplacementConclusion,
    ObjectDisplacementReasoner,
)

_logger = get_logger_for_module(__name__)

_FORBIDDEN_PAIR_COST = 1.0e6


@dataclass(frozen=True)
class _PresentObject:
    """One object seen in view this frame, with its world place and look."""

    world_center: tuple[float, float]
    frame_bounding_box: BoundingBox
    feature_signature: ObservedObjectFeatures


@dataclass(frozen=True)
class MissingObjectMarker:
    """An anchored object now believed missing, drawn where it used to be."""

    object_identity: str
    world_center: tuple[float, float]
    frame_bounding_box: BoundingBox | None


@dataclass
class RoomSceneObservation:
    """Everything the room monitor concludes about one processed frame."""

    frame_processing_result: FrameProcessingResult
    world_offset: tuple[float, float]
    missing_object_markers: list[MissingObjectMarker] = field(default_factory=list)
    newly_concluded_displacements: list[DisplacementConclusion] = field(default_factory=list)
    all_displacement_conclusions: list[DisplacementConclusion] = field(default_factory=list)


class RoomObjectMonitor:
    """Tracks a room and reasons about objects moved while out of view."""

    def __init__(
        self, object_detector: ObjectDetector, configuration: TrackerConfiguration
    ) -> None:
        self._configuration = configuration
        self._tracker = MultiObjectTracker(object_detector, configuration)
        self._camera_pose_tracker = CameraPoseTracker()
        self._anchor_memory = WorldAnchorMemory(configuration)
        self._displacement_reasoner = ObjectDisplacementReasoner(configuration)
        self._frame_index = 0
        # Anchors are keyed by an identity the monitor mints itself, never by a
        # tracker track id. The tracker's own appearance re-identification would
        # happily collapse two identical objects (the bed pillow and its twin on
        # the floor) onto one track id -- reusing that as an anchor key would let a
        # newly appeared object overwrite the very anchor whose disappearance is
        # the point. A monitor-owned identity keeps each anchor distinct.
        self._anchor_identities_assigned = 0

    @property
    def tracker(self) -> MultiObjectTracker:
        return self._tracker

    @property
    def anchor_memory(self) -> WorldAnchorMemory:
        return self._anchor_memory

    def observe_frame(self, frame: np.ndarray) -> RoomSceneObservation:
        self._frame_index += 1
        frame = self._resize_frame_to_configured_size(frame)

        frame_processing_result = self._tracker.process_frame(frame)
        self._camera_pose_tracker.update_with_frame_to_frame_homography(
            self._tracker.most_recent_camera_motion_homography, frame.shape
        )
        world_offset = self._camera_pose_tracker.current_world_offset

        present_objects = self._collect_present_objects(frame, frame_processing_result)
        self._confirm_or_create_anchors(present_objects)
        missing_object_markers = self._refute_in_view_anchors_not_confirmed(world_offset)
        newly_concluded_displacements = (
            self._displacement_reasoner.reconcile_into_displacement_conclusions(self._frame_index)
        )

        return RoomSceneObservation(
            frame_processing_result=frame_processing_result,
            world_offset=world_offset,
            missing_object_markers=missing_object_markers,
            newly_concluded_displacements=newly_concluded_displacements,
            all_displacement_conclusions=self._displacement_reasoner.displacement_conclusions,
        )

    # -- present objects ------------------------------------------------------

    def _collect_present_objects(
        self, frame: np.ndarray, frame_processing_result: FrameProcessingResult
    ) -> list[_PresentObject]:
        """Big objects actually visible this frame, taken from the detector.

        Presence is read from the detector's regions, not from the focus tracker.
        Resting furniture is often a flat, featureless colour that defeats the
        template cross-correlation the focus layer relies on, yet the detector
        still sees it every frame; using the detector is what lets a plain pillow
        confirm its anchor.
        """
        present_objects: list[_PresentObject] = []
        for detection_bounding_box in frame_processing_result.detection_bounding_boxes:
            if not self._box_is_wholly_in_view(detection_bounding_box):
                # A detection clipped by a frame edge has a drifted, untrustworthy
                # box, so it must not maintain an anchor -- it is in the ambiguous
                # border zone where no present/absent verdict can be made fairly.
                continue
            feature_signature = describe_observed_object(
                self._extract_image_patch(frame, detection_bounding_box),
                detection_bounding_box,
                frame.shape,
            )
            if feature_signature is None:
                continue
            frame_center = self._box_center(detection_bounding_box)
            world_center = self._camera_pose_tracker.world_position_of(*frame_center)
            present_objects.append(
                _PresentObject(
                    world_center=world_center,
                    frame_bounding_box=detection_bounding_box,
                    feature_signature=feature_signature,
                )
            )
        return present_objects

    # -- anchor confirmation / creation ---------------------------------------

    def _confirm_or_create_anchors(self, present_objects: list[_PresentObject]) -> None:
        """Match present objects to anchors by place + look; create anchors for the rest."""
        existing_anchors = self._anchor_memory.all_anchors
        matched_anchor_index_by_object_index = self._match_present_objects_to_anchors(
            present_objects, existing_anchors
        )

        matched_object_indices = set(matched_anchor_index_by_object_index.keys())
        for object_index, anchor_index in matched_anchor_index_by_object_index.items():
            present_object = present_objects[object_index]
            anchor = existing_anchors[anchor_index]
            self._anchor_memory.confirm_anchor(
                anchor.object_identity,
                present_object.world_center,
                present_object.feature_signature,
                self._frame_index,
            )

        for object_index, present_object in enumerate(present_objects):
            if object_index in matched_object_indices:
                continue
            self._register_new_anchor_for(present_object)

    def _match_present_objects_to_anchors(
        self,
        present_objects: list[_PresentObject],
        existing_anchors: list[WorldObjectAnchor],
    ) -> dict[int, int]:
        """Optimally pair present objects to anchors by world distance + feature.

        Returns ``{present_object_index: anchor_index}`` for accepted pairings. A
        pairing is forbidden unless the object sits within the confirmation
        distance of the anchor and looks similar enough to be the same thing.
        """
        if not present_objects or not existing_anchors:
            return {}

        maximum_distance = self._configuration.anchor_confirmation_maximum_world_distance_in_pixels
        minimum_similarity = self._configuration.anchor_confirmation_minimum_feature_similarity
        cost_matrix = np.full(
            (len(present_objects), len(existing_anchors)), _FORBIDDEN_PAIR_COST, dtype=np.float64
        )
        for object_index, present_object in enumerate(present_objects):
            for anchor_index, anchor in enumerate(existing_anchors):
                world_distance = float(
                    np.hypot(
                        present_object.world_center[0] - anchor.world_center_x,
                        present_object.world_center[1] - anchor.world_center_y,
                    )
                )
                if world_distance > maximum_distance:
                    continue
                feature_similarity = compare_observed_object_features(
                    present_object.feature_signature, anchor.feature_signature
                )
                if feature_similarity < minimum_similarity:
                    continue
                cost_matrix[object_index, anchor_index] = world_distance

        matched_object_rows, matched_anchor_columns = linear_sum_assignment(cost_matrix)
        accepted: dict[int, int] = {}
        for object_index, anchor_index in zip(matched_object_rows, matched_anchor_columns):
            if cost_matrix[object_index, anchor_index] >= _FORBIDDEN_PAIR_COST:
                continue
            accepted[int(object_index)] = int(anchor_index)
        return accepted

    def _register_new_anchor_for(self, present_object: _PresentObject) -> None:
        """Anchor a freshly seen object; after priming, flag it as an appearance."""
        anchor_identity = self._mint_anchor_identity()
        self._anchor_memory.register_anchor(
            anchor_identity,
            present_object.feature_signature,
            present_object.world_center,
            (present_object.frame_bounding_box[2], present_object.frame_bounding_box[3]),
            self._frame_index,
        )
        if self._frame_index > self._configuration.scene_priming_frame_count:
            self._displacement_reasoner.record_object_appeared(
                anchor_identity,
                present_object.feature_signature,
                present_object.world_center,
                self._frame_index,
            )
            _logger.info(
                "New object %s appeared at world (%.0f, %.0f) on frame %d",
                anchor_identity,
                present_object.world_center[0],
                present_object.world_center[1],
                self._frame_index,
            )

    def _mint_anchor_identity(self) -> str:
        self._anchor_identities_assigned += 1
        return f"room-object-{self._anchor_identities_assigned}"

    # -- refutation -----------------------------------------------------------

    def _refute_in_view_anchors_not_confirmed(
        self, world_offset: tuple[float, float]
    ) -> list[MissingObjectMarker]:
        """Penalize in-view anchors no present object explains; report the missing.

        Presence has already been folded into anchor state by
        :meth:`_confirm_or_create_anchors` (it stamps every re-found anchor's
        ``last_confirmed_frame_index`` to this frame), so this pass needs only the
        anchors, not the present-object list -- any in-view anchor not stamped this
        frame is unexplained and therefore a refutation.
        """
        confirmed_identities = {
            anchor.object_identity
            for anchor in self._anchor_memory.all_anchors
            if anchor.last_confirmed_frame_index == self._frame_index
        }
        missing_object_markers: list[MissingObjectMarker] = []
        for anchor in self._anchor_memory.anchors_expected_in_view(
            world_offset, self._configuration.frame_width_in_pixels, self._configuration.frame_height_in_pixels
        ):
            if anchor.object_identity in confirmed_identities:
                continue
            declared_missing = self._anchor_memory.refute_anchor(anchor.object_identity)
            if not declared_missing:
                continue
            self._displacement_reasoner.record_object_disappeared(
                anchor.object_identity,
                anchor.feature_signature,
                anchor.world_center,
                self._frame_index,
            )
            missing_object_markers.append(
                MissingObjectMarker(
                    object_identity=anchor.object_identity,
                    world_center=anchor.world_center,
                    frame_bounding_box=self._anchor_frame_box(anchor, world_offset),
                )
            )
            self._anchor_memory.discard_anchor(anchor.object_identity)
            _logger.info(
                "Anchored object %s declared missing at world (%.0f, %.0f) on frame %d",
                anchor.object_identity,
                anchor.world_center_x,
                anchor.world_center_y,
                self._frame_index,
            )
        return missing_object_markers

    def _anchor_frame_box(
        self, anchor: WorldObjectAnchor, world_offset: tuple[float, float]
    ) -> BoundingBox:
        frame_center_x = anchor.world_center_x - world_offset[0]
        frame_center_y = anchor.world_center_y - world_offset[1]
        return (
            int(frame_center_x - anchor.box_width / 2),
            int(frame_center_y - anchor.box_height / 2),
            anchor.box_width,
            anchor.box_height,
        )

    # -- small helpers --------------------------------------------------------

    def _resize_frame_to_configured_size(self, frame: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _extract_image_patch(frame: np.ndarray, bounding_box: BoundingBox) -> np.ndarray:
        left_x, top_y, width, height = bounding_box
        return frame[top_y : top_y + height, left_x : left_x + width]

    @staticmethod
    def _box_center(bounding_box: BoundingBox) -> tuple[float, float]:
        left_x, top_y, width, height = bounding_box
        return left_x + width / 2.0, top_y + height / 2.0

    def _box_is_wholly_in_view(self, bounding_box: BoundingBox) -> bool:
        """True when the box clears every frame edge by the anchor border margin."""
        border_margin = self._configuration.anchor_in_view_border_margin_in_pixels
        left_x, top_y, width, height = bounding_box
        return (
            left_x >= border_margin
            and top_y >= border_margin
            and left_x + width <= self._configuration.frame_width_in_pixels - border_margin
            and top_y + height <= self._configuration.frame_height_in_pixels - border_margin
        )
