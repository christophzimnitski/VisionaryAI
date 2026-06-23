"""The focus layer: anticipatory, per-track local search.

While the *filter layer* watches the whole frame cheaply for big changes (new
objects entering), the focus layer does the opposite: for every object already
being tracked it looks **only** inside a small window around where the Kalman
filter predicts that object will be. Inside that window it re-locates the object
by normalized cross-correlation against the object's last clean appearance,
biased toward the predicted trajectory.

Two design choices make this hold identity straight through a collision:

* **Motion-biased matching.** When two objects look identical (so appearance
  cannot tell them apart), the match score is combined with how well a candidate
  position agrees with the track's own predicted motion. Each track therefore
  locks onto the blob continuing in *its* direction.
* **Occlusion ordering.** If two tracks would land on the same spot, only the
  higher-scoring one occupies the visible blob; the other is declared occluded
  and coasts on prediction alone, re-acquiring its object when they separate.

This is "track-before-detect with attentional regions": the cost depends on the
window size, not the frame size, which is what makes it cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration
from visionary.geometry import BoundingBox


@dataclass(frozen=True)
class PredictedTrackState:
    """One track's Kalman prediction, handed to the focus layer to search around."""

    track_id: str
    predicted_center_x: float
    predicted_center_y: float
    predicted_bounding_box: BoundingBox
    clean_appearance_template_patch: np.ndarray | None
    # Extra search-window margin reflecting how uncertain the prediction is. It
    # grows while a track is out of view, because the predicted position drifts
    # with no measurements to correct it, so a returning object must be searched
    # for over a wider area.
    search_region_expansion_in_pixels: int = 0


@dataclass
class FocusPlacement:
    """Where the focus layer believes one track's object is in the current frame."""

    track_id: str
    located_bounding_box: BoundingBox
    appearance_match_confidence: float
    is_confidently_located: bool
    is_clean_unoccluded_view: bool
    located_appearance_patch: np.ndarray | None
    # True when this (winning) placement shares its blob with another track, so
    # the object is overlapping a second one and its velocity must be protected.
    is_contesting_a_shared_blob: bool = False
    # Score used only to rank tracks for occlusion ordering (appearance + motion).
    _combined_ranking_score: float = 0.0


class AnticipatoryFocusTracker:
    """Re-locates already-tracked objects inside small predicted search windows."""

    def __init__(self, configuration: TrackerConfiguration) -> None:
        self._configuration = configuration

    def locate_tracks_within_search_regions(
        self, frame: np.ndarray, predicted_track_states: list[PredictedTrackState]
    ) -> list[FocusPlacement]:
        """Locate every predicted track inside its own window, resolving overlaps."""
        focus_placements = [
            self._locate_single_track(frame, predicted_track_state)
            for predicted_track_state in predicted_track_states
        ]
        self._resolve_overlapping_placements_by_occlusion_ordering(focus_placements)
        return focus_placements

    # -- single-track search --------------------------------------------------

    def _locate_single_track(
        self, frame: np.ndarray, predicted_track_state: PredictedTrackState
    ) -> FocusPlacement:
        template_patch = predicted_track_state.clean_appearance_template_patch
        clamped_predicted_box = self._clamp_box_to_frame(
            predicted_track_state.predicted_bounding_box, frame.shape
        )

        if template_patch is None or template_patch.size == 0:
            # No template yet (object only just discovered): nothing to match, so
            # the focus layer abstains and the track coasts this frame.
            return self._build_coasting_placement(predicted_track_state, clamped_predicted_box)

        search_region_box = self._build_search_region_box(
            predicted_track_state.predicted_bounding_box,
            frame.shape,
            predicted_track_state.search_region_expansion_in_pixels,
        )
        region_left_x, region_top_y, region_width, region_height = search_region_box
        template_height, template_width = template_patch.shape[:2]
        if region_width < template_width or region_height < template_height:
            # The search window collapsed (object near a border, or it shrank
            # below the template). Abstain rather than force a bad match.
            return self._build_coasting_placement(predicted_track_state, clamped_predicted_box)

        search_region_image = frame[
            region_top_y : region_top_y + region_height,
            region_left_x : region_left_x + region_width,
        ]
        appearance_correlation_map = cv2.matchTemplate(
            search_region_image, template_patch, cv2.TM_CCOEFF_NORMED
        )
        motion_consistency_map = self._build_motion_consistency_map(
            appearance_correlation_map.shape,
            region_left_x,
            region_top_y,
            template_width,
            template_height,
            predicted_track_state.predicted_center_x,
            predicted_track_state.predicted_center_y,
        )
        combined_score_map = (
            self._configuration.focus_match_score_appearance_weight
            * appearance_correlation_map
            + self._configuration.focus_match_score_motion_consistency_weight
            * motion_consistency_map
        )

        best_row, best_column = np.unravel_index(
            int(np.argmax(combined_score_map)), combined_score_map.shape
        )
        raw_appearance_confidence = float(appearance_correlation_map[best_row, best_column])
        combined_ranking_score = float(combined_score_map[best_row, best_column])

        located_left_x = region_left_x + int(best_column)
        located_top_y = region_top_y + int(best_row)
        located_bounding_box = (located_left_x, located_top_y, template_width, template_height)
        located_appearance_patch = frame[
            located_top_y : located_top_y + template_height,
            located_left_x : located_left_x + template_width,
        ]

        is_confidently_located = (
            raw_appearance_confidence
            >= self._configuration.focus_minimum_template_match_confidence
        )
        is_clean_unoccluded_view = (
            raw_appearance_confidence
            >= self._configuration.focus_clean_template_refresh_confidence
        )
        return FocusPlacement(
            track_id=predicted_track_state.track_id,
            located_bounding_box=located_bounding_box,
            appearance_match_confidence=raw_appearance_confidence,
            is_confidently_located=is_confidently_located,
            is_clean_unoccluded_view=is_clean_unoccluded_view,
            located_appearance_patch=located_appearance_patch,
            _combined_ranking_score=combined_ranking_score,
        )

    def _build_coasting_placement(
        self, predicted_track_state: PredictedTrackState, clamped_predicted_box: BoundingBox
    ) -> FocusPlacement:
        return FocusPlacement(
            track_id=predicted_track_state.track_id,
            located_bounding_box=clamped_predicted_box,
            appearance_match_confidence=0.0,
            is_confidently_located=False,
            is_clean_unoccluded_view=False,
            located_appearance_patch=None,
            _combined_ranking_score=0.0,
        )

    # -- occlusion ordering ---------------------------------------------------

    def _resolve_overlapping_placements_by_occlusion_ordering(
        self, focus_placements: list[FocusPlacement]
    ) -> None:
        """Demote the weaker of any two near-coincident placements to "occluded".

        When two tracks resolve to nearly the same point, one object is in front
        of the other. The stronger match keeps the visible blob; the weaker one
        is marked not-confidently-located so it coasts on its prediction and
        re-acquires its object once the two separate.
        """
        separation_threshold = (
            self._configuration.focus_occlusion_center_separation_in_pixels
        )
        confident_placements = sorted(
            (placement for placement in focus_placements if placement.is_confidently_located),
            key=lambda placement: placement._combined_ranking_score,
            reverse=True,
        )

        accepted_placements: list[FocusPlacement] = []
        for placement in confident_placements:
            colliding_stronger_placement = self._find_closer_placement_than(
                placement, accepted_placements, separation_threshold
            )
            if colliding_stronger_placement is not None:
                # The weaker track is behind the stronger one: it coasts on
                # prediction. The stronger track keeps the visible blob but is now
                # contesting it, so its velocity must be protected from coalescence.
                placement.is_confidently_located = False
                placement.is_clean_unoccluded_view = False
                colliding_stronger_placement.is_contesting_a_shared_blob = True
            else:
                accepted_placements.append(placement)

        self._protect_velocity_of_close_but_separate_placements(accepted_placements)

    def _find_closer_placement_than(
        self,
        placement: FocusPlacement,
        candidate_placements: list[FocusPlacement],
        separation_threshold: float,
    ) -> FocusPlacement | None:
        """Return the first candidate whose centre is nearer than the threshold."""
        for candidate_placement in candidate_placements:
            if (
                self._distance_between_placement_centers(placement, candidate_placement)
                < separation_threshold
            ):
                return candidate_placement
        return None

    def _distance_between_placement_centers(
        self, first_placement: FocusPlacement, second_placement: FocusPlacement
    ) -> float:
        first_center_x, first_center_y = self._box_center(first_placement.located_bounding_box)
        second_center_x, second_center_y = self._box_center(second_placement.located_bounding_box)
        return float(np.hypot(first_center_x - second_center_x, first_center_y - second_center_y))

    def _protect_velocity_of_close_but_separate_placements(
        self, accepted_placements: list["FocusPlacement"]
    ) -> None:
        """Flag still-separate tracks that pass close, to protect their velocity.

        Two tracks can be distinct (each on its own blob) yet close enough that a
        normal position correction would corrupt their velocities and let them
        swap objects as they separate. Flagging them as contesting keeps the
        boxes on their blobs while preserving the established motion.
        """
        protection_distance = (
            self._configuration.focus_velocity_protection_center_separation_in_pixels
        )
        for first_index in range(len(accepted_placements)):
            first_placement = accepted_placements[first_index]
            for second_index in range(first_index + 1, len(accepted_placements)):
                second_placement = accepted_placements[second_index]
                if (
                    self._distance_between_placement_centers(first_placement, second_placement)
                    < protection_distance
                ):
                    first_placement.is_contesting_a_shared_blob = True
                    second_placement.is_contesting_a_shared_blob = True

    # -- geometry helpers -----------------------------------------------------

    def _build_search_region_box(
        self,
        predicted_bounding_box: BoundingBox,
        frame_shape: tuple[int, ...],
        extra_expansion_in_pixels: int = 0,
    ) -> BoundingBox:
        left_x, top_y, width, height = predicted_bounding_box
        horizontal_margin = (
            max(
                self._configuration.focus_search_region_minimum_margin_in_pixels,
                int(self._configuration.focus_search_region_margin_ratio * width),
            )
            + extra_expansion_in_pixels
        )
        vertical_margin = (
            max(
                self._configuration.focus_search_region_minimum_margin_in_pixels,
                int(self._configuration.focus_search_region_margin_ratio * height),
            )
            + extra_expansion_in_pixels
        )
        expanded_box = (
            left_x - horizontal_margin,
            top_y - vertical_margin,
            width + 2 * horizontal_margin,
            height + 2 * vertical_margin,
        )
        return self._clamp_box_to_frame(expanded_box, frame_shape)

    @staticmethod
    def _clamp_box_to_frame(
        bounding_box: BoundingBox, frame_shape: tuple[int, ...]
    ) -> BoundingBox:
        frame_height, frame_width = frame_shape[:2]
        left_x, top_y, width, height = bounding_box
        clamped_left_x = max(0, min(left_x, frame_width - 1))
        clamped_top_y = max(0, min(top_y, frame_height - 1))
        clamped_width = max(0, min(width, frame_width - clamped_left_x))
        clamped_height = max(0, min(height, frame_height - clamped_top_y))
        return (clamped_left_x, clamped_top_y, clamped_width, clamped_height)

    def _build_motion_consistency_map(
        self,
        correlation_map_shape: tuple[int, int],
        region_left_x: int,
        region_top_y: int,
        template_width: int,
        template_height: int,
        predicted_center_x: float,
        predicted_center_y: float,
    ) -> np.ndarray:
        """Return a [0, 1] map that is highest where a match agrees with motion.

        Each cell of the correlation map corresponds to a candidate object centre;
        the closer that centre is to the Kalman-predicted centre, the higher the
        returned consistency, falling linearly to zero at the normalization
        distance. This is what lets a track prefer the blob continuing in its own
        direction when appearance is uninformative.
        """
        map_height, map_width = correlation_map_shape
        candidate_center_x = region_left_x + np.arange(map_width) + template_width / 2.0
        candidate_center_y = region_top_y + np.arange(map_height) + template_height / 2.0
        horizontal_offset = candidate_center_x[np.newaxis, :] - predicted_center_x
        vertical_offset = candidate_center_y[:, np.newaxis] - predicted_center_y
        distance_to_prediction = np.sqrt(horizontal_offset**2 + vertical_offset**2)
        normalized_distance = (
            distance_to_prediction
            / self._configuration.focus_motion_consistency_distance_normalization
        )
        return 1.0 - np.clip(normalized_distance, 0.0, 1.0)

    @staticmethod
    def _box_center(bounding_box: BoundingBox) -> tuple[float, float]:
        left_x, top_y, width, height = bounding_box
        return left_x + width / 2.0, top_y + height / 2.0
