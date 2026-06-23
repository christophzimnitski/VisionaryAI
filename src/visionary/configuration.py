"""Central, immutable configuration for the multi-object tracking pipeline.

All tunable parameters live here as a single frozen dataclass so they can be
passed explicitly through the system (dependency injection) instead of being read
from module-level globals. This makes parameter sweeps and per-run overrides easy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackerConfiguration:
    """Every tunable knob the tracking pipeline uses, in one place."""

    # --- Frame geometry -------------------------------------------------------
    frame_width_in_pixels: int = 640
    frame_height_in_pixels: int = 480

    # --- Motion-difference detector ------------------------------------------
    motion_difference_binarization_threshold: int = 25
    motion_dilation_kernel_size: tuple[int, int] = (5, 5)
    motion_dilation_iteration_count: int = 2
    minimum_motion_contour_area_in_pixels: float = 500.0
    maximum_motion_contour_area_in_pixels: float = 30000.0
    # An 11x11 blur denoises the difference image without washing out small
    # frame-to-frame motion. The original 21x21 kernel suppressed slow motion so
    # heavily that genuine moving objects fell just under the contour-area gate.
    gaussian_blur_kernel_size: tuple[int, int] = (11, 11)

    # --- Template-matching detector ------------------------------------------
    template_match_confidence_threshold: float = 0.7
    non_maximum_suppression_overlap_threshold: float = 0.3

    # --- Track lifecycle ------------------------------------------------------
    maximum_consecutive_missed_frames_before_lost: int = 10
    maximum_frames_to_retain_lost_track: int = 60
    # Object permanence: while a track's expected position is *outside* the frame
    # it is held this many frames before being given up, far longer than an
    # in-view miss, because a thing that left the frame is expected to be absent.
    maximum_frames_to_retain_out_of_view_track: int = 300

    # --- Appearance re-identification (active memory) -------------------------
    # When a detection appears that no in-view track explains, it is compared to
    # the remembered intrinsic features of objects that are out of view or lost;
    # a partial-match score at or above this re-binds it to that object's id
    # (rather than spawning a new one) -- recognising the returning object.
    appearance_reidentification_minimum_similarity: float = 0.6

    # --- Data association gates and score weights -----------------------------
    intersection_over_union_gate_threshold: float = 0.25
    mahalanobis_distance_gate_threshold: float = 12.0
    # Looser gates used when trying to recover a previously lost track.
    lost_track_intersection_over_union_gate_threshold: float = 0.2
    lost_track_mahalanobis_distance_gate_threshold: float = 15.0
    # Relative contribution of each cue to the combined association score.
    intersection_over_union_score_weight: float = 0.4
    mahalanobis_distance_score_weight: float = 0.3
    appearance_similarity_score_weight: float = 0.3
    # Normalization constant that maps a Mahalanobis distance into a [0, 1] score.
    mahalanobis_distance_score_normalization: float = 20.0

    # --- Focus layer (anticipatory per-track local search) --------------------
    # The focus layer only inspects a small window around each track's predicted
    # position, so its cost is independent of frame size. These knobs shape that
    # window and the appearance match used to confirm the (possibly half-covered)
    # object inside it.
    #
    # The search window is the predicted box grown on every side by this fraction
    # of the box size, but never by fewer than the minimum pixel margin. A larger
    # window survives faster motion at the cost of more pixels to scan.
    focus_search_region_margin_ratio: float = 0.6
    focus_search_region_minimum_margin_in_pixels: int = 12
    # An appearance (normalized cross-correlation) peak at or above this value is
    # accepted as "the object is here", even when partially occluded. It is kept
    # deliberately low because a half-covered rocket produces a weaker peak.
    focus_minimum_template_match_confidence: float = 0.30
    # The stored appearance template is only refreshed from a *clean* view this
    # confident, so a partially occluded patch never poisons the template.
    focus_clean_template_refresh_confidence: float = 0.62
    # If two tracks would land closer together than this, the lower-scoring one is
    # treated as occluded (it coasts on prediction) -- a minimal occlusion
    # ordering that stops two identities from collapsing onto one visible blob.
    focus_occlusion_center_separation_in_pixels: float = 20.0
    # While two tracks are within this (wider) distance they are still both
    # tracked, but their velocity is protected from coalescence -- this covers the
    # messy separation frames, just after an overlap, where a normal update would
    # otherwise drag both velocities toward zero and let the tracks swap objects.
    focus_velocity_protection_center_separation_in_pixels: float = 44.0
    # Converts the distance between a candidate match and the predicted centre
    # into a motion-consistency penalty, so each track prefers the match that
    # agrees with its own trajectory. The motion term deliberately *dominates*
    # appearance: when two objects look identical, trajectory continuity is the
    # only thing that can tell them apart, so a short normalization distance makes
    # the blob nearest a track's prediction win decisively.
    focus_motion_consistency_distance_normalization: float = 30.0
    focus_match_score_appearance_weight: float = 0.4
    focus_match_score_motion_consistency_weight: float = 0.6

    # --- Redundant-track reconciliation --------------------------------------
    # Two active tracks that overlap this much *and* are moving together are
    # treated as duplicates of one object and merged (the younger is culled).
    # Velocity agreement is what distinguishes a duplicate (the two boxes move
    # the same way) from two real objects crossing (which move oppositely), so a
    # collision never triggers a merge no matter how much the boxes overlap.
    merge_coincident_track_minimum_overlap: float = 0.5
    merge_coincident_track_velocity_agreement: float = 0.5
    # When at least one track is nearly stationary, velocity direction is
    # meaningless, so fall back to merging only on very high overlap.
    merge_coincident_track_high_overlap_fallback: float = 0.7
    merge_coincident_track_minimum_speed: float = 0.5

    # --- Camera motion compensation ------------------------------------------
    enable_camera_motion_compensation: bool = False
    camera_motion_orb_feature_count: int = 2000
    camera_motion_minimum_match_count: int = 8
    camera_motion_ransac_reprojection_threshold: float = 3.0
