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

    # --- Camera motion compensation ------------------------------------------
    enable_camera_motion_compensation: bool = False
    camera_motion_orb_feature_count: int = 2000
    camera_motion_minimum_match_count: int = 8
    camera_motion_ransac_reprojection_threshold: float = 3.0
