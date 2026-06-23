"""Camera motion compensation (global motion estimation).

Estimates the frame-to-frame homography induced by camera movement using ORB
feature matches plus RANSAC. The original code computed this homography and then
*never used it*; here we return it so the tracker can warp its predictions, which
keeps camera motion from being mistaken for object motion during association.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionary.configuration import TrackerConfiguration


def estimate_camera_motion_homography(
    previous_grayscale_frame: np.ndarray,
    current_grayscale_frame: np.ndarray,
    configuration: TrackerConfiguration,
) -> np.ndarray | None:
    """Return the 3x3 homography mapping previous-frame points to current-frame
    points, or ``None`` if it cannot be estimated reliably."""
    orb_feature_detector = cv2.ORB_create(configuration.camera_motion_orb_feature_count)
    previous_keypoints, previous_descriptors = orb_feature_detector.detectAndCompute(
        previous_grayscale_frame, None
    )
    current_keypoints, current_descriptors = orb_feature_detector.detectAndCompute(
        current_grayscale_frame, None
    )

    if (
        previous_descriptors is None
        or current_descriptors is None
        or len(previous_descriptors) <= configuration.camera_motion_minimum_match_count
        or len(current_descriptors) <= configuration.camera_motion_minimum_match_count
    ):
        return None

    brute_force_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    feature_matches = brute_force_matcher.match(previous_descriptors, current_descriptors)
    feature_matches = sorted(feature_matches, key=lambda match: match.distance)[:100]
    if len(feature_matches) < configuration.camera_motion_minimum_match_count:
        return None

    previous_matched_points = np.float32(
        [previous_keypoints[match.queryIdx].pt for match in feature_matches]
    ).reshape(-1, 1, 2)
    current_matched_points = np.float32(
        [current_keypoints[match.trainIdx].pt for match in feature_matches]
    ).reshape(-1, 1, 2)

    homography_matrix, _ = cv2.findHomography(
        previous_matched_points,
        current_matched_points,
        cv2.RANSAC,
        configuration.camera_motion_ransac_reprojection_threshold,
    )
    return homography_matrix
