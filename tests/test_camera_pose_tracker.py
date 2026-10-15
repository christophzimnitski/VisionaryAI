"""The camera pose tracker should follow the known pan of the room scene.

The panning-camera scene exposes its true pan offset, so we can check that the
offset integrated from per-frame egomotion follows it: rising monotonically while
the camera pans right and staying close to ground truth over a short window
(before accumulated drift -- the documented limitation -- grows large).
"""

from __future__ import annotations

import cv2

from visionary.configuration import TrackerConfiguration
from visionary.io.video_sources import PanningCameraSceneSource
from visionary.tracking.camera_motion_compensation import estimate_camera_motion_homography
from visionary.tracking.camera_pose_tracker import CameraPoseTracker


def test_tracked_world_offset_follows_the_known_pan() -> None:
    configuration = TrackerConfiguration()
    scene_source = PanningCameraSceneSource(total_frame_count=26, random_seed=0)
    pose_tracker = CameraPoseTracker()

    previous_grayscale_frame = None
    first_true_pan_offset = None
    final_true_pan_offset = None
    tracked_offsets_x: list[float] = []
    while True:
        frame = scene_source.read_next_frame()
        if frame is None:
            break
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        true_pan_offset = scene_source._current_pan_offset_x()
        if first_true_pan_offset is None:
            first_true_pan_offset = true_pan_offset
        final_true_pan_offset = true_pan_offset
        if previous_grayscale_frame is not None:
            homography = estimate_camera_motion_homography(
                previous_grayscale_frame, grayscale_frame, configuration
            )
            pose_tracker.update_with_frame_to_frame_homography(homography, frame.shape)
            tracked_offsets_x.append(pose_tracker.current_world_offset[0])
        previous_grayscale_frame = grayscale_frame

    # The camera pans steadily right, so the tracked offset should climb.
    assert tracked_offsets_x[-1] > tracked_offsets_x[0]
    assert tracked_offsets_x[-1] > 60.0
    assert all(
        later >= earlier - 2.0
        for earlier, later in zip(tracked_offsets_x, tracked_offsets_x[1:], strict=False)
    )

    # And it should land near the ground-truth displacement from the first frame.
    true_displacement = final_true_pan_offset - first_true_pan_offset
    assert abs(tracked_offsets_x[-1] - true_displacement) < 40.0

    # No vertical pan, so the world y-offset must stay near zero.
    assert abs(pose_tracker.current_world_offset[1]) < 25.0
