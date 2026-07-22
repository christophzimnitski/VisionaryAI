"""All on-frame drawing lives here, isolated from tracking logic.

The tracker emits a drawing-free :class:`FrameProcessingResult`; this module turns
that result into an annotated frame. Keeping rendering separate means the tracker
can run headless (on a server, in CI) and lets the visual style change without
touching any tracking code.
"""

from __future__ import annotations

import cv2
import numpy as np

from visionary.tracking.multi_object_tracker import FrameProcessingResult, TrackResult

# Colours are expressed in OpenCV's BGR order.
_NEW_TRACK_COLOR_BGR = (255, 0, 0)
_RECOVERED_TRACK_COLOR_BGR = (0, 255, 255)
_ESTABLISHED_TRACK_COLOR_BGR = (0, 255, 0)
_DETECTION_BOX_COLOR_BGR = (0, 255, 255)
_COLLISION_BANNER_COLOR_BGR = (0, 165, 255)
_METADATA_TEXT_COLOR_BGR = (255, 255, 255)
_OUT_OF_VIEW_INDICATOR_COLOR_BGR = (255, 0, 255)


def _select_track_color_bgr(track_result: TrackResult) -> tuple[int, int, int]:
    if track_result.was_created_this_frame:
        return _NEW_TRACK_COLOR_BGR
    if track_result.was_recovered_this_frame:
        return _RECOVERED_TRACK_COLOR_BGR
    return _ESTABLISHED_TRACK_COLOR_BGR


def _draw_off_screen_indicator(frame: np.ndarray, track_result: TrackResult) -> None:
    """Draw the edge arrow + bearing pointing to an out-of-view object."""
    expectation = track_result.off_screen_expectation
    frame_height, frame_width = frame.shape[:2]

    arrow_length_in_pixels = 38
    arrow_start_x = int(
        expectation.edge_indicator_x - expectation.direction_unit_x * arrow_length_in_pixels
    )
    arrow_start_y = int(
        expectation.edge_indicator_y - expectation.direction_unit_y * arrow_length_in_pixels
    )
    arrow_start_x = min(max(arrow_start_x, 2), frame_width - 2)
    arrow_start_y = min(max(arrow_start_y, 2), frame_height - 2)
    arrow_end_x = min(max(expectation.edge_indicator_x, 1), frame_width - 1)
    arrow_end_y = min(max(expectation.edge_indicator_y, 1), frame_height - 1)

    cv2.arrowedLine(
        frame,
        (arrow_start_x, arrow_start_y),
        (arrow_end_x, arrow_end_y),
        _OUT_OF_VIEW_INDICATOR_COLOR_BGR,
        3,
        tipLength=0.4,
    )
    bearing_label = (
        f"ID:{track_result.track_id} "
        f"{expectation.bearing_degrees_clockwise_from_up:.0f}deg"
    )
    label_x = min(max(arrow_start_x - 30, 4), frame_width - 190)
    label_y = min(max(arrow_start_y - 8, 16), frame_height - 6)
    cv2.putText(
        frame,
        bearing_label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        _OUT_OF_VIEW_INDICATOR_COLOR_BGR,
        2,
    )


def render_frame_processing_result(
    frame: np.ndarray,
    frame_processing_result: FrameProcessingResult,
    draw_raw_detections: bool = True,
) -> np.ndarray:
    """Draw detections, tracks, identifiers and status banners onto a copy of the
    frame and return the annotated frame."""
    annotated_frame = frame.copy()

    if draw_raw_detections:
        for detection_left_x, detection_top_y, detection_width, detection_height in (
            frame_processing_result.detection_bounding_boxes
        ):
            cv2.rectangle(
                annotated_frame,
                (detection_left_x, detection_top_y),
                (detection_left_x + detection_width, detection_top_y + detection_height),
                _DETECTION_BOX_COLOR_BGR,
                1,
            )

    for track_result in frame_processing_result.track_results:
        if track_result.is_out_of_view and track_result.off_screen_expectation is not None:
            _draw_off_screen_indicator(annotated_frame, track_result)
            continue
        track_color_bgr = _select_track_color_bgr(track_result)
        left_x, top_y, width, height = track_result.bounding_box
        cv2.rectangle(
            annotated_frame, (left_x, top_y), (left_x + width, top_y + height), track_color_bgr, 2
        )
        cv2.putText(
            annotated_frame,
            f"ID:{track_result.track_id} [{track_result.object_type_token}]",
            (left_x, top_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            track_color_bgr,
            2,
        )

    if frame_processing_result.collision_or_occlusion_detected:
        cv2.putText(
            annotated_frame,
            "COLLISION / OCCLUSION DETECTED!",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _COLLISION_BANNER_COLOR_BGR,
            2,
        )

    return annotated_frame


_MISSING_OBJECT_COLOR_BGR = (0, 0, 255)
_FOUND_DISPLACED_OBJECT_COLOR_BGR = (255, 0, 255)
# Drawn for a displacement end whose box size is not carried in the conclusion.
_DISPLACEMENT_MARKER_HALF_WIDTH = 34
_DISPLACEMENT_MARKER_HALF_HEIGHT = 28


def render_room_scene_observation(frame: np.ndarray, room_scene_observation) -> np.ndarray:
    """Annotate a frame with tracks plus the room monitor's spatial reasoning.

    On top of the usual track overlay this draws, in world-projected positions: a
    red box where an object used to be (its place is now empty), and a magenta box
    where a displaced object was found, labelled with where it moved from. Together
    these are the "there was a pillow here" / "the missing pillow is on the floor"
    annotations.
    """
    annotated_frame = render_frame_processing_result(
        frame, room_scene_observation.frame_processing_result, draw_raw_detections=False
    )
    frame_height, frame_width = annotated_frame.shape[:2]
    world_offset_x, world_offset_y = room_scene_observation.world_offset

    for missing_object_marker in room_scene_observation.missing_object_markers:
        if missing_object_marker.frame_bounding_box is not None:
            _draw_labelled_box(
                annotated_frame,
                missing_object_marker.frame_bounding_box,
                "MISSING: was here",
                _MISSING_OBJECT_COLOR_BGR,
            )

    for displacement_conclusion in room_scene_observation.all_displacement_conclusions:
        from_frame_x = displacement_conclusion.from_world_position[0] - world_offset_x
        from_frame_y = displacement_conclusion.from_world_position[1] - world_offset_y
        if _point_is_in_frame(from_frame_x, from_frame_y, frame_width, frame_height):
            _draw_labelled_box(
                annotated_frame,
                _marker_box_around(from_frame_x, from_frame_y),
                "was here (moved away)",
                _MISSING_OBJECT_COLOR_BGR,
            )

        to_frame_x = displacement_conclusion.to_world_position[0] - world_offset_x
        to_frame_y = displacement_conclusion.to_world_position[1] - world_offset_y
        if _point_is_in_frame(to_frame_x, to_frame_y, frame_width, frame_height):
            _draw_labelled_box(
                annotated_frame,
                _marker_box_around(to_frame_x, to_frame_y),
                "FOUND (moved here)",
                _FOUND_DISPLACED_OBJECT_COLOR_BGR,
            )

    return annotated_frame


def _draw_labelled_box(
    frame: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    label: str,
    color_bgr: tuple[int, int, int],
) -> None:
    left_x, top_y, width, height = (int(value) for value in bounding_box)
    cv2.rectangle(frame, (left_x, top_y), (left_x + width, top_y + height), color_bgr, 2)
    cv2.putText(
        frame,
        label,
        (left_x, max(top_y - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_bgr,
        2,
    )


def _marker_box_around(center_x: float, center_y: float) -> tuple[int, int, int, int]:
    return (
        int(center_x) - _DISPLACEMENT_MARKER_HALF_WIDTH,
        int(center_y) - _DISPLACEMENT_MARKER_HALF_HEIGHT,
        2 * _DISPLACEMENT_MARKER_HALF_WIDTH,
        2 * _DISPLACEMENT_MARKER_HALF_HEIGHT,
    )


def _point_is_in_frame(
    point_x: float, point_y: float, frame_width: int, frame_height: int
) -> bool:
    return 0 <= point_x < frame_width and 0 <= point_y < frame_height


def draw_frame_metadata(
    frame: np.ndarray, tracked_object_count: int, frame_index: int
) -> np.ndarray:
    """Overlay the running object count and frame index in the top-left corner."""
    cv2.putText(
        frame,
        f"Objects: {tracked_object_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        _METADATA_TEXT_COLOR_BGR,
        2,
    )
    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        _METADATA_TEXT_COLOR_BGR,
        2,
    )
    return frame
