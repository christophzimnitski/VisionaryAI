"""Command-line application that wires the pipeline together and runs it.

Usage examples::

    python -m visionary.application --source synthetic
    python -m visionary.application --source webcam --camera-index 0
    python -m visionary.application --source video --video-path path/to/clip.mp4
    python -m visionary.application --source synthetic --headless --max-frames 500

This module only *wires up* the injected pieces (frame source, detector, tracker,
renderer); all real work lives in the library modules it imports.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Callable

import cv2

from visionary.configuration import TrackerConfiguration
from visionary.detection.base import NullObjectDetector
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.detection.salient_region_detector import SalientRegionDetector
from visionary.io.video_sources import (
    FrameSource,
    PanningCameraSceneSource,
    RoomWithMovableObjectsSceneSource,
    SyntheticRocketSceneSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)
from visionary.logging_configuration import configure_root_logging, get_logger_for_module
from visionary.tracking.multi_object_tracker import MultiObjectTracker
from visionary.tracking.room_object_monitor import RoomObjectMonitor
from visionary.visualization.overlay_renderer import (
    draw_frame_metadata,
    render_frame_processing_result,
    render_room_scene_observation,
)

_logger = get_logger_for_module(__name__)
_DISPLAY_WINDOW_TITLE = "Visionary Multi-Object Tracking"


def _parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Run the Visionary multi-object tracker on a chosen frame source."
    )
    argument_parser.add_argument(
        "--source",
        choices=["synthetic", "webcam", "video", "room", "room-objects"],
        default="synthetic",
        help=(
            "Where frames come from ('room' is the panning-camera demo; "
            "'room-objects' is the move-a-pillow-while-looking-away demo)."
        ),
    )
    argument_parser.add_argument(
        "--monitor",
        action="store_true",
        help=(
            "Enable room-object monitoring: auto-discover big objects, anchor them "
            "in the room, and reason about ones moved while out of view. Implied by "
            "--source room-objects; add it to --source webcam for a live room."
        ),
    )
    argument_parser.add_argument(
        "--video-path", default=None, help="Path to a video file when --source video."
    )
    argument_parser.add_argument(
        "--camera-index", type=int, default=0, help="Camera device index when --source webcam."
    )
    argument_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (mainly for the synthetic source).",
    )
    argument_parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open a display window (for servers / CI).",
    )
    return argument_parser.parse_args()


def _build_frame_source(
    command_line_arguments: argparse.Namespace, configuration: TrackerConfiguration
) -> FrameSource:
    if command_line_arguments.source == "webcam":
        return WebcamFrameSource(
            command_line_arguments.camera_index,
            frame_width_in_pixels=configuration.frame_width_in_pixels,
            frame_height_in_pixels=configuration.frame_height_in_pixels,
        )
    if command_line_arguments.source == "video":
        if not command_line_arguments.video_path:
            raise SystemExit("--source video requires --video-path")
        return VideoFileFrameSource(command_line_arguments.video_path)
    if command_line_arguments.source == "room":
        return PanningCameraSceneSource(
            frame_width_in_pixels=configuration.frame_width_in_pixels,
            frame_height_in_pixels=configuration.frame_height_in_pixels,
            total_frame_count=command_line_arguments.max_frames,
        )
    if command_line_arguments.source == "room-objects":
        return RoomWithMovableObjectsSceneSource(
            frame_width_in_pixels=configuration.frame_width_in_pixels,
            frame_height_in_pixels=configuration.frame_height_in_pixels,
            total_frame_count=command_line_arguments.max_frames or 132,
        )
    return SyntheticRocketSceneSource(
        frame_width_in_pixels=configuration.frame_width_in_pixels,
        frame_height_in_pixels=configuration.frame_height_in_pixels,
        total_frame_count=command_line_arguments.max_frames,
    )


def _register_designated_painting_on_first_frame(
    tracker: MultiObjectTracker, frame, frame_source: FrameSource
) -> None:
    """Point the tracker at the painting in the panning-camera demo's first frame."""
    ground_truth_boxes = getattr(frame_source, "current_ground_truth_bounding_boxes", {})
    designated_painting_box = ground_truth_boxes.get("painting_red")
    if designated_painting_box is not None:
        tracker.register_object_of_interest(designated_painting_box, frame)


def run_tracking_loop(
    frame_source: FrameSource,
    tracker: MultiObjectTracker,
    display_window_enabled: bool,
    after_first_frame: Callable[[MultiObjectTracker, object, FrameSource], None] | None = None,
) -> None:
    """Read frames until the source is exhausted or the user quits, tracking each.

    ``after_first_frame`` is an optional hook (used by the panning-camera demo to
    designate the object to follow once the first frame has primed the tracker).
    """
    frame_index = 0
    try:
        while True:
            frame = frame_source.read_next_frame()
            if frame is None:
                break
            frame_index += 1

            frame_processing_result = tracker.process_frame(frame)
            if frame_index == 1 and after_first_frame is not None:
                after_first_frame(tracker, frame, frame_source)
            annotated_frame = render_frame_processing_result(frame, frame_processing_result)
            annotated_frame = draw_frame_metadata(
                annotated_frame, tracker.active_track_count, frame_index
            )

            if display_window_enabled:
                cv2.imshow(_DISPLAY_WINDOW_TITLE, annotated_frame)
                if (cv2.waitKey(30) & 0xFF) == ord("q"):
                    break
    finally:
        frame_source.release()
        if display_window_enabled:
            cv2.destroyAllWindows()


def run_room_monitor_loop(
    frame_source: FrameSource,
    room_object_monitor: RoomObjectMonitor,
    display_window_enabled: bool,
) -> None:
    """Read frames, monitor the room, and draw its spatial reasoning each frame."""
    frame_index = 0
    try:
        while True:
            frame = frame_source.read_next_frame()
            if frame is None:
                break
            frame_index += 1

            room_scene_observation = room_object_monitor.observe_frame(frame)
            for displacement_conclusion in room_scene_observation.newly_concluded_displacements:
                _logger.info(
                    "Realised an object moved: it was at world (%.0f, %.0f) and is "
                    "now at world (%.0f, %.0f)",
                    displacement_conclusion.from_world_position[0],
                    displacement_conclusion.from_world_position[1],
                    displacement_conclusion.to_world_position[0],
                    displacement_conclusion.to_world_position[1],
                )
            annotated_frame = render_room_scene_observation(frame, room_scene_observation)
            annotated_frame = draw_frame_metadata(
                annotated_frame, room_object_monitor.anchor_memory.anchor_count, frame_index
            )

            if display_window_enabled:
                cv2.imshow(_DISPLAY_WINDOW_TITLE, annotated_frame)
                if (cv2.waitKey(30) & 0xFF) == ord("q"):
                    break
    finally:
        frame_source.release()
        if display_window_enabled:
            cv2.destroyAllWindows()


def main() -> None:
    configure_root_logging(logging.INFO)
    command_line_arguments = _parse_command_line_arguments()
    configuration = TrackerConfiguration()

    is_room_object_monitoring = (
        command_line_arguments.monitor or command_line_arguments.source == "room-objects"
    )
    if is_room_object_monitoring:
        # A moving camera that discovers big objects, anchors them in the room, and
        # reasons about ones moved while it was looking elsewhere.
        configuration = dataclasses.replace(
            configuration, enable_camera_motion_compensation=True
        )
        frame_source = _build_frame_source(command_line_arguments, configuration)
        room_object_monitor = RoomObjectMonitor(
            SalientRegionDetector(configuration), configuration
        )
        _logger.info(
            "Starting room monitor with source '%s'", command_line_arguments.source
        )
        run_room_monitor_loop(
            frame_source,
            room_object_monitor,
            display_window_enabled=not command_line_arguments.headless,
        )
        return

    is_panning_camera_demo = command_line_arguments.source == "room"
    if is_panning_camera_demo:
        # A moving camera over static objects: estimate egomotion to drive the
        # predictions, and designate the object to follow rather than discover it.
        configuration = dataclasses.replace(
            configuration, enable_camera_motion_compensation=True
        )
        object_detector = NullObjectDetector()
        after_first_frame = _register_designated_painting_on_first_frame
    else:
        object_detector = MotionDifferenceDetector(configuration)
        after_first_frame = None

    frame_source = _build_frame_source(command_line_arguments, configuration)
    tracker = MultiObjectTracker(object_detector, configuration)

    _logger.info("Starting tracking loop with source '%s'", command_line_arguments.source)
    run_tracking_loop(
        frame_source,
        tracker,
        display_window_enabled=not command_line_arguments.headless,
        after_first_frame=after_first_frame,
    )


if __name__ == "__main__":
    main()
