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
import logging

import cv2

from visionary.configuration import TrackerConfiguration
from visionary.detection.motion_difference_detector import MotionDifferenceDetector
from visionary.io.video_sources import (
    FrameSource,
    SyntheticRocketSceneSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)
from visionary.logging_configuration import configure_root_logging, get_logger_for_module
from visionary.tracking.multi_object_tracker import MultiObjectTracker
from visionary.visualization.overlay_renderer import (
    draw_frame_metadata,
    render_frame_processing_result,
)

_logger = get_logger_for_module(__name__)
_DISPLAY_WINDOW_TITLE = "Visionary Multi-Object Tracking"


def _parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description="Run the Visionary multi-object tracker on a chosen frame source."
    )
    argument_parser.add_argument(
        "--source",
        choices=["synthetic", "webcam", "video"],
        default="synthetic",
        help="Where frames come from.",
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
        return WebcamFrameSource(command_line_arguments.camera_index)
    if command_line_arguments.source == "video":
        if not command_line_arguments.video_path:
            raise SystemExit("--source video requires --video-path")
        return VideoFileFrameSource(command_line_arguments.video_path)
    return SyntheticRocketSceneSource(
        frame_width_in_pixels=configuration.frame_width_in_pixels,
        frame_height_in_pixels=configuration.frame_height_in_pixels,
        total_frame_count=command_line_arguments.max_frames,
    )


def run_tracking_loop(
    frame_source: FrameSource,
    tracker: MultiObjectTracker,
    display_window_enabled: bool,
) -> None:
    """Read frames until the source is exhausted or the user quits, tracking each."""
    frame_index = 0
    try:
        while True:
            frame = frame_source.read_next_frame()
            if frame is None:
                break
            frame_index += 1

            frame_processing_result = tracker.process_frame(frame)
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


def main() -> None:
    configure_root_logging(logging.INFO)
    command_line_arguments = _parse_command_line_arguments()
    configuration = TrackerConfiguration()

    frame_source = _build_frame_source(command_line_arguments, configuration)
    motion_detector = MotionDifferenceDetector(configuration)
    tracker = MultiObjectTracker(motion_detector, configuration)

    _logger.info("Starting tracking loop with source '%s'", command_line_arguments.source)
    run_tracking_loop(
        frame_source, tracker, display_window_enabled=not command_line_arguments.headless
    )


if __name__ == "__main__":
    main()
