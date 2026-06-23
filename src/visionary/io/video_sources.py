"""Frame sources.

Decouples *where frames come from* (a video file, a live webcam, or a synthetic
simulation) from the tracking logic. Every source yields BGR frames as numpy
arrays through the same :class:`FrameSource` protocol, so the tracker and CLI work
identically regardless of the input.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    """Anything that can produce a stream of video frames."""

    def read_next_frame(self) -> np.ndarray | None:
        """Return the next frame, or ``None`` when the stream is exhausted."""
        ...

    def release(self) -> None:
        """Release any underlying resources (file handle, camera device)."""
        ...


class VideoFileFrameSource:
    """Reads frames sequentially from a video file on disk."""

    def __init__(self, video_file_path: str) -> None:
        self._video_capture = cv2.VideoCapture(video_file_path)
        if not self._video_capture.isOpened():
            raise FileNotFoundError(f"Could not open video file at {video_file_path}")

    def read_next_frame(self) -> np.ndarray | None:
        frame_was_read, frame = self._video_capture.read()
        return frame if frame_was_read else None

    def release(self) -> None:
        self._video_capture.release()


class WebcamFrameSource:
    """Reads frames from a live camera device."""

    def __init__(self, camera_device_index: int = 0) -> None:
        self._video_capture = cv2.VideoCapture(camera_device_index)
        if not self._video_capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam at device index {camera_device_index}"
            )

    def read_next_frame(self) -> np.ndarray | None:
        frame_was_read, frame = self._video_capture.read()
        return frame if frame_was_read else None

    def release(self) -> None:
        self._video_capture.release()


class SyntheticRocketSceneSource:
    """A deterministic simulation of rockets bouncing around the frame.

    Besides being a runnable demo with no camera required, this source exposes the
    *ground-truth* bounding box of every visible object each frame, which the
    evaluation harness uses to score tracking quality. The simulation is fully
    deterministic for a given ``random_seed`` so tests are reproducible.
    """

    def __init__(
        self,
        frame_width_in_pixels: int = 640,
        frame_height_in_pixels: int = 480,
        rocket_sprite_image: np.ndarray | None = None,
        total_frame_count: int | None = None,
        random_seed: int = 0,
    ) -> None:
        self._frame_width_in_pixels = frame_width_in_pixels
        self._frame_height_in_pixels = frame_height_in_pixels
        self._total_frame_count = total_frame_count
        self._random_number_generator = np.random.default_rng(random_seed)
        self._current_frame_index = 0

        self._rocket_sprite_image = (
            rocket_sprite_image
            if rocket_sprite_image is not None
            else self._build_default_rocket_sprite()
        )
        self._sprite_height, self._sprite_width = self._rocket_sprite_image.shape[:2]

        self._simulated_rockets: dict[str, dict] = {
            "rocket_a": {"left_x": 50, "top_y": 100, "velocity_x": 2, "velocity_y": 1, "is_visible": True},
            "rocket_b": {"left_x": 200, "top_y": 300, "velocity_x": -1, "velocity_y": 2, "is_visible": True},
        }
        self.current_ground_truth_bounding_boxes: dict[str, tuple[int, int, int, int]] = {}

    @staticmethod
    def _build_default_rocket_sprite() -> np.ndarray:
        rocket_sprite_image = np.zeros((30, 60, 3), dtype=np.uint8)
        cv2.rectangle(rocket_sprite_image, (5, 5), (25, 55), (200, 200, 200), -1)
        cv2.circle(rocket_sprite_image, (15, 5), 5, (0, 0, 255), -1)
        return rocket_sprite_image

    def read_next_frame(self) -> np.ndarray | None:
        if (
            self._total_frame_count is not None
            and self._current_frame_index >= self._total_frame_count
        ):
            return None
        self._current_frame_index += 1
        self._advance_simulation_one_step()
        return self._render_current_frame()

    def _advance_simulation_one_step(self) -> None:
        for rocket_state in self._simulated_rockets.values():
            if not rocket_state["is_visible"]:
                continue
            rocket_state["left_x"] += rocket_state["velocity_x"]
            rocket_state["top_y"] += rocket_state["velocity_y"]
            if (
                rocket_state["left_x"] <= 0
                or rocket_state["left_x"] >= self._frame_width_in_pixels - self._sprite_width
            ):
                rocket_state["velocity_x"] *= -1
            if (
                rocket_state["top_y"] <= 0
                or rocket_state["top_y"] >= self._frame_height_in_pixels - self._sprite_height
            ):
                rocket_state["velocity_y"] *= -1

        # Scripted events that exercise new-object discovery and lost/recover logic.
        if self._current_frame_index == 150 and "rocket_c" not in self._simulated_rockets:
            self._simulated_rockets["rocket_c"] = {
                "left_x": 10, "top_y": 400, "velocity_x": 3, "velocity_y": -1, "is_visible": True
            }
        if self._current_frame_index == 300 and self._simulated_rockets["rocket_a"]["is_visible"]:
            self._simulated_rockets["rocket_a"]["is_visible"] = False
        if self._current_frame_index == 450 and not self._simulated_rockets["rocket_a"]["is_visible"]:
            self._simulated_rockets["rocket_a"]["is_visible"] = True
            self._simulated_rockets["rocket_a"]["left_x"] = 50
            self._simulated_rockets["rocket_a"]["top_y"] = 100

    def _render_current_frame(self) -> np.ndarray:
        frame = self._random_number_generator.integers(
            0, 5, (self._frame_height_in_pixels, self._frame_width_in_pixels, 3), dtype=np.uint8
        )
        self.current_ground_truth_bounding_boxes = {}
        for rocket_identifier, rocket_state in self._simulated_rockets.items():
            if not rocket_state["is_visible"]:
                continue
            clamped_left_x = max(
                0, min(rocket_state["left_x"], self._frame_width_in_pixels - self._sprite_width)
            )
            clamped_top_y = max(
                0, min(rocket_state["top_y"], self._frame_height_in_pixels - self._sprite_height)
            )
            drawable_width = min(self._sprite_width, self._frame_width_in_pixels - clamped_left_x)
            drawable_height = min(self._sprite_height, self._frame_height_in_pixels - clamped_top_y)
            if drawable_width <= 0 or drawable_height <= 0:
                continue
            frame[
                clamped_top_y : clamped_top_y + drawable_height,
                clamped_left_x : clamped_left_x + drawable_width,
            ] = self._rocket_sprite_image[0:drawable_height, 0:drawable_width]
            self.current_ground_truth_bounding_boxes[rocket_identifier] = (
                clamped_left_x, clamped_top_y, drawable_width, drawable_height
            )
        return frame

    def release(self) -> None:  # noqa: D401 - nothing to release for the simulation
        return None

    def iterate_frames_with_ground_truth(
        self,
    ) -> Iterator[tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]]:
        """Yield ``(frame, ground_truth_bounding_boxes)`` until the stream ends."""
        while True:
            frame = self.read_next_frame()
            if frame is None:
                return
            yield frame, dict(self.current_ground_truth_bounding_boxes)
