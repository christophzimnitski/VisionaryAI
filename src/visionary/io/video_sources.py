"""Frame sources.

Decouples *where frames come from* (a video file, a live webcam, or a synthetic
simulation) from the tracking logic. Every source yields BGR frames as numpy
arrays through the same :class:`FrameSource` protocol, so the tracker and CLI work
identically regardless of the input.
"""

from __future__ import annotations

from importlib import resources
from typing import Iterator, Protocol

import cv2
import numpy as np

_BUNDLED_ROCKET_SPRITE_RESOURCE = "assets/rocket_template.jpeg"


def load_bundled_rocket_sprite_image() -> np.ndarray | None:
    """Load the rocket sprite shipped as package data, or ``None`` if unavailable.

    Using ``importlib.resources`` means the image is found whether the package is
    run from source (``PYTHONPATH=src``) or installed as a wheel, regardless of the
    current working directory.
    """
    try:
        rocket_sprite_resource = resources.files("visionary").joinpath(
            _BUNDLED_ROCKET_SPRITE_RESOURCE
        )
        with resources.as_file(rocket_sprite_resource) as rocket_sprite_path:
            return cv2.imread(str(rocket_sprite_path), cv2.IMREAD_COLOR)
    except (FileNotFoundError, ModuleNotFoundError):
        return None


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

        if rocket_sprite_image is None:
            # Prefer the real rocket image bundled with the package; only fall back
            # to a generated placeholder sprite if it cannot be loaded.
            rocket_sprite_image = load_bundled_rocket_sprite_image()
        if rocket_sprite_image is None:
            rocket_sprite_image = self._build_placeholder_rocket_sprite()
        self._rocket_sprite_image = rocket_sprite_image
        self._sprite_height, self._sprite_width = self._rocket_sprite_image.shape[:2]

        # Per-rocket recoloured sprites are cached here so a tinted variant is
        # built once, not every frame.
        self._tinted_sprite_cache: dict[tuple[float, float, float], np.ndarray] = {}

        # The rockets travel on diagonal paths and bounce off every wall, so the
        # motion looks natural rather than gliding along a single axis. rocket_a
        # and rocket_b are placed on the *same* diagonal line travelling in
        # opposite directions, so they still meet in a head-on collision near the
        # centre of the frame -- which exercises the identity-through-occlusion
        # handling. Their sprites are identical, so only motion can tell them apart.
        #
        # rocket_d is a *distinct* blue rocket roaming its own diagonal. Because it
        # does not look like the others, the tracker can recognise it by its
        # intrinsic features (its feature memory) and hold a single stable identity
        # for it even through crossings -- the "this is the same blue object"
        # reasoning that the camera stage will reuse for a panned-away landmark.
        self._simulated_rockets: dict[str, dict] = {
            "rocket_a": {
                "left_x": 40, "top_y": 80, "velocity_x": 3, "velocity_y": 2, "is_visible": True
            },
            "rocket_b": {
                "left_x": 520, "top_y": 400, "velocity_x": -3, "velocity_y": -2, "is_visible": True
            },
            "rocket_d": {
                "left_x": 320, "top_y": 250, "velocity_x": 2, "velocity_y": -3,
                "color_tint": (1.7, 0.55, 0.55), "is_visible": True,
            },
        }
        self.current_ground_truth_bounding_boxes: dict[str, tuple[int, int, int, int]] = {}

    def _sprite_image_for_rocket(self, rocket_state: dict) -> np.ndarray:
        """Return this rocket's sprite, recoloured by its tint if it has one."""
        color_tint = rocket_state.get("color_tint")
        if color_tint is None:
            return self._rocket_sprite_image
        cached_sprite = self._tinted_sprite_cache.get(color_tint)
        if cached_sprite is None:
            cached_sprite = np.clip(
                self._rocket_sprite_image.astype(np.float32)
                * np.array(color_tint, dtype=np.float32),
                0,
                255,
            ).astype(np.uint8)
            self._tinted_sprite_cache[color_tint] = cached_sprite
        return cached_sprite

    @staticmethod
    def _build_placeholder_rocket_sprite() -> np.ndarray:
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
            # A third rocket enters on its own diagonal, crossing the others'
            # paths and bouncing off the walls like the rest.
            self._simulated_rockets["rocket_c"] = {
                "left_x": 300, "top_y": 40, "velocity_x": -2, "velocity_y": 3, "is_visible": True
            }
        rocket_a_state = self._simulated_rockets["rocket_a"]
        if self._current_frame_index == 300 and rocket_a_state["is_visible"]:
            rocket_a_state["is_visible"] = False
        if self._current_frame_index == 450 and not rocket_a_state["is_visible"]:
            self._simulated_rockets["rocket_a"]["is_visible"] = True
            self._simulated_rockets["rocket_a"]["left_x"] = 40
            self._simulated_rockets["rocket_a"]["top_y"] = 80
            self._simulated_rockets["rocket_a"]["velocity_x"] = 3
            self._simulated_rockets["rocket_a"]["velocity_y"] = 2

    def _render_current_frame(self) -> np.ndarray:
        frame = self._random_number_generator.integers(
            0, 5, (self._frame_height_in_pixels, self._frame_width_in_pixels, 3), dtype=np.uint8
        )
        self.current_ground_truth_bounding_boxes = {}
        for rocket_identifier, rocket_state in self._simulated_rockets.items():
            if not rocket_state["is_visible"]:
                continue
            sprite_image = self._sprite_image_for_rocket(rocket_state)
            box_left_x = int(round(rocket_state["left_x"]))
            box_top_y = int(round(rocket_state["top_y"]))

            # Intersect the sprite's box with the frame. A rocket that is fully
            # outside the frame contributes no pixels and no ground truth -- it is
            # genuinely gone, so the tracker must rely on permanence to hold it.
            visible_left_x = max(0, box_left_x)
            visible_top_y = max(0, box_top_y)
            visible_right_x = min(self._frame_width_in_pixels, box_left_x + self._sprite_width)
            visible_bottom_y = min(self._frame_height_in_pixels, box_top_y + self._sprite_height)
            if visible_right_x <= visible_left_x or visible_bottom_y <= visible_top_y:
                continue

            sprite_offset_x = visible_left_x - box_left_x
            sprite_offset_y = visible_top_y - box_top_y
            frame[visible_top_y:visible_bottom_y, visible_left_x:visible_right_x] = sprite_image[
                sprite_offset_y : sprite_offset_y + (visible_bottom_y - visible_top_y),
                sprite_offset_x : sprite_offset_x + (visible_right_x - visible_left_x),
            ]
            self.current_ground_truth_bounding_boxes[rocket_identifier] = (
                visible_left_x,
                visible_top_y,
                visible_right_x - visible_left_x,
                visible_bottom_y - visible_top_y,
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
