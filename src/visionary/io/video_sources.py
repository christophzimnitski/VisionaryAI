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
    """Reads frames from a live camera device.

    Two camera behaviours actively fight change detection on a panning camera, so
    this source tries to tame them: it requests a fixed capture resolution, and it
    asks the driver to hold exposure and white balance steady. Auto-exposure and
    auto-white-balance re-meter as the camera pans, which makes the *same* wall
    change brightness and colour and look like a change when nothing moved.
    Cameras may ignore these requests -- they are best-effort, never required.
    """

    def __init__(
        self,
        camera_device_index: int = 0,
        frame_width_in_pixels: int | None = None,
        frame_height_in_pixels: int | None = None,
        hold_exposure_and_white_balance_steady: bool = True,
    ) -> None:
        self._video_capture = cv2.VideoCapture(camera_device_index)
        if not self._video_capture.isOpened():
            raise RuntimeError(
                f"Could not open webcam at device index {camera_device_index}"
            )
        if frame_width_in_pixels is not None:
            self._video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width_in_pixels)
        if frame_height_in_pixels is not None:
            self._video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height_in_pixels)
        if hold_exposure_and_white_balance_steady:
            self._request_steady_exposure_and_white_balance()

    def _request_steady_exposure_and_white_balance(self) -> None:
        # The "auto" properties are quirky across drivers (the 0.25 magic value is
        # the common incantation for "manual" on V4L2 backends); wrapping in a
        # try/except keeps an unsupported camera from crashing the source.
        try:
            self._video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._video_capture.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        except cv2.error:
            return

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


class RoomWithMovableObjectsSceneSource:
    """A panning room where an object is *moved while the camera looks away*.

    This is the headline scenario, made deterministic. A camera looks at a "bed"
    (a blanket with two identical pillows on it), pans right to a "desk", and while
    the bed is out of view the scene mutates: one pillow vanishes from the bed and
    a matching pillow appears on the floor by the desk. The camera then pans back.

    The point is that the move is *never observed* -- there is no continuous motion
    to track from bed to floor. The room monitor must instead notice, on the return
    pan, that a pillow's place on the bed is empty (a disappearance) and that a new
    pillow turned up by the desk (an appearance), and reason that they are the same
    pillow. The background is faintly textured so egomotion can be estimated, yet
    quantizes to a couple of dominant colours so the salient-region detector still
    finds the distinctly-coloured furniture.
    """

    _PILLOW_COLOR_BGR = (235, 235, 235)
    _BLANKET_COLOR_BGR = (150, 110, 40)
    _DESK_COLOR_BGR = (40, 70, 110)

    def __init__(
        self,
        frame_width_in_pixels: int = 640,
        frame_height_in_pixels: int = 480,
        total_frame_count: int | None = 132,
        random_seed: int = 0,
    ) -> None:
        self._viewport_width_in_pixels = frame_width_in_pixels
        self._viewport_height_in_pixels = frame_height_in_pixels
        self._total_frame_count = total_frame_count
        self._current_frame_index = 0

        self._room_canvas_width = 1200
        self._room_canvas_height = frame_height_in_pixels

        # Static furniture in world (room) coordinates: (left_x, top_y, width, height).
        self._blanket_world_box = (70, 200, 240, 110)
        self._pillow_left_world_box = (110, 210, 64, 48)
        self._pillow_right_world_box = (210, 210, 64, 48)
        self._desk_world_box = (700, 250, 150, 120)
        # The pillow that will appear on the floor by the desk after the move.
        self._floor_pillow_world_box = (760, 400, 64, 48)
        self._scene_mutation_frame_index = 60
        self._scene_has_been_mutated = False

        random_number_generator = np.random.default_rng(random_seed)
        self._room_canvas = self._build_faintly_textured_room(random_number_generator)
        self._paint_static_furniture()

    def _build_faintly_textured_room(
        self, random_number_generator: np.random.Generator
    ) -> np.ndarray:
        """A low-contrast two-tone texture: corners for egomotion, few colours."""
        # Two grey tones that share one coarse colour bin, so the whole wall reads
        # as a single background colour to the salient detector, yet differ enough
        # to give egomotion plenty of corners to lock onto.
        room_canvas = np.full(
            (self._room_canvas_height, self._room_canvas_width, 3), 50, dtype=np.uint8
        )
        for _ in range(1800):
            patch_left_x = int(random_number_generator.integers(0, self._room_canvas_width - 20))
            patch_top_y = int(random_number_generator.integers(0, self._room_canvas_height - 20))
            patch_width = int(random_number_generator.integers(6, 18))
            patch_height = int(random_number_generator.integers(6, 18))
            cv2.rectangle(
                room_canvas,
                (patch_left_x, patch_top_y),
                (patch_left_x + patch_width, patch_top_y + patch_height),
                (80, 80, 80),
                -1,
            )
        return room_canvas

    def _paint_static_furniture(self) -> None:
        self._fill_world_box(self._blanket_world_box, self._BLANKET_COLOR_BGR)
        self._fill_world_box(self._pillow_left_world_box, self._PILLOW_COLOR_BGR)
        self._fill_world_box(self._pillow_right_world_box, self._PILLOW_COLOR_BGR)
        self._fill_world_box(self._desk_world_box, self._DESK_COLOR_BGR)

    def _fill_world_box(
        self, world_box: tuple[int, int, int, int], color_bgr: tuple[int, int, int]
    ) -> None:
        left_x, top_y, width, height = world_box
        cv2.rectangle(
            self._room_canvas, (left_x, top_y), (left_x + width, top_y + height), color_bgr, -1
        )

    def _mutate_scene_moving_a_pillow_to_the_floor(self) -> None:
        """Remove the right bed pillow (revealing blanket) and add a floor pillow."""
        # Erase the right pillow by repainting the blanket it rested on.
        self._fill_world_box(self._pillow_right_world_box, self._BLANKET_COLOR_BGR)
        self._fill_world_box(self._floor_pillow_world_box, self._PILLOW_COLOR_BGR)
        self._scene_has_been_mutated = True

    def _current_pan_offset_x(self) -> int:
        """Hold on the bed, pan right to the desk, then pan back to the bed."""
        frame_index = self._current_frame_index
        hold_frames = 12
        pan_speed_pixels_per_frame = 8
        maximum_pan_offset = 480
        pan_out_frames = maximum_pan_offset // pan_speed_pixels_per_frame
        if frame_index <= hold_frames:
            return 0
        if frame_index <= hold_frames + pan_out_frames:
            return (frame_index - hold_frames) * pan_speed_pixels_per_frame
        frames_panning_back = frame_index - hold_frames - pan_out_frames
        return max(0, maximum_pan_offset - frames_panning_back * pan_speed_pixels_per_frame)

    def read_next_frame(self) -> np.ndarray | None:
        if (
            self._total_frame_count is not None
            and self._current_frame_index >= self._total_frame_count
        ):
            return None
        self._current_frame_index += 1

        if (
            not self._scene_has_been_mutated
            and self._current_frame_index >= self._scene_mutation_frame_index
        ):
            self._mutate_scene_moving_a_pillow_to_the_floor()

        pan_offset_x = self._current_pan_offset_x()
        viewport = self._room_canvas[
            0 : self._viewport_height_in_pixels,
            pan_offset_x : pan_offset_x + self._viewport_width_in_pixels,
        ].copy()
        return viewport

    def release(self) -> None:  # noqa: D401 - nothing to release for the simulation
        return None


class PanningCameraSceneSource:
    """A static, textured 'room' that a virtual camera pans across.

    The mirror image of the rocket scene: here the objects are *static* in the
    world and the camera moves -- the configuration the room/painting use case
    needs. The wall is textured so frame-to-frame camera motion can be estimated
    from it (egomotion), and the 'paintings' hang at fixed places on it. Each frame
    is the camera's current viewport onto the wall; the source exposes every
    painting's ground-truth box within that viewport, or its absence once the
    camera has panned past it.
    """

    def __init__(
        self,
        frame_width_in_pixels: int = 640,
        frame_height_in_pixels: int = 480,
        total_frame_count: int | None = None,
        random_seed: int = 0,
    ) -> None:
        self._viewport_width_in_pixels = frame_width_in_pixels
        self._viewport_height_in_pixels = frame_height_in_pixels
        self._total_frame_count = total_frame_count
        self._current_frame_index = 0

        self._wall_canvas_width = frame_width_in_pixels + 440
        self._wall_canvas_height = frame_height_in_pixels + 160
        self._viewport_top_y = 100
        self._maximum_pan_offset_x = self._wall_canvas_width - frame_width_in_pixels

        random_number_generator = np.random.default_rng(random_seed)
        self._wall_canvas = self._build_textured_wall(random_number_generator)
        # Paintings: fixed (left_x, top_y, width, height) boxes on the wall, plus a
        # distinct fill colour, so each can be re-recognised by its features.
        self._paintings: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int]]] = {
            "painting_red": ((150, 250, 84, 104), (40, 40, 205)),
            "painting_blue": ((900, 230, 96, 86), (205, 120, 40)),
        }
        self._paint_paintings_onto_wall()
        self.current_ground_truth_bounding_boxes: dict[str, tuple[int, int, int, int]] = {}

    def _build_textured_wall(self, random_number_generator: np.random.Generator) -> np.ndarray:
        wall = random_number_generator.integers(
            50, 80, (self._wall_canvas_height, self._wall_canvas_width, 3), dtype=np.uint8
        )
        # Scatter small rectangles so the wall has the corners ORB needs to lock onto.
        for _ in range(900):
            patch_left_x = int(random_number_generator.integers(0, self._wall_canvas_width - 26))
            patch_top_y = int(random_number_generator.integers(0, self._wall_canvas_height - 26))
            patch_width = int(random_number_generator.integers(6, 24))
            patch_height = int(random_number_generator.integers(6, 24))
            patch_color = [int(channel) for channel in random_number_generator.integers(30, 200, 3)]
            cv2.rectangle(
                wall,
                (patch_left_x, patch_top_y),
                (patch_left_x + patch_width, patch_top_y + patch_height),
                patch_color,
                -1,
            )
        return wall

    def _paint_paintings_onto_wall(self) -> None:
        for (left_x, top_y, width, height), fill_color in self._paintings.values():
            cv2.rectangle(
                self._wall_canvas, (left_x, top_y), (left_x + width, top_y + height), fill_color, -1
            )
            cv2.rectangle(
                self._wall_canvas,
                (left_x, top_y),
                (left_x + width, top_y + height),
                (245, 245, 245),
                3,
            )
            cv2.line(
                self._wall_canvas,
                (left_x + 8, top_y + 8),
                (left_x + width - 8, top_y + height - 8),
                (245, 245, 245),
                2,
            )

    def _current_pan_offset_x(self) -> int:
        # Pan right to the far wall, then back -- a triangle wave so the camera
        # sweeps each painting out of view and later brings it back.
        pan_cycle_length = 200
        phase = self._current_frame_index % pan_cycle_length
        half_cycle = pan_cycle_length // 2
        if phase <= half_cycle:
            fraction = phase / half_cycle
        else:
            fraction = (pan_cycle_length - phase) / half_cycle
        return int(self._maximum_pan_offset_x * fraction)

    def read_next_frame(self) -> np.ndarray | None:
        if (
            self._total_frame_count is not None
            and self._current_frame_index >= self._total_frame_count
        ):
            return None
        self._current_frame_index += 1

        pan_offset_x = self._current_pan_offset_x()
        viewport_top_y = self._viewport_top_y
        viewport = self._wall_canvas[
            viewport_top_y : viewport_top_y + self._viewport_height_in_pixels,
            pan_offset_x : pan_offset_x + self._viewport_width_in_pixels,
        ].copy()

        self.current_ground_truth_bounding_boxes = {}
        for painting_identifier, painting_definition in self._paintings.items():
            (left_x, top_y, width, height), _fill_color = painting_definition
            viewport_left_x = left_x - pan_offset_x
            viewport_top_within = top_y - viewport_top_y
            visible_left_x = max(0, viewport_left_x)
            visible_top_y = max(0, viewport_top_within)
            visible_right_x = min(self._viewport_width_in_pixels, viewport_left_x + width)
            visible_bottom_y = min(self._viewport_height_in_pixels, viewport_top_within + height)
            if visible_right_x <= visible_left_x or visible_bottom_y <= visible_top_y:
                continue
            self.current_ground_truth_bounding_boxes[painting_identifier] = (
                visible_left_x,
                visible_top_y,
                visible_right_x - visible_left_x,
                visible_bottom_y - visible_top_y,
            )
        return viewport

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
