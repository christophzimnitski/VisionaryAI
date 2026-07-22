"""End-to-end test of the room monitor: the missing pillow is found on the floor.

A camera looks at a bed with two identical pillows, pans away to a desk, and while
the bed is out of view one pillow is moved to the floor. On panning back, the
monitor must notice the bed pillow is gone (a disappearance), having earlier seen
a matching pillow appear by the desk (an appearance), and conclude they are the
same object -- the whole point of the system, with the move never observed.
"""

from __future__ import annotations

import dataclasses

from visionary.configuration import TrackerConfiguration
from visionary.detection.salient_region_detector import SalientRegionDetector
from visionary.io.video_sources import RoomWithMovableObjectsSceneSource
from visionary.tracking.room_object_monitor import RoomObjectMonitor


def _run_monitor_over_scene() -> RoomObjectMonitor:
    configuration = dataclasses.replace(
        TrackerConfiguration(), enable_camera_motion_compensation=True
    )
    scene_source = RoomWithMovableObjectsSceneSource(total_frame_count=132, random_seed=0)
    monitor = RoomObjectMonitor(SalientRegionDetector(configuration), configuration)
    while True:
        frame = scene_source.read_next_frame()
        if frame is None:
            break
        monitor.observe_frame(frame)
    return monitor


def test_a_pillow_moved_while_out_of_view_is_recognised_on_the_floor() -> None:
    monitor = _run_monitor_over_scene()
    conclusions = monitor._displacement_reasoner.displacement_conclusions

    # Exactly the bed-to-floor move should be concluded (the desk merely appearing
    # must not be mistaken for a displacement).
    bed_to_floor_conclusions = [
        conclusion
        for conclusion in conclusions
        if conclusion.from_world_position[0] < 400
        and conclusion.to_world_position[0] > 600
        and conclusion.feature_similarity > 0.8
    ]
    assert len(bed_to_floor_conclusions) == 1

    conclusion = bed_to_floor_conclusions[0]
    # Vanished from the bed (right pillow ~ world x 242), found on the floor by the
    # desk (~ world x 792).
    assert 200 < conclusion.from_world_position[0] < 300
    assert conclusion.to_world_position[0] > 700
    # The move happened off-camera, so many frames separate the two observations.
    assert conclusion.frames_between_observations > 20


def test_no_false_displacement_when_nothing_is_moved() -> None:
    # The same room and pan, but the pillow is never moved: the monitor must not
    # invent a displacement out of objects merely entering and leaving view.
    configuration = dataclasses.replace(
        TrackerConfiguration(), enable_camera_motion_compensation=True
    )
    scene_source = RoomWithMovableObjectsSceneSource(total_frame_count=132, random_seed=0)
    # Disable the scripted move by pushing it past the end of the run.
    scene_source._scene_mutation_frame_index = 10_000
    monitor = RoomObjectMonitor(SalientRegionDetector(configuration), configuration)
    while True:
        frame = scene_source.read_next_frame()
        if frame is None:
            break
        monitor.observe_frame(frame)

    assert monitor._displacement_reasoner.displacement_conclusions == []
