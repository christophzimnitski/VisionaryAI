"""A lightweight room map: persistent anchors for the big objects in a room.

Once the system locks onto a big object it drops an *anchor* for it in fixed room
(world) coordinates -- the pillow's place on the bed, the painting's place on the
wall. Anchors outlive the track that spawned them and outlive the object leaving
view, which is what lets the camera pan away and, on returning, ask a sharp
question: *the spot where I left this object is back on screen -- is the object
still there?*

The map deliberately stores only an anchor per object (world position, size, and
intrinsic feature signature), not a dense pixel mosaic. That keeps it cheap and
robust to lighting, and it reuses the same intrinsic features the rest of the
system already trusts for recognition. An anchor that is repeatedly expected in
view but not re-found is declared *missing* -- the negative change the
displacement reasoner pairs with a positive change to conclude that it moved.
"""

from __future__ import annotations

from dataclasses import dataclass

from visionary.configuration import TrackerConfiguration
from visionary.features.object_feature_memory import ObservedObjectFeatures

# How strongly an anchor's remembered size is pulled toward each fresh, confirmed
# observation -- slow, so a single noisy box never rewrites the anchor's geometry.
_ANCHOR_GEOMETRY_BLEND_TOWARD_NEW = 0.2


@dataclass
class WorldObjectAnchor:
    """One big object's persistent place and look within the room."""

    object_identity: str
    feature_signature: ObservedObjectFeatures
    world_center_x: float
    world_center_y: float
    box_width: int
    box_height: int
    created_frame_index: int
    last_confirmed_frame_index: int
    consecutive_refutation_count: int = 0

    @property
    def world_center(self) -> tuple[float, float]:
        return self.world_center_x, self.world_center_y


class WorldAnchorMemory:
    """Stores and maintains the room's big-object anchors in world coordinates."""

    def __init__(self, configuration: TrackerConfiguration) -> None:
        self._configuration = configuration
        self._anchors_by_identity: dict[str, WorldObjectAnchor] = {}

    # -- maintenance ----------------------------------------------------------

    def register_anchor(
        self,
        object_identity: str,
        feature_signature: ObservedObjectFeatures,
        world_center: tuple[float, float],
        box_size: tuple[int, int],
        frame_index: int,
    ) -> WorldObjectAnchor:
        """Drop (or refresh) an anchor for an object at a world position."""
        anchor = WorldObjectAnchor(
            object_identity=object_identity,
            feature_signature=feature_signature,
            world_center_x=world_center[0],
            world_center_y=world_center[1],
            box_width=box_size[0],
            box_height=box_size[1],
            created_frame_index=frame_index,
            last_confirmed_frame_index=frame_index,
        )
        self._anchors_by_identity[object_identity] = anchor
        return anchor

    def confirm_anchor(
        self,
        object_identity: str,
        world_center: tuple[float, float],
        feature_signature: ObservedObjectFeatures,
        frame_index: int,
    ) -> None:
        """Mark an anchor re-found this frame, refreshing its place and look.

        Re-finding clears the refutation count and gently corrects the anchor's
        world position and size toward the fresh observation, so an object that
        was nudged a little is followed without the anchor jumping around.
        """
        anchor = self._anchors_by_identity.get(object_identity)
        if anchor is None:
            return
        anchor.world_center_x = (
            (1.0 - _ANCHOR_GEOMETRY_BLEND_TOWARD_NEW) * anchor.world_center_x
            + _ANCHOR_GEOMETRY_BLEND_TOWARD_NEW * world_center[0]
        )
        anchor.world_center_y = (
            (1.0 - _ANCHOR_GEOMETRY_BLEND_TOWARD_NEW) * anchor.world_center_y
            + _ANCHOR_GEOMETRY_BLEND_TOWARD_NEW * world_center[1]
        )
        anchor.feature_signature = feature_signature
        anchor.last_confirmed_frame_index = frame_index
        anchor.consecutive_refutation_count = 0

    def refute_anchor(self, object_identity: str) -> bool:
        """Count one frame where an in-view anchor was not re-found.

        Returns ``True`` once the anchor has been refuted on enough consecutive
        in-view frames to be declared missing (a few frames of patience absorbs a
        single bad detection or a brief self-occlusion).
        """
        anchor = self._anchors_by_identity.get(object_identity)
        if anchor is None:
            return False
        anchor.consecutive_refutation_count += 1
        return (
            anchor.consecutive_refutation_count
            >= self._configuration.anchor_consecutive_refutations_before_declared_missing
        )

    def discard_anchor(self, object_identity: str) -> WorldObjectAnchor | None:
        """Remove an anchor (e.g. once its disappearance has been recorded)."""
        return self._anchors_by_identity.pop(object_identity, None)

    # -- queries --------------------------------------------------------------

    def anchors_expected_in_view(
        self, world_offset: tuple[float, float], frame_width: int, frame_height: int
    ) -> list[WorldObjectAnchor]:
        """Anchors whose whole box currently sits comfortably inside the viewport.

        The full projected box -- not just the centre -- must clear the frame by
        the configured border margin, so an anchor is judged present/absent only
        when it is wholly on screen, never while it is clipped by an edge. A
        clipped object's detection box drifts inward, which would otherwise both
        corrupt the anchor's remembered place and trigger a false "missing".
        """
        border_margin = self._configuration.anchor_in_view_border_margin_in_pixels
        world_offset_x, world_offset_y = world_offset
        anchors_in_view: list[WorldObjectAnchor] = []
        for anchor in self._anchors_by_identity.values():
            frame_left_x = anchor.world_center_x - anchor.box_width / 2 - world_offset_x
            frame_top_y = anchor.world_center_y - anchor.box_height / 2 - world_offset_y
            frame_right_x = frame_left_x + anchor.box_width
            frame_bottom_y = frame_top_y + anchor.box_height
            if (
                frame_left_x >= border_margin
                and frame_top_y >= border_margin
                and frame_right_x <= frame_width - border_margin
                and frame_bottom_y <= frame_height - border_margin
            ):
                anchors_in_view.append(anchor)
        return anchors_in_view

    def get_anchor(self, object_identity: str) -> WorldObjectAnchor | None:
        return self._anchors_by_identity.get(object_identity)

    def has_anchor(self, object_identity: str) -> bool:
        return object_identity in self._anchors_by_identity

    @property
    def all_anchors(self) -> list[WorldObjectAnchor]:
        return list(self._anchors_by_identity.values())

    @property
    def anchor_count(self) -> int:
        return len(self._anchors_by_identity)
