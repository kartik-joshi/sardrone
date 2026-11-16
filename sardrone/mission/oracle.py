"""A stand-in detector that reads the simulator's answer key.

The real detector (YOLO, Phase 3) does not exist yet, but the search
behaviour needs SOMETHING that says "I see a person". The oracle answers
that from ground truth: it projects each person's true 3D box into the
image and reports a sighting when the box is fully inside the picture and
tall enough to be detectable.

It is deliberately flagged everywhere (`detector: oracle` in the sidecar):
a mission that succeeds with the oracle shows that the FLYING logic works,
not that people can be found in images. The oracle also ignores occlusion,
so it "sees" a person under a tree.

Any detector with the same `detect()` method can replace it.
"""

from dataclasses import dataclass
from typing import Protocol

from sardrone.mcap.frames import Quat, Vec3, rotate_vector
from sardrone.mission.projection import BBox, CameraModel, DronePose, pixel_to_ground, project_box


@dataclass(frozen=True)
class Sighting:
    person_id: str | None  # the oracle knows who it is; a real detector will not
    bbox: BBox
    # Where the middle of the box's BOTTOM edge hits the ground: (east,
    # north) in `map`. The bottom edge is where the person touches the
    # ground, so this is close to their true position; the box centre would
    # be a point about waist-high, and a ray through it lands metres behind
    # the person when seen from a shallow angle.
    ground_east_north: tuple[float, float] | None
    confidence: float = 1.0  # the oracle is always sure; a real detector is not


class Detector(Protocol):
    name: str

    def detect(self, pose: DronePose, jpeg: bytes | None = None) -> list[Sighting]: ...


def box_corners(person: dict) -> list[tuple[float, float, float]]:
    """The 8 corners of a ground-truth person box (dict as in the sidecar)."""
    q = Quat(*person["box_quat_wxyz"])
    half = [size / 2 for size in person["box_size"]]
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                offset = rotate_vector(q, Vec3(sx * half[0], sy * half[1], sz * half[2]))
                corners.append(tuple(c + o for c, o in zip(person["box_centre"], offset)))
    return corners


class OracleDetector:
    name = "oracle"

    def __init__(
        self,
        people: list[dict],
        model: CameraModel,
        *,
        min_person_px: float = 12.0,
        ground_z: float = 0.0,
    ):
        self.people = people
        self.model = model
        self.min_person_px = min_person_px
        self.ground_z = ground_z  # height of the ground in `map` (below the takeoff origin)
        self._corners = {person["id"]: box_corners(person) for person in people}

    def detect(self, pose: DronePose, jpeg: bytes | None = None) -> list[Sighting]:
        sightings = []
        for person in self.people:
            bbox = project_box(self.model, pose, self._corners[person["id"]])
            if bbox is None or not bbox.inside(self.model.width, self.model.height):
                continue
            if max(bbox.width, bbox.height) < self.min_person_px:
                continue  # too small in the image to count as detectable
            u = (bbox.x0 + bbox.x1) / 2
            ground = pixel_to_ground(self.model, pose, u, bbox.y1, self.ground_z)
            sightings.append(Sighting(person["id"], bbox, ground))
        return sightings
