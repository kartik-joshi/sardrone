"""Decide where everything goes: VariationRow -> WorldSpec.

Coordinates are Gazebo's world frame, which this project sets up as ENU:
x = east, y = north, z = up, in metres. The drone takes off from (0, 0), so
world coordinates and the episode's `map` frame agree (apart from the small
height of the drone's body above the ground, see `DRONE_BODY_HEIGHT_M`).

Placement is "rejection sampling": pick a random spot, keep it if it obeys
the rules (not too close to anything else, inside the area), otherwise try
again. Each kind of object has its own random stream, so changing the number
of people does not move the trees.
"""

import math
import random
from dataclasses import asdict, dataclass, field

from sardrone.mcap.frames import Quat, Vec3, quat_from_rpy, rotate_vector
from sardrone.worlds.assets import CLOTHING, PEOPLE, TREES
from sardrone.worlds.matrix import POSES, VariationRow
from sardrone.worlds.sun import SunSpec, sun_for

GZ_WORLD_NAME = "sar"
DRONE_MODEL = "x500_sar_cam"
DRONE_NAME = "x500_sar_cam_0"
# Where the camera sits relative to the drone's body origin (`base_link`):
# forward, left, up, in metres. Low at the front, so that a camera looking
# 45 deg down does not see the drone itself.
CAMERA_MOUNT_M = (0.16, 0.0, -0.04)
# PX4's x500 model puts `base_link` 0.24 m above the model's own origin
# (x500_base/model.sdf, <pose>0 0 .24 ...). SDF poses inside the drone model
# are given from the model origin, so this is added when writing them.
BASE_LINK_IN_MODEL_Z_M = 0.24
# Height of `base_link` above the ground when the drone stands on its legs.
# PX4 calls the takeoff point z = 0, so `map` z = world z - this. Measured
# in Gazebo: the model origin settles 0.013 m below ground level.
DRONE_BODY_HEIGHT_M = 0.227

TAKEOFF_CLEAR_M = 8.0  # nothing is placed this close to the takeoff point
PERSON_SPACING_M = 2.5
TREE_SPACING_M = 3.0
MIN_CLEARANCE_ABOVE_TREES_M = 5.0

TREE_SCALE = {"oak": (0.8, 1.3), "pine": (1.0, 1.8)}


@dataclass(frozen=True)
class TreeSpec:
    kind: str
    x: float
    y: float
    yaw: float
    scale: float

    @property
    def canopy_radius_m(self) -> float:
        asset = TREES[self.kind]
        return max(abs(asset.lo[0]), abs(asset.hi[0]), abs(asset.lo[1]), abs(asset.hi[1])) * self.scale

    @property
    def height_m(self) -> float:
        return TREES[self.kind].hi[2] * self.scale


@dataclass(frozen=True)
class PersonSpec:
    id: str
    pose: str  # standing | sitting | lying
    clothing: str
    occlusion: str  # none | partial
    # Model placement, exactly as written to the SDF <pose>.
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    # Ground truth: an oriented 3D box around the body, in world coordinates.
    box_centre: tuple[float, float, float]
    box_size: tuple[float, float, float]  # along the box's own x, y, z axes
    box_quat_wxyz: tuple[float, float, float, float]

    @property
    def height_m(self) -> float:
        """How tall the person stands above the ground, in this pose."""
        return max(corner[2] for corner in self.box_corners())

    def box_corners(self) -> list[tuple[float, float, float]]:
        """The 8 corners of the ground-truth box in world coordinates."""
        q = Quat(*self.box_quat_wxyz)
        half = [s / 2 for s in self.box_size]
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    offset = rotate_vector(q, Vec3(sx * half[0], sy * half[1], sz * half[2]))
                    corners.append(tuple(c + o for c, o in zip(self.box_centre, offset)))
        return corners


@dataclass(frozen=True)
class WorldSpec:
    id: str
    world: str
    seed: int
    area_m: float
    time_of_day: str
    weather: str
    altitude_m: float
    camera_pitch_deg: float
    occlusion: str
    sun: SunSpec
    trees: tuple[TreeSpec, ...]
    people: tuple[PersonSpec, ...]
    gz_world: str = GZ_WORLD_NAME
    gz_model: str = DRONE_NAME
    camera_mount_m: tuple[float, float, float] = CAMERA_MOUNT_M
    drone_body_height_m: float = DRONE_BODY_HEIGHT_M
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- helpers -----------------------------------------------------------------


def _far_enough(x: float, y: float, others, spacing: float) -> bool:
    return all(math.hypot(x - ox, y - oy) >= spacing for ox, oy in others)


def _sample(rng: random.Random, accept, draw, what: str, tries: int = 4000):
    for _ in range(tries):
        x, y = draw()
        if accept(x, y):
            return round(x, 3), round(y, 3)
    raise RuntimeError(f"could not place {what} after {tries} tries: the area is too crowded")


def place_person(pose: str, x: float, y: float, yaw: float, face_up: bool = True):
    """SDF pose and ground-truth box for one person standing/sitting/lying at (x, y)."""
    asset = PEOPLE[pose]
    if pose == "lying":
        # The mesh stands upright. Rolling it 90 deg lays it flat; it then
        # rests on what was its back (or front), so lift it by half its depth.
        roll = -math.pi / 2 if face_up else math.pi / 2
        z = asset.size[1] / 2 + 0.02
    else:
        roll = 0.0
        z = -asset.lo[2]  # put the lowest point of the mesh on the ground
    q = quat_from_rpy(roll, 0.0, yaw)
    offset = rotate_vector(q, Vec3(*asset.centre))
    centre = tuple(round(p + o, 3) for p, o in zip((x, y, z), offset))
    quat = tuple(round(v, 6) for v in (q.w, q.x, q.y, q.z))
    return (round(z, 3), round(roll, 6)), centre, asset.size, quat


# ---- trees -------------------------------------------------------------------


def _tree(rng: random.Random, kind: str, x: float, y: float) -> TreeSpec:
    low, high = TREE_SCALE[kind]
    return TreeSpec(kind, x, y, round(rng.uniform(0, 2 * math.pi), 4), round(rng.uniform(low, high), 3))


def _place_trees(row: VariationRow, rng: random.Random) -> list[TreeSpec]:
    half = row.area_m / 2
    trees: list[TreeSpec] = []

    def ok(x, y):
        return (
            abs(x) <= half
            and abs(y) <= half
            and math.hypot(x, y) >= TAKEOFF_CLEAR_M + 6.0
            and _far_enough(x, y, [(t.x, t.y) for t in trees], TREE_SPACING_M)
        )

    def add(kind, draw, what):
        x, y = _sample(rng, ok, draw, what)
        trees.append(_tree(rng, kind, x, y))

    anywhere = lambda: (rng.uniform(-half, half), rng.uniform(-half, half))  # noqa: E731

    if row.world == "meadow":
        # Open grass with a few lone trees.
        for _ in range(rng.randint(3, 6)):
            add(rng.choice(["oak", "oak", "pine"]), anywhere, "meadow tree")

    elif row.world == "forest_edge":
        # A dense band of trees along the north side, thinning out southwards.
        band = row.area_m * 0.3
        count = max(8, int(row.area_m * band / 110))
        in_band = lambda: (rng.uniform(-half, half), half - abs(rng.gauss(0, band / 2)))  # noqa: E731
        for _ in range(count):
            add(rng.choice(["pine", "pine", "oak"]), in_band, "forest tree")
        for _ in range(3):
            add("oak", anywhere, "stray tree")

    elif row.world == "park":
        # Small groups of trees, like planted clumps.
        for _ in range(rng.randint(4, 6)):
            cx, cy = _sample(rng, ok, anywhere, "tree group")
            near = lambda cx=cx, cy=cy: (cx + rng.gauss(0, 5), cy + rng.gauss(0, 5))  # noqa: E731
            for _ in range(rng.randint(2, 4)):
                add(rng.choice(["oak", "pine"]), near, "park tree")
    return trees


# ---- people ------------------------------------------------------------------


def _place_people(row: VariationRow, trees: list[TreeSpec], rng: random.Random) -> list[PersonSpec]:
    half = row.area_m / 2 - 2.0
    poses = list(row.poses) or [rng.choice(POSES) for _ in range(row.people)]
    colours = list(CLOTHING)
    people: list[PersonSpec] = []

    for index, pose in enumerate(poses, start=1):
        taken = [(p.x, p.y) for p in people]

        def inside(x, y):
            return (
                abs(x) <= half
                and abs(y) <= half
                and math.hypot(x, y) >= TAKEOFF_CLEAR_M
                and _far_enough(x, y, taken, PERSON_SPACING_M)
            )

        if row.occlusion == "partial":
            # Under a canopy: 25..75 % of the way from the trunk to its edge.
            def draw():
                tree = rng.choice(trees)
                distance = tree.canopy_radius_m * rng.uniform(0.25, 0.75)
                angle = rng.uniform(0, 2 * math.pi)
                return tree.x + distance * math.cos(angle), tree.y + distance * math.sin(angle)

            accept = inside
        else:
            # In the open: clear of every canopy by at least 2 m.
            def draw():
                return rng.uniform(-half, half), rng.uniform(-half, half)

            def accept(x, y):
                return inside(x, y) and all(
                    math.hypot(x - t.x, y - t.y) >= t.canopy_radius_m + 2.0 for t in trees
                )

        x, y = _sample(rng, accept, draw, f"person {index}")
        yaw = round(rng.uniform(0, 2 * math.pi), 4)
        (z, roll), centre, size, quat = place_person(pose, x, y, yaw, face_up=rng.random() < 0.5)
        people.append(
            PersonSpec(
                id=f"person_{index}",
                pose=pose,
                clothing=rng.choice(colours),
                occlusion=row.occlusion,
                x=x, y=y, z=z, roll=roll, pitch=0.0, yaw=yaw,
                box_centre=centre, box_size=size, box_quat_wxyz=quat,
            )  # fmt: skip
        )
    return people


def build_world(row: VariationRow) -> WorldSpec:
    """The whole world for one matrix row. Same row -> same world, always."""
    trees = _place_trees(row, random.Random(f"{row.seed}:trees"))
    people = _place_people(row, trees, random.Random(f"{row.seed}:people"))
    sun = sun_for(row.time_of_day, random.Random(f"{row.seed}:sun"))

    tallest = max((t.height_m for t in trees), default=0.0)
    if row.altitude_m - tallest < MIN_CLEARANCE_ABOVE_TREES_M:
        raise ValueError(
            f"{row.id}: flying at {row.altitude_m} m leaves less than "
            f"{MIN_CLEARANCE_ABOVE_TREES_M} m above the tallest tree ({tallest:.1f} m)"
        )
    return WorldSpec(
        id=row.id,
        world=row.world,
        seed=row.seed,
        area_m=row.area_m,
        time_of_day=row.time_of_day,
        weather=row.weather,
        altitude_m=row.altitude_m,
        camera_pitch_deg=row.camera_pitch_deg,
        occlusion=row.occlusion,
        sun=sun,
        trees=tuple(trees),
        people=tuple(people),
    )
