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
from sardrone.worlds.assets import CLOTHING, CLOTHING_RANDOMIZED, PEOPLE, ROCK_MODEL, TREES, VEHICLE_MODELS
from sardrone.worlds.matrix import POSES, POSES_RICH, VariationRow
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
class ClutterSpec:
    """A thing on the ground that is NOT a person: a distractor.

    Clutter gives the detector chances to be wrong during training (a rock
    in shade, a bright box, a log the length of a body). Without it, in
    these simple worlds "anything that is not grass or tree" would be a
    person, which no real scene allows.
    """

    kind: str  # rock | log | bush | box | vehicle
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    size: tuple[float, float, float]  # sphere: radii; box: sides; log: (radius, radius, length)
    rgb: tuple[float, float, float]


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
    scale: float = 1.0  # body size relative to the mesh (domain randomization)

    @property
    def label(self) -> int:
        """The number Gazebo reports for this person: 1 for person_1, ..."""
        return int(self.id.rsplit("_", 1)[1])

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
    clutter: tuple[ClutterSpec, ...] = ()
    randomize: bool = False
    gz_world: str = GZ_WORLD_NAME
    gz_model: str = DRONE_NAME
    camera_mount_m: tuple[float, float, float] = CAMERA_MOUNT_M
    drone_body_height_m: float = DRONE_BODY_HEIGHT_M
    boxes_topic: str = "/sar/boxes"  # Gazebo's label boxes for the drone camera
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        document = asdict(self)
        for person, entry in zip(self.people, document["people"]):
            entry["label"] = person.label
        return document


# ---- helpers -----------------------------------------------------------------


def _far_enough(x: float, y: float, others, spacing: float) -> bool:
    return all(math.hypot(x - ox, y - oy) >= spacing for ox, oy in others)


def _sample(rng: random.Random, accept, draw, what: str, tries: int = 4000):
    for _ in range(tries):
        x, y = draw()
        if accept(x, y):
            return round(x, 3), round(y, 3)
    raise RuntimeError(f"could not place {what} after {tries} tries: the area is too crowded")


def place_person(pose: str, x: float, y: float, yaw: float, face_up: bool = True, scale: float = 1.0):
    """SDF pose and ground-truth box for one person standing/sitting/lying at (x, y).

    `scale` makes the body bigger or smaller. A mesh is scaled about its
    own origin, so every measure taken from the mesh scales with it.
    """
    asset = PEOPLE[pose]
    size = tuple(round(s * scale, 3) for s in asset.size)
    if pose == "lying":
        # The mesh stands upright. Rolling it 90 deg lays it flat; it then
        # rests on what was its back (or front), so lift it by half its depth.
        roll = -math.pi / 2 if face_up else math.pi / 2
        z = size[1] / 2 + 0.02
    else:
        roll = 0.0
        z = -asset.lo[2] * scale  # put the lowest point of the mesh on the ground
    q = quat_from_rpy(roll, 0.0, yaw)
    offset = rotate_vector(q, Vec3(*(c * scale for c in asset.centre)))
    centre = tuple(round(p + o, 3) for p, o in zip((x, y, z), offset))
    quat = tuple(round(v, 6) for v in (q.w, q.x, q.y, q.z))
    return (round(z, 3), round(roll, 6)), centre, size, quat


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
    choices = POSES_RICH if row.content == "rich" else POSES
    poses = list(row.poses) or [rng.choice(choices) for _ in range(row.people)]
    colours = list(CLOTHING_RANDOMIZED if row.randomize else CLOTHING)
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
        face_up = rng.random() < 0.5
        # Drawn only when randomizing, so plain rows keep their random stream.
        scale = round(rng.uniform(0.85, 1.1), 3) if row.randomize else 1.0
        (z, roll), centre, size, quat = place_person(pose, x, y, yaw, face_up=face_up, scale=scale)
        people.append(
            PersonSpec(
                id=f"person_{index}",
                pose=pose,
                clothing=rng.choice(colours) if PEOPLE[pose].recolour else "as modelled",
                occlusion=row.occlusion,
                x=x, y=y, z=z, roll=roll, pitch=0.0, yaw=yaw,
                box_centre=centre, box_size=size, box_quat_wxyz=quat, scale=scale,
            )  # fmt: skip
        )
    return people


# ---- vehicles (distractors) ---------------------------------------------------

# Common car body colours (0..1 RGB): red, silver, white, black, navy, grey, blue.
VEHICLE_COLOURS = [
    (0.75, 0.08, 0.10), (0.72, 0.73, 0.75), (0.92, 0.92, 0.92), (0.07, 0.07, 0.08),
    (0.10, 0.14, 0.35), (0.40, 0.41, 0.43), (0.15, 0.30, 0.65),
]  # fmt: skip
VEHICLE_SIZE_M = (4.2, 1.8, 1.5)  # length, width, height of a small car


def _place_vehicles(row: VariationRow, people: list, trees: list, rng: random.Random) -> list[ClutterSpec]:
    """`row.vehicles` parked cars, clear of people (by 4 m), trees and takeoff."""
    half = row.area_m / 2 - 4.0
    vehicles: list[ClutterSpec] = []
    for _ in range(row.vehicles):
        for _ in range(200):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            if (
                math.hypot(x, y) >= TAKEOFF_CLEAR_M + 3.0
                and all(math.hypot(x - p.x, y - p.y) >= 4.0 for p in people)
                and all(math.hypot(x - t.x, y - t.y) >= t.canopy_radius_m + 2.5 for t in trees)
                and all(math.hypot(x - v.x, y - v.y) >= 5.0 for v in vehicles)
            ):
                break
        else:
            continue  # no room left; fewer vehicles rather than overlapping ones
        if row.content == "rich":  # a real vehicle mesh; size holds its scale
            model = rng.choice(sorted(VEHICLE_MODELS))
            vehicles.append(ClutterSpec(f"vehicle:{model}", round(x, 3), round(y, 3), 0.0, 0.0, 0.0,
                                        round(rng.uniform(0, 2 * math.pi), 4), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)))
            continue
        length, width, height = VEHICLE_SIZE_M
        vehicles.append(
            ClutterSpec(
                "vehicle", round(x, 3), round(y, 3), 0.0, 0.0, 0.0, round(rng.uniform(0, 2 * math.pi), 4),
                (round(length * rng.uniform(0.9, 1.1), 3), round(width, 3), round(height, 3)),
                rng.choice(VEHICLE_COLOURS),
            )  # fmt: skip
        )
    return vehicles


# ---- clutter (domain randomization) -------------------------------------------

CLUTTER_COLOURS = {
    "rock": [(0.45, 0.45, 0.45), (0.6, 0.58, 0.52), (0.35, 0.33, 0.3)],
    "log": [(0.35, 0.24, 0.14), (0.45, 0.33, 0.2)],
    "bush": [(0.12, 0.3, 0.1), (0.2, 0.36, 0.12), (0.3, 0.33, 0.12)],
}


def _place_clutter(row: VariationRow, people: list[PersonSpec], rng: random.Random) -> list[ClutterSpec]:
    half = row.area_m / 2
    clutter: list[ClutterSpec] = []
    for _ in range(rng.randint(15, 45)):
        kind = rng.choice(["rock", "rock", "log", "bush", "bush", "box"])
        for _ in range(50):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            if math.hypot(x, y) >= TAKEOFF_CLEAR_M and all(math.hypot(x - p.x, y - p.y) >= 1.5 for p in people):
                break
        else:
            continue
        yaw = round(rng.uniform(0, 2 * math.pi), 4)
        if kind == "rock":
            r = rng.uniform(0.2, 0.8)
            size = (round(r, 3), round(r * rng.uniform(0.6, 1.0), 3), round(r * rng.uniform(0.4, 0.8), 3))
            z, roll, pitch, rgb = size[2] * 0.6, 0.0, 0.0, rng.choice(CLUTTER_COLOURS["rock"])
        elif kind == "log":
            radius, length = round(rng.uniform(0.1, 0.22), 3), round(rng.uniform(1.0, 3.0), 3)
            size = (radius, radius, length)
            z, roll, pitch, rgb = radius, 0.0, math.pi / 2, rng.choice(CLUTTER_COLOURS["log"])  # lying down
        elif kind == "bush":
            r = rng.uniform(0.4, 1.2)
            size = (round(r, 3), round(r * rng.uniform(0.7, 1.0), 3), round(r * rng.uniform(0.5, 0.8), 3))
            z, roll, pitch, rgb = size[2] * 0.7, 0.0, 0.0, rng.choice(CLUTTER_COLOURS["bush"])
        else:  # a man-made box or tarp in any colour, including clothing-like ones
            size = (round(rng.uniform(0.4, 2.0), 3), round(rng.uniform(0.4, 1.5), 3), round(rng.uniform(0.05, 0.6), 3))
            z, roll, pitch = size[2] / 2, 0.0, 0.0
            rgb = tuple(round(rng.uniform(0.05, 0.95), 3) for _ in range(3))
        clutter.append(
            ClutterSpec(kind, round(x, 3), round(y, 3), round(z, 3), roll, round(pitch, 6), yaw, size, tuple(rgb))
        )
    return clutter


def _place_rock_meshes(row: VariationRow, people: list, trees: list, rng: random.Random) -> list[ClutterSpec]:
    """3 to 8 rock meshes (content: rich), 0.3 to 1.2 times the model's 1.5 m,
    clear of people by 1.5 m and of the takeoff point. Real search photos are
    full of stone; the basic worlds have none that looks like it."""
    half = row.area_m / 2 - 2.0
    rocks: list[ClutterSpec] = []
    for _ in range(rng.randint(3, 8)):
        for _ in range(100):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            if math.hypot(x, y) >= TAKEOFF_CLEAR_M and all(math.hypot(x - p.x, y - p.y) >= 1.5 for p in people):
                break
        else:
            continue
        scale = round(rng.uniform(0.3, 1.2), 3)
        z = round(-ROCK_MODEL.lo[2] * scale * 0.7, 3)  # partly sunk into the ground
        rocks.append(ClutterSpec("rock_mesh", round(x, 3), round(y, 3), z, 0.0, 0.0,
                                 round(rng.uniform(0, 2 * math.pi), 4), (scale, scale, scale), (1.0, 1.0, 1.0)))
    return rocks


def build_world(row: VariationRow) -> WorldSpec:
    """The whole world for one matrix row. Same row -> same world, always."""
    trees = _place_trees(row, random.Random(f"{row.seed}:trees"))
    people = _place_people(row, trees, random.Random(f"{row.seed}:people"))
    sun = sun_for(row.time_of_day, random.Random(f"{row.seed}:sun"), randomize=row.randomize)
    clutter = _place_clutter(row, people, random.Random(f"{row.seed}:clutter")) if row.randomize else []
    # Vehicles have their own random stream, so adding them moves nothing else.
    clutter += _place_vehicles(row, people, trees, random.Random(f"{row.seed}:vehicles"))
    if row.content == "rich":
        clutter += _place_rock_meshes(row, people, trees, random.Random(f"{row.seed}:rocks"))

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
        clutter=tuple(clutter),
        randomize=row.randomize,
    )
