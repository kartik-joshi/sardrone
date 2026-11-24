"""Generate world folders from a variation matrix.

    make world ROW=pilot-meadow                 # one row of configs/pilot.yaml
    python -m sardrone.worlds.generate --all    # every row

Each world gets its own folder:

    data/worlds/<id>/world.sdf       the Gazebo world
    data/worlds/<id>/spec.json       everything about it, incl. ground truth
    data/worlds/<id>/ground.jpg      the seeded ground texture
    data/worlds/<id>/models/         the drone with its pitched camera
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from sardrone.worlds import assets
from sardrone.worlds.layout import DRONE_MODEL, WorldSpec, build_world
from sardrone.worlds.matrix import VariationRow, find_row, load_matrix
from sardrone.worlds.sdf import DRONE_MODEL_CONFIG, PreviewCamera, drone_model_sdf, world_sdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "pilot.yaml"
DEFAULT_WORLD_ROOT = PROJECT_ROOT / "data" / "worlds"


@dataclass(frozen=True)
class WorldPaths:
    folder: Path

    @property
    def sdf(self) -> Path:
        return self.folder / "world.sdf"

    @property
    def spec(self) -> Path:
        return self.folder / "spec.json"

    @property
    def models(self) -> Path:
        return self.folder / "models"

    @property
    def ground(self) -> Path:
        return self.folder / "ground.jpg"

    @property
    def ground_far(self) -> Path:
        return self.folder / "ground_far.png"


def default_preview_cameras(spec: WorldSpec) -> tuple[PreviewCamera, ...]:
    """An overview from the south edge, plus a close look at the first person
    from the flight altitude and camera pitch of this world."""
    half = spec.area_m / 2
    cameras = [PreviewCamera("preview_overview", 0.0, -half - 10.0, half * 0.9, 38.0, 90.0)]
    for person in spec.people:
        # Stand back so that a camera pitched down sees the person at the
        # image centre: horizontal distance = height / tan(pitch).
        back = spec.altitude_m / math.tan(math.radians(spec.camera_pitch_deg))
        cameras.append(
            PreviewCamera(
                f"preview_{person.id}", person.x, person.y - back, spec.altitude_m,
                spec.camera_pitch_deg, 90.0,
            )  # fmt: skip
        )
    return tuple(cameras)


def load_spec(world_dir: Path | str) -> dict:
    return json.loads((Path(world_dir) / "spec.json").read_text())


def generate(
    row: VariationRow,
    world_root: Path = DEFAULT_WORLD_ROOT,
    *,
    fuel_dir: Path = assets.FUEL_DIR,
    variant_dir: Path = assets.VARIANT_DIR,
    preview: bool = True,
) -> WorldPaths:
    """Build one world folder. Running it again gives identical files."""
    spec = build_world(row)
    paths = WorldPaths(Path(world_root) / row.id)
    paths.folder.mkdir(parents=True, exist_ok=True)

    texture = assets.ground_texture(spec.world, spec.seed, randomize=spec.randomize)
    texture.save(paths.ground, quality=90)
    assets.far_ground_texture(texture).save(paths.ground_far)

    for tree_kind in {tree.kind for tree in spec.trees}:
        assets.require(assets.TREES[tree_kind], fuel_dir)
    for item in spec.clutter:  # mesh distractors (content: rich)
        if item.kind.startswith("vehicle:"):
            assets.require(assets.VEHICLE_MODELS[item.kind.split(":", 1)[1]], fuel_dir)
        elif item.kind == "rock_mesh":
            assets.require(assets.ROCK_MODEL, fuel_dir)
    variants = {
        person.id: assets.clothing_variant(person.pose, person.clothing, fuel_dir, variant_dir)
        for person in spec.people
    }
    cameras = default_preview_cameras(spec) if preview else ()

    paths.sdf.write_text(
        world_sdf(
            spec,
            fuel_dir=fuel_dir,
            variant_dirs=variants,
            ground_texture=paths.ground,
            ground_far_texture=paths.ground_far,
            preview_cameras=cameras,
        )
    )

    drone_dir = paths.models / DRONE_MODEL
    drone_dir.mkdir(parents=True, exist_ok=True)
    (drone_dir / "model.sdf").write_text(drone_model_sdf(spec.camera_pitch_deg, spec.camera_mount_m))
    (drone_dir / "model.config").write_text(DRONE_MODEL_CONFIG)

    document = spec.to_dict()
    document["preview_cameras"] = [camera.name for camera in cameras]
    paths.spec.write_text(json.dumps(document, indent=2) + "\n")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Gazebo worlds from a variation matrix.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--row", help="episode id to generate")
    parser.add_argument("--all", action="store_true", help="generate every row")
    parser.add_argument("--out", type=Path, default=DEFAULT_WORLD_ROOT)
    parser.add_argument("--fetch", action="store_true", help="download missing models first")
    args = parser.parse_args(argv)

    rows = load_matrix(args.matrix)
    if args.fetch:
        assets.fetch_all()
    if not args.all and not args.row:
        print("rows:", ", ".join(row.id for row in rows))
        parser.error("pass --row ID or --all")
    for row in rows if args.all else [find_row(rows, args.row)]:
        paths = generate(row, args.out)
        spec = load_spec(paths.folder)
        print(
            f"{row.id}: {spec['world']}, {len(spec['trees'])} trees, "
            f"{len(spec['people'])} people, {spec['time_of_day']} -> {paths.sdf}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
