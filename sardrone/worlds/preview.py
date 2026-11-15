"""Look at a generated world without flying: render its fixed preview cameras.

    make preview ROW=pilot-meadow    # -> data/worlds/pilot-meadow/preview_*.jpg

Starts Gazebo only (no PX4), takes one picture from each preview camera the
generator placed in the world, and stops Gazebo again. A flight takes a
minute; this takes a few seconds, which makes it the quick check that trees,
people, clothing and lighting came out as intended.
"""

import argparse
import sys
import time
from pathlib import Path

from sardrone.sim.camera import CameraError, GazeboCamera
from sardrone.sim.launch import Simulation, SimulationError, gz_python


def render_previews(world_dir: Path, settle_s: float = 3.0) -> list[Path]:
    world_dir = Path(world_dir)
    written = []
    with Simulation(world_dir, with_px4=False) as sim:
        time.sleep(settle_s)  # let textures and shadows finish loading
        for name in sim.spec["preview_cameras"]:
            with GazeboCamera(
                model=name, world=sim.spec["gz_world"], gz_python=gz_python(), start_timeout_s=40
            ) as camera:
                time.sleep(1.0)
                path = world_dir / f"{name}.jpg"
                path.write_bytes(camera.read())
                written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a world's preview cameras.")
    parser.add_argument("--world-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        for path in render_previews(args.world_dir):
            print(f"wrote {path}")
    except (SimulationError, CameraError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
