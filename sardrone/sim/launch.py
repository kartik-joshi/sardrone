"""Start Gazebo on a generated world, and PX4 attached to the drone in it.

    make sim-world ROW=pilot-meadow        # leave running; Ctrl-C stops both

Why not `make px4_sitl gz_x500`? That command starts Gazebo on one of PX4's
own worlds and spawns one of PX4's own models. Our world and our drone model
live in this repo, so we start Gazebo ourselves and tell PX4 (through
environment variables its start script reads) to ATTACH to the drone that is
already in the world. No file inside PX4-Autopilot is changed.

Checked in PX4's ROMFS/px4fmu_common/init.d-posix/px4-rc.gzsim: if a Gazebo
world is already running it is reused, and if PX4_GZ_MODEL_NAME is set PX4
attaches to that model instead of spawning one.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from sardrone.worlds.generate import load_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def px4_dir() -> Path:
    return Path(os.environ.get("PX4_DIR") or PROJECT_ROOT.parent / "PX4-Autopilot").resolve()


def gz_python() -> Path:
    return px4_dir() / "Tools/setup/macos/.pixi/envs/default/bin/python"


class SimulationError(RuntimeError):
    pass


def _build_dir() -> Path:
    build = px4_dir() / "build" / "px4_sitl_default"
    if not (build / "bin" / "px4").exists():
        raise SimulationError(
            f"PX4 is not built in {build}. See 'PX4 setup' in the README, or set PX4_DIR."
        )
    return build


def gazebo_command(world_dir: Path) -> list[str]:
    """Shell command that runs the Gazebo server on world_dir/world.sdf.

    PX4's generated gz_env.sh puts Gazebo on the PATH and adds PX4's models
    and plugins; our own models folder is put in front of them.
    """
    world_dir = Path(world_dir).resolve()
    script = (
        f'export GZ_SIM_RESOURCE_PATH="{world_dir / "models"}"; '
        f'source "{_build_dir() / "gz_env.sh"}"; '
        "export GZ_IP=127.0.0.1; "
        f'exec gz sim -s -r --verbose=1 "{world_dir / "world.sdf"}"'
    )
    return ["/bin/bash", "-c", script]


def px4_environment(spec: dict) -> dict:
    env = dict(os.environ)
    env.update(
        GZ_IP="127.0.0.1",
        PX4_SIM_MODEL="gz_x500",  # selects the x500 airframe parameters
        PX4_GZ_MODEL_NAME=spec["gz_model"],  # attach, do not spawn
        PX4_GZ_WORLD=spec["gz_world"],
        HEADLESS="1",
    )
    return env


def _shown(path: Path) -> str:
    """A path for messages that end up in committed manifests: relative to the
    project when it is inside it, so no home folder name is recorded."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _wait_for(log: Path, needle: str, process: subprocess.Popen, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log.exists() and needle in log.read_text(errors="replace"):
            return
        if process.poll() is not None:
            raise SimulationError(f"{what} exited early; see {_shown(log)}")
        time.sleep(0.5)
    raise SimulationError(f"{what} was not ready after {timeout_s:g} s; see {_shown(log)}")


def _stop(process: subprocess.Popen | None) -> None:
    """Stop a process AND its children (gz starts helper processes)."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Simulation:
    """Gazebo (and optionally PX4) for one world folder.

        with Simulation("data/worlds/pilot-meadow") as sim:
            ...                       # fly, record

    Logs go to <world>/logs/. PX4's console gets a stdin that never ends:
    with a closed stdin it prints its prompt forever (hundreds of MB of log).
    """

    def __init__(self, world_dir: Path | str, *, with_px4: bool = True):
        self.world_dir = Path(world_dir).resolve()
        self.spec = load_spec(self.world_dir)
        self.with_px4 = with_px4
        self.log_dir = self.world_dir / "logs"
        self._gazebo: subprocess.Popen | None = None
        self._px4: subprocess.Popen | None = None
        self._logs = []

    def start(self) -> "Simulation":
        self.log_dir.mkdir(exist_ok=True)
        # A simulator left over from a killed run would be silently reused
        # by PX4, with the WRONG world. Refuse instead of recording that.
        if self._clock_topic() in self._topics():
            raise SimulationError(
                "a Gazebo simulator is already running. Stop it first: pkill -f 'gz sim'"
            )
        try:
            gz_log = self.log_dir / "gazebo.log"
            out = open(gz_log, "w")
            self._logs.append(out)
            self._gazebo = subprocess.Popen(
                gazebo_command(self.world_dir), stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )  # fmt: skip
            self._wait_for_world(gz_log)

            if self.with_px4:
                px4_log = self.log_dir / "px4.log"
                out = open(px4_log, "w")
                self._logs.append(out)
                build = _build_dir()
                self._px4 = subprocess.Popen(
                    [str(build / "bin" / "px4")], cwd=build, env=px4_environment(self.spec),
                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                    start_new_session=True,
                )  # fmt: skip
                _wait_for(px4_log, "Startup script returned successfully", self._px4, 60, "PX4")
        except BaseException:
            self.stop()
            raise
        return self

    def _clock_topic(self) -> str:
        return f"/world/{self.spec['gz_world']}/clock"

    def _topics(self) -> list[str]:
        script = f'source "{_build_dir() / "gz_env.sh"}"; export GZ_IP=127.0.0.1; gz topic -l'
        listing = subprocess.run(
            ["/bin/bash", "-c", script], capture_output=True, text=True, timeout=30
        )
        return listing.stdout.split()

    def _wait_for_world(self, gz_log: Path, timeout_s: float = 60.0) -> None:
        """Gazebo is ready when the world's clock topic exists."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._gazebo.poll() is not None:
                raise SimulationError(f"Gazebo exited early; see {_shown(gz_log)}")
            if self._clock_topic() in self._topics():
                return
            time.sleep(1.0)
        raise SimulationError(f"Gazebo world was not ready after {timeout_s:g} s; see {_shown(gz_log)}")

    def stop(self) -> None:
        _stop(self._px4)
        _stop(self._gazebo)
        self._px4 = self._gazebo = None
        for log in self._logs:
            log.close()
        self._logs = []

    def __enter__(self) -> "Simulation":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gazebo + PX4 on a generated world.")
    parser.add_argument("--world-dir", type=Path, required=True)
    parser.add_argument("--no-px4", action="store_true", help="Gazebo only")
    args = parser.parse_args(argv)
    try:
        with Simulation(args.world_dir, with_px4=not args.no_px4) as sim:
            print(f"simulation running: world {sim.spec['id']!r}, logs in {sim.log_dir}")
            print("press Ctrl-C to stop")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopped")
        return 0
    except SimulationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
