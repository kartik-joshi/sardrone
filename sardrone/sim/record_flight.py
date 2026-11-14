"""Fly a square and record it as one episode.

    make record                                   # real PX4 SITL on udp://:14540
    python -m sardrone.sim.record_flight --fake   # no simulator, instant

Mission: arm, take off to 10 m, fly a 20 m square, land. While flying, the
recorder samples the drone at 10 Hz and writes pose, GPS, telemetry and a
camera frame for every sample, plus events when something happens.
"""

import argparse
import asyncio
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sardrone.mcap.episode_writer import (
    EVENT_LAND,
    EVENT_PERSON_SEEN,
    EVENT_TAKEOFF,
    EVENT_WAYPOINT_REACHED,
    EpisodeMeta,
    EpisodeWriter,
)
from sardrone.mcap.timeutil import NS_PER_SEC, EpisodeClock, px4_usec_to_ns
from sardrone.sim.camera import CameraSource, SyntheticCamera, pinhole_intrinsics
from sardrone.sim.drone import Drone, DroneState, FakeDrone

DEFAULT_EPISODE_DIR = Path("data/episodes")


@dataclass
class FlightPlan:
    altitude_m: float = 10.0
    side_m: float = 20.0
    rate_hz: float = 10.0
    reach_tolerance_m: float = 1.0
    phase_timeout_s: float = 90.0
    # A pretend person on the ground, as (north, east) from the takeoff point.
    # The synthetic camera draws them when the drone is close. This stands in
    # for a real detector until the Gazebo camera is wired up.
    person_north_east_m: tuple[float, float] | None = (20.0, 20.0)
    person_sighting_radius_m: float = 3.0

    def square_corners(self) -> list[tuple[float, float, float]]:
        """(north, east, yaw_deg) offsets from the takeoff point, nose along each leg."""
        s = self.side_m
        return [(s, 0.0, 0.0), (s, s, 90.0), (0.0, s, 180.0), (0.0, 0.0, 270.0)]


class MissionTimeout(RuntimeError):
    pass


def next_episode_path(directory: Path = DEFAULT_EPISODE_DIR, prefix: str = "flight") -> Path:
    """First unused data/episodes/flight-NNNN.mcap (flight-0001 on a fresh folder)."""
    number = 1
    while (directory / f"{prefix}-{number:04d}.mcap").exists():
        number += 1
    return directory / f"{prefix}-{number:04d}.mcap"


class FlightRecorder:
    """Runs the mission and logs it. One instance = one episode."""

    def __init__(
        self,
        drone: Drone,
        episode: EpisodeWriter,
        camera: CameraSource,
        *,
        plan: FlightPlan | None = None,
        sleep=asyncio.sleep,
        start_wall_ns: int | None = None,
    ):
        self.drone = drone
        self.episode = episode
        self.camera = camera
        self.plan = plan or FlightPlan()
        self._sleep = sleep
        self._start_wall_ns = time.time_ns() if start_wall_ns is None else start_wall_ns

        self.clock: EpisodeClock | None = None
        self.clock_source = ""  # "px4" or "host", decided once at the first sample
        self.state: DroneState | None = None
        self.now_ns = self._start_wall_ns
        self._person_seen = False
        self._origin: tuple[float, float, float] | None = None  # NED position at takeoff

    # ---- time ---------------------------------------------------------------

    def _source_ns(self, state: DroneState) -> int:
        if self.clock_source == "px4":
            return px4_usec_to_ns(state.px4_time_usec)
        return time.monotonic_ns()

    def _wall_ns(self, state: DroneState) -> int:
        """Episode time for this sample. The clock is anchored on the first call."""
        if self.clock is None:
            # ONE clock per episode: prefer PX4's own timestamps; fall back to
            # this computer's monotonic clock if PX4 does not provide them.
            self.clock_source = "px4" if state.px4_time_usec else "host"
            self.clock = EpisodeClock(self._start_wall_ns)
            self.clock.anchor(self.clock_source, self._source_ns(state))
            return self._start_wall_ns
        return self.clock.to_wall_ns(self.clock_source, self._source_ns(state))

    # ---- sampling -----------------------------------------------------------

    async def _sample(self) -> None:
        """Read the drone once and write one message on every stream."""
        state = await self.drone.read_state()
        wall_ns = self._wall_ns(state)
        episode = self.episode

        episode.log_pose_ned(wall_ns, state.north_m, state.east_m, state.down_m, state.q_ned_frd)
        episode.log_gps(
            wall_ns, state.latitude_deg, state.longitude_deg, state.absolute_altitude_m
        )
        episode.log_telemetry(
            wall_ns,
            battery_percent=state.battery_percent,
            battery_voltage_v=state.battery_voltage_v,
            flight_mode=state.flight_mode,
            velocity_ned_m_s=(state.v_north_m_s, state.v_east_m_s, state.v_down_m_s),
        )

        self.state, self.now_ns = state, wall_ns
        person_in_view = self._person_in_view(state)
        self._update_synthetic_camera(state, person_in_view)
        episode.log_image(wall_ns, self.camera.read())

        if person_in_view and not self._person_seen:
            self._person_seen = True
            episode.meta.people_found += 1
            self._event(EVENT_PERSON_SEEN, f"at {state.height_m:.1f} m height")

    def _person_in_view(self, state: DroneState) -> bool:
        person = self.plan.person_north_east_m
        if person is None or self._origin is None or not state.in_air:
            return False
        north = self._origin[0] + person[0]
        east = self._origin[1] + person[1]
        distance = math.hypot(state.north_m - north, state.east_m - east)
        return distance <= self.plan.person_sighting_radius_m

    def _update_synthetic_camera(self, state: DroneState, person_in_view: bool) -> None:
        if isinstance(self.camera, SyntheticCamera):
            self.camera.north_m, self.camera.east_m = state.north_m, state.east_m
            self.camera.person_visible = person_in_view
            self.camera.caption = f"{state.flight_mode}  h={state.height_m:4.1f} m"

    def _event(self, event: str, detail: str = "") -> None:
        self.episode.log_event(self.now_ns, event, detail)
        print(f"[{(self.now_ns - self._start_wall_ns) / NS_PER_SEC:6.1f} s] {event} {detail}")

    async def _record_until(self, done, what: str) -> None:
        """Keep sampling until done(state) is true, or raise MissionTimeout."""
        deadline_ns = self.now_ns + int(self.plan.phase_timeout_s * NS_PER_SEC)
        while True:
            await self._sleep(1.0 / self.plan.rate_hz)
            await self._sample()
            if done(self.state):
                return
            if self.now_ns > deadline_ns:
                raise MissionTimeout(f"timed out after {self.plan.phase_timeout_s:g} s: {what}")

    # ---- the mission --------------------------------------------------------

    async def run(self) -> None:
        plan, drone = self.plan, self.drone

        fx, fy, cx, cy = pinhole_intrinsics(
            self.camera.width, self.camera.height, self.camera.hfov_deg
        )
        await self._sample()  # first sample anchors the clock at episode start
        self.episode.log_calibration(
            self.now_ns, self.camera.width, self.camera.height, fx, fy, cx, cy
        )
        self._origin = (self.state.north_m, self.state.east_m, self.state.down_m)
        north0, east0, down0 = self._origin
        cruise_down = down0 - plan.altitude_m

        await drone.arm()
        await drone.takeoff(plan.altitude_m)
        self._event(EVENT_TAKEOFF, f"climbing to {plan.altitude_m:g} m")
        await self._record_until(
            lambda s: s.height_m - (-down0) >= plan.altitude_m - 0.5, "reach takeoff altitude"
        )

        corners = plan.square_corners()
        for index, (d_north, d_east, yaw_deg) in enumerate(corners, start=1):
            target_north, target_east = north0 + d_north, east0 + d_east
            await drone.goto_ned(target_north, target_east, cruise_down, yaw_deg)
            await self._record_until(
                lambda s: math.dist(
                    (s.north_m, s.east_m, s.down_m), (target_north, target_east, cruise_down)
                )
                <= plan.reach_tolerance_m,
                f"reach corner {index}",
            )
            self._event(EVENT_WAYPOINT_REACHED, f"{index}/{len(corners)}")

        await drone.land()
        self._event(EVENT_LAND, "descending")
        await self._record_until(lambda s: not s.in_air, "touch down")
        self._event(EVENT_LAND, "touchdown")

        meta = self.episode.meta
        meta.success = meta.people_found == meta.people_placed


async def record_flight(
    drone: Drone,
    path: Path,
    meta: EpisodeMeta,
    *,
    camera: CameraSource | None = None,
    plan: FlightPlan | None = None,
    sleep=asyncio.sleep,
    start_wall_ns: int | None = None,
    overwrite: bool = False,
) -> EpisodeWriter:
    """Connect, fly, record. The episode file is always closed properly, and
    if the mission fails part-way we still try to land."""
    plan = plan or FlightPlan()
    meta.people_placed = 0 if plan.person_north_east_m is None else 1
    await drone.connect()
    try:
        await drone.wait_until_ready()
        with EpisodeWriter(path, meta, overwrite=overwrite) as episode:
            recorder = FlightRecorder(
                drone,
                episode,
                camera or SyntheticCamera(),
                plan=plan,
                sleep=sleep,
                start_wall_ns=start_wall_ns,
            )
            try:
                await recorder.run()
            except BaseException:  # includes Ctrl-C: never leave the drone flying
                episode.meta.success = False
                print("mission failed - commanding land", file=sys.stderr)
                try:
                    await drone.land()
                except Exception as land_error:  # noqa: BLE001
                    print(f"land command failed too: {land_error}", file=sys.stderr)
                raise
        return episode
    finally:
        await drone.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fly a square and record one .mcap episode.")
    parser.add_argument("--url", default="udpin://0.0.0.0:14540", help="MAVSDK connection URL")
    parser.add_argument("--out", type=Path, help="output .mcap (default: next flight-NNNN)")
    parser.add_argument("--fake", action="store_true", help="use FakeDrone, no simulator")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    parser.add_argument("--world", default="default")
    parser.add_argument("--weather", default="clear")
    parser.add_argument("--time-of-day", default="noon")
    parser.add_argument(
        "--prompt", default="Fly a 20 m square at 10 m and report any person you see."
    )
    args = parser.parse_args(argv)

    path = args.out or next_episode_path(prefix="fake" if args.fake else "flight")
    if path.exists() and not args.overwrite:
        print(f"error: {path} already exists (pass --overwrite to replace it)", file=sys.stderr)
        return 1
    meta = EpisodeMeta(
        world=args.world,
        weather=args.weather,
        time_of_day=args.time_of_day,
        mission_prompt=args.prompt,
    )

    if args.fake:
        drone = FakeDrone()
        sleep = drone.sleep  # simulated time: the mission finishes instantly
    else:
        from sardrone.sim.mavsdk_drone import MavsdkDrone  # needs the mavsdk package

        drone = MavsdkDrone(args.url)
        sleep = asyncio.sleep

    try:
        episode = asyncio.run(
            record_flight(drone, path, meta, sleep=sleep, overwrite=args.overwrite)
        )
    except KeyboardInterrupt:
        print("interrupted - land was commanded; the partial episode was saved", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - top-level CLI error report
        print(f"error: {error}", file=sys.stderr)
        return 1

    sidecar = episode.sidecar()
    print(f"wrote {episode.path}  ({sidecar['duration_s']:.1f} s, success={sidecar['success']})")
    print(f"wrote {episode.sidecar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
