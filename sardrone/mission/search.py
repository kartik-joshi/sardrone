"""The search mission: a state machine on top of the flight recorder.

    make search ROW=pilot-meadow          # real PX4 (run `make sim-world` first)
    make search-fake ROW=pilot-meadow     # no simulator, finishes instantly

A STATE MACHINE means the drone is always in exactly one named state, and
only certain events move it to another one. That makes behaviour easy to
reason about ("what is it doing, and why?") and easy to test: the list of
state changes IS the story of the mission.

    TAKEOFF -> SEARCH -> CONFIRM -> REPORT -> RESUME -> SEARCH ... -> RTL -> LAND -> DONE
                  |          \\-> RESUME (false alarm)
                  \\-> RTL (battery low, or the whole area is covered)

    SEARCH   fly the lawnmower lanes, asking the detector about every frame
    CONFIRM  a sighting: descend and circle the spot, camera on the target
    REPORT   seen often enough while circling: record the person's position
    RESUME   climb back and return to where the search was interrupted
    RTL      "return to launch": fly home

Every state change is written to /events, so it shows on the episode's
timeline next to the camera and the 3D view.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sardrone.mcap.episode_writer import (
    EVENT_BATTERY_LOW,
    EVENT_FALSE_ALARM,
    EVENT_LAND,
    EVENT_PERSON_CONFIRMED,
    EVENT_PERSON_REPORTED,
    EVENT_PERSON_SEEN,
    EVENT_SEARCH_COMPLETE,
    EVENT_STATE,
    EVENT_TAKEOFF,
    EVENT_WAYPOINT_REACHED,
    EpisodeMeta,
    EpisodeWriter,
)
from sardrone.mcap.geo import enu_offset_to_latlon
from sardrone.mcap.timeutil import NS_PER_SEC
from sardrone.mission.coverage import CoveragePlan, Point, lawnmower
from sardrone.mission.footprint import footprint
from sardrone.mission.oracle import Detector, OracleDetector, Sighting
from sardrone.mission.projection import CameraModel, DronePose
from sardrone.sim.camera import CameraSource, SyntheticCamera, pinhole_intrinsics
from sardrone.sim.drone import Drone, FakeDrone
from sardrone.worlds.sdf import CAMERA_HEIGHT, CAMERA_HFOV_RAD, CAMERA_WIDTH
from sardrone.sim.record_flight import (
    FlightPlan,
    FlightRecorder,
    MissionTimeout,
    next_episode_path,
    world_settings,
)

TAKEOFF, SEARCH, CONFIRM, REPORT, RESUME, RTL, LAND, DONE = (
    "TAKEOFF", "SEARCH", "CONFIRM", "REPORT", "RESUME", "RTL", "LAND", "DONE",
)  # fmt: skip

# Which state may follow which. Anything else is a bug and raises.
ALLOWED = {
    None: {TAKEOFF},
    TAKEOFF: {SEARCH},
    SEARCH: {CONFIRM, RTL},
    CONFIRM: {REPORT, RESUME},
    REPORT: {RESUME},
    RESUME: {SEARCH, RTL},
    RTL: {LAND},
    LAND: {DONE},
    DONE: set(),
}


@dataclass
class SearchConfig:
    altitude_m: float = 15.0
    overlap: float = 0.25  # the plan asks for 20-30 %
    lane_heading_deg: float | None = None  # None = along the area's longest side
    min_person_px: float = 12.0
    search_speed_m_s: float = 3.0
    confirm_speed_m_s: float = 2.0
    confirm_altitude_m: float = 10.0
    orbit_arc_deg: float = 120.0  # the "short orbit"
    confirm_min_frames: int = 10  # frames with the target in view, to accept it
    retry_after_s: float = 30.0  # ignore a rejected sighting for this long
    battery_rtl_percent: float = 30.0
    rate_hz: float = 10.0
    reach_tolerance_m: float = 1.0
    arrive_speed_m_s: float = 0.5
    phase_timeout_s: float = 180.0


def bearing_deg(from_ne: Point, to_ne: Point) -> float:
    """Compass direction from one (north, east) point to another: 0 = north, 90 = east."""
    return math.degrees(math.atan2(to_ne[1] - from_ne[1], to_ne[0] - from_ne[0])) % 360.0


class SearchRecorder(FlightRecorder):
    """Flies the search mission and records it. One instance = one episode."""

    def __init__(
        self,
        drone: Drone,
        episode: EpisodeWriter,
        camera: CameraSource,
        *,
        polygon: list[Point],
        detector: Detector,
        config: SearchConfig | None = None,
        truth: list[dict] | None = None,
        ground_z: float = 0.0,
        sleep=asyncio.sleep,
        start_wall_ns: int | None = None,
    ):
        config = config or SearchConfig()
        super().__init__(
            drone,
            episode,
            camera,
            plan=FlightPlan(
                altitude_m=config.altitude_m,
                rate_hz=config.rate_hz,
                person_north_east_m=None,
                phase_timeout_s=config.phase_timeout_s,
            ),
            sleep=sleep,
            start_wall_ns=start_wall_ns,
            ground_truth_people=truth,
        )
        self.config = config
        self.polygon = polygon
        self.detector = detector
        self.truth = {person["id"]: person for person in (truth or [])}
        self.ground_z = ground_z
        self.model = CameraModel.from_source(camera)

        self.mission_state: str | None = None
        self.transitions: list[tuple[str, str]] = []  # (state, reason), for tests and the sidecar
        self.coverage: CoveragePlan | None = None
        self.reports: list[dict] = []
        self.orbits: list[dict] = []  # centre and radius of each confirmation circle
        self.confirmed: set[str] = set()
        self._rejected_until: dict[str, int] = {}
        self._home_latlon: tuple[float, float] = (0.0, 0.0)
        self._down0 = 0.0
        self._yaw_deg = 0.0

    # ---- state machine ------------------------------------------------------

    def _enter(self, state: str, reason: str = "") -> None:
        if state not in ALLOWED[self.mission_state]:
            raise RuntimeError(f"illegal transition {self.mission_state} -> {state}")
        previous, self.mission_state = self.mission_state, state
        self.transitions.append((state, reason))
        text = f"{previous or 'START'} -> {state}" + (f" ({reason})" if reason else "")
        self._event(EVENT_STATE, text)

    # ---- flying -------------------------------------------------------------

    def _position(self) -> tuple[float, float, float]:
        return self.state.north_m, self.state.east_m, self.state.down_m

    def _down_for(self, altitude_m: float) -> float:
        return self._down0 - altitude_m

    async def _fly_line(self, target, speed_m_s: float, what: str, *, yaw_deg=None, watch=None):
        """Fly straight to `target` (north, east, down) at `speed_m_s`.

        Like `_fly_leg`, the setpoint moves along the line so the drone
        follows at our speed. New here: after every sample `watch()` is
        called, and if it returns something, the flight stops there and that
        value is returned. That is how a sighting interrupts a lane.
        Returns None when the target was reached.
        """
        config = self.config
        period = 1.0 / config.rate_hz
        start = self._position()
        length = math.dist(start, target)
        if yaw_deg is None:
            horizontal = math.dist(start[:2], target[:2])
            yaw_deg = bearing_deg(start[:2], target[:2]) if horizontal > 0.5 else self._yaw_deg
        self._yaw_deg = yaw_deg
        travelled = 0.0
        deadline_ns = self.now_ns + int(config.phase_timeout_s * NS_PER_SEC)
        while True:
            travelled = min(length, travelled + speed_m_s * period)
            fraction = travelled / length if length > 0 else 1.0
            setpoint = [a + fraction * (b - a) for a, b in zip(start, target)]
            await self.drone.goto_ned(*setpoint, yaw_deg)
            await self._sleep(period)
            await self._sample()

            if watch is not None:
                interruption = watch()
                if interruption is not None:
                    return interruption
            speed = math.hypot(self.state.v_north_m_s, self.state.v_east_m_s)
            if (
                travelled >= length
                and math.dist(self._position(), target) <= config.reach_tolerance_m
                and speed <= config.arrive_speed_m_s
            ):
                return None
            if self.now_ns > deadline_ns:
                raise MissionTimeout(f"timed out after {config.phase_timeout_s:g} s: {what}")

    # ---- watching -----------------------------------------------------------

    def _battery_low(self) -> bool:
        return self.state.battery_percent <= self.config.battery_rtl_percent

    def _new_sightings(self) -> list[Sighting]:
        """Sightings of people not yet confirmed (and not recently rejected)."""
        sightings = self.detector.detect(DronePose.from_state(self.state))
        fresh = []
        for sighting in sightings:
            key = sighting.person_id or "unknown"
            if key in self.confirmed or self.now_ns < self._rejected_until.get(key, 0):
                continue
            if sighting.ground_east_north is not None:
                fresh.append(sighting)
        return fresh

    def _watch_search(self):
        if self._battery_low():
            return ("battery", None)
        sightings = self._new_sightings()
        if sightings:
            # Take the one that is largest in the image: the clearest view.
            return ("sighting", max(sightings, key=lambda s: s.bbox.height * s.bbox.width))
        return None

    def _watch_battery(self):
        return ("battery", None) if self._battery_low() else None

    # ---- CONFIRM and REPORT -------------------------------------------------

    async def _confirm(self, sighting: Sighting) -> tuple[bool, tuple[float, float], int]:
        """Descend and circle the sighting with the camera pointed at it.

        The camera is fixed to the drone, tilted down by `pitch`. Its centre
        looks at the ground `height / tan(pitch)` ahead. So circling at
        exactly that radius, nose towards the centre, keeps the target in
        the middle of the image.

        Returns (confirmed, estimated (east, north), frames with target in view).
        """
        config = self.config
        east, north = sighting.ground_east_north
        key = sighting.person_id or "unknown"
        down = self._down_for(config.confirm_altitude_m)
        height_above_ground = config.confirm_altitude_m - self.ground_z
        radius = height_above_ground / math.tan(math.radians(max(self.model.pitch_deg, 20.0)))

        self.orbits.append({"east_m": east, "north_m": north, "radius_m": radius, "down_m": down})

        # Enter the circle at the point nearest to where we are now.
        here = self._position()
        angle = math.atan2(here[1] - east, here[0] - north)  # from target to drone
        entry = (north + radius * math.cos(angle), east + radius * math.sin(angle), down)
        await self._fly_line(
            entry, config.confirm_speed_m_s, "reach the confirmation orbit",
            yaw_deg=bearing_deg(here[:2], (north, east)),
        )  # fmt: skip

        period = 1.0 / config.rate_hz
        step = config.confirm_speed_m_s * period / radius  # radians per tick
        ticks = math.ceil(math.radians(config.orbit_arc_deg) / step)
        frames, estimates = 0, []
        for _ in range(ticks):
            angle += step
            point = (north + radius * math.cos(angle), east + radius * math.sin(angle))
            self._yaw_deg = bearing_deg(point, (north, east))
            await self.drone.goto_ned(point[0], point[1], down, self._yaw_deg)
            await self._sleep(period)
            await self._sample()
            for seen in self.detector.detect(DronePose.from_state(self.state)):
                if (seen.person_id or "unknown") == key and seen.ground_east_north is not None:
                    frames += 1
                    estimates.append(seen.ground_east_north)
                    break

        if not estimates:
            return False, (east, north), 0
        # Each estimate is a little short, TOWARDS the camera (the bottom of
        # the box is the near side of the body). Seen from many sides,
        # those errors partly cancel, so the average beats any single view.
        mean = (sum(e[0] for e in estimates) / len(estimates), sum(e[1] for e in estimates) / len(estimates))
        return frames >= config.confirm_min_frames, mean, frames

    def _report(self, sighting: Sighting, estimate: tuple[float, float], frames: int) -> None:
        east, north = estimate
        latitude, longitude = enu_offset_to_latlon(*self._home_latlon, east, north)
        report = {
            "person_id": sighting.person_id,
            "east_m": round(east, 2),
            "north_m": round(north, 2),
            "latitude_deg": round(latitude, 7),
            "longitude_deg": round(longitude, 7),
            "frames_in_view": frames,
            "time_s": round((self.now_ns - self._start_wall_ns) / NS_PER_SEC, 1),
        }
        truth = self.truth.get(sighting.person_id or "")
        if truth is not None:
            true_east, true_north = truth["box_centre"][0], truth["box_centre"][1]
            report["error_m"] = round(math.hypot(east - true_east, north - true_north), 2)
        self.reports.append(report)
        self.confirmed.add(sighting.person_id or "unknown")
        self.episode.meta.people_found += 1
        error = f", error {report['error_m']:.2f} m" if "error_m" in report else ""
        self._event(
            EVENT_PERSON_REPORTED,
            f"{sighting.person_id or 'person'} at {latitude:.6f}, {longitude:.6f}{error}",
        )

    async def _handle_sighting(self, sighting: Sighting) -> None:
        """CONFIRM, then REPORT or reject, then RESUME back to the lane."""
        name = sighting.person_id or "person"
        resume_at = self._position()
        resume_yaw = self._yaw_deg
        size = max(sighting.bbox.width, sighting.bbox.height)
        self._event(EVENT_PERSON_SEEN, f"{name}, {size:.0f} px in the image")
        self._enter(CONFIRM, f"{name} sighted")
        confirmed, estimate, frames = await self._confirm(sighting)
        if confirmed:
            self._event(EVENT_PERSON_CONFIRMED, f"{name} in view for {frames} frames")
            self._enter(REPORT, name)
            self._report(sighting, estimate, frames)
            self._enter(RESUME, "reported")
        else:
            until = self.now_ns + int(self.config.retry_after_s * NS_PER_SEC)
            self._rejected_until[sighting.person_id or "unknown"] = until
            self._event(EVENT_FALSE_ALARM, f"{name} in view for only {frames} frames")
            self._enter(RESUME, "not confirmed")
        await self._fly_line(
            (resume_at[0], resume_at[1], self._down_for(self.config.altitude_m)),
            self.config.search_speed_m_s,
            "return to the search lane",
        )
        self._yaw_deg = resume_yaw

    # ---- the mission --------------------------------------------------------

    async def run(self) -> None:
        config, drone = self.config, self.drone
        fx, fy, cx, cy = pinhole_intrinsics(self.camera.width, self.camera.height, self.camera.hfov_deg)
        await self._sample()  # anchors the clock
        self.episode.log_calibration(self.now_ns, self.camera.width, self.camera.height, fx, fy, cx, cy)
        self._origin = self._position()
        self._down0 = self.state.down_m
        self._home_latlon = (self.state.latitude_deg, self.state.longitude_deg)
        home = (self.state.north_m, self.state.east_m)

        view = footprint(self.model, config.altitude_m - self.ground_z, config.min_person_px)
        self.coverage = lawnmower(
            self.polygon, view, overlap=config.overlap, heading_deg=config.lane_heading_deg, start=home
        )
        cruise_down = self._down_for(config.altitude_m)

        self._enter(TAKEOFF)
        await drone.arm()
        await drone.takeoff(config.altitude_m)
        self._event(EVENT_TAKEOFF, f"climbing to {config.altitude_m:g} m")
        await self._record_until(
            lambda s: s.height_m - (-self._down0) >= config.altitude_m - 0.5, "reach takeoff altitude"
        )

        self._enter(SEARCH, f"{self.coverage.lanes} lanes, {self.coverage.spacing_m:.0f} m apart")
        reason = "search complete"
        index = 0
        waypoints = self.coverage.waypoints
        while index < len(waypoints):
            waypoint = waypoints[index]
            # On a lane the nose points along the lane, so the camera sweeps
            # the strip it was planned for, also right after an interruption.
            lane_yaw = None
            if waypoint.kind == "lane_end":
                begin = waypoints[index - 1]
                lane_yaw = bearing_deg((begin.north_m, begin.east_m), (waypoint.north_m, waypoint.east_m))
            interruption = await self._fly_line(
                (waypoint.north_m, waypoint.east_m, cruise_down),
                config.search_speed_m_s,
                f"reach {waypoint.kind} of lane {waypoint.lane}",
                yaw_deg=lane_yaw,
                watch=self._watch_search,
            )
            if interruption is None:
                self._event(EVENT_WAYPOINT_REACHED, f"lane {waypoint.lane} {waypoint.kind}")
                index += 1
                continue
            kind, sighting = interruption
            if kind == "battery":
                reason = "battery low"
                break
            await self._handle_sighting(sighting)
            if self._battery_low():
                reason = "battery low"
                break
            self._enter(SEARCH, "back on the lane")

        if reason == "battery low":
            self._event(EVENT_BATTERY_LOW, f"{self.state.battery_percent:.0f} %")
        else:
            self._event(EVENT_SEARCH_COMPLETE, f"{self.coverage.lanes} lanes flown")
        self._enter(RTL, reason)
        await self._fly_line((home[0], home[1], cruise_down), config.search_speed_m_s, "fly home")

        self._enter(LAND)
        await drone.land()
        self._event(EVENT_LAND, "descending")
        await self._record_until(lambda s: not s.in_air, "touch down")
        self._event(EVENT_LAND, "touchdown")
        self._enter(DONE)

        meta = self.episode.meta
        meta.success = reason == "search complete" and meta.people_found == meta.people_placed

    def summary(self, rtl_reason: str | None = None) -> dict:
        """What goes into the sidecar about the search itself."""
        coverage = self.coverage
        return {
            "detector": self.detector.name,
            "search": None if coverage is None else {
                "altitude_m": self.config.altitude_m,
                "lanes": coverage.lanes,
                "lane_spacing_m": round(coverage.spacing_m, 2),
                "swath_half_width_m": round(coverage.half_width_m, 2),
                "overlap": round(coverage.overlap, 3),
                "lane_heading_deg": round(coverage.heading_deg, 1),
                "planned_path_m": round(coverage.path_length_m(), 1),
            },
            "transitions": [state for state, _ in self.transitions],
            "reports": self.reports,
        }  # fmt: skip


async def run_search(
    drone: Drone,
    path: Path,
    meta: EpisodeMeta,
    *,
    camera: CameraSource,
    polygon: list[Point],
    truth: list[dict],
    config: SearchConfig | None = None,
    detector: Detector | None = None,
    ground_z: float = 0.0,
    sleep=asyncio.sleep,
    start_wall_ns: int | None = None,
    overwrite: bool = False,
    extra_sidecar: dict | None = None,
) -> tuple[EpisodeWriter, SearchRecorder]:
    """Connect, search, record. Lands and saves the episode even on failure."""
    config = config or SearchConfig()
    meta.people_placed = len(truth)
    if detector is None:
        detector = OracleDetector(
            truth, CameraModel.from_source(camera), min_person_px=config.min_person_px, ground_z=ground_z
        )
    await drone.connect()
    try:
        await drone.wait_until_ready()
        with EpisodeWriter(
            path,
            meta,
            overwrite=overwrite,
            camera_mount_m=camera.mount_m,
            camera_pitch_deg=camera.pitch_deg,
            extra_sidecar=extra_sidecar,
        ) as episode:
            recorder = SearchRecorder(
                drone, episode, camera,
                polygon=polygon, detector=detector, config=config, truth=truth,
                ground_z=ground_z, sleep=sleep, start_wall_ns=start_wall_ns,
            )  # fmt: skip
            try:
                await recorder.run()
            except BaseException:
                episode.meta.success = False
                print("mission failed - commanding land", file=sys.stderr)
                try:
                    await drone.land()
                except Exception as land_error:  # noqa: BLE001
                    print(f"land command failed too: {land_error}", file=sys.stderr)
                raise
            finally:
                episode.extra_sidecar.update(recorder.summary())
        return episode, recorder
    finally:
        await drone.close()


def square_polygon(side_m: float) -> list[Point]:
    """A square search area centred on the takeoff point, as (north, east) corners."""
    half = side_m / 2
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


def safe_confirm_altitude(spec: dict, wanted_m: float, margin_m: float = 3.0) -> float:
    """Never circle lower than the tallest tree plus a margin."""
    from sardrone.worlds.assets import TREES

    tallest = max((TREES[t["kind"]].hi[2] * t["scale"] for t in spec["trees"]), default=0.0)
    return min(spec["altitude_m"], max(wanted_m, math.ceil(tallest + margin_m)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a generated world and record the episode.")
    parser.add_argument("--world-dir", type=Path, required=True)
    parser.add_argument("--url", default="udpin://0.0.0.0:14540")
    parser.add_argument("--fake", action="store_true", help="FakeDrone and generated images")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--battery-rtl", type=float, default=30.0, help="return home at this percent")
    parser.add_argument(
        "--prompt", default="Search the area, confirm and report every person, then return home."
    )
    args = parser.parse_args(argv)

    spec, truth, extra = world_settings(args.world_dir)
    config = SearchConfig(altitude_m=spec["altitude_m"], battery_rtl_percent=args.battery_rtl)
    config.confirm_altitude_m = safe_confirm_altitude(spec, config.confirm_altitude_m)
    ground_z = -spec["drone_body_height_m"]
    meta = EpisodeMeta(
        world=spec["world"], weather=spec["weather"], time_of_day=spec["time_of_day"],
        mission_prompt=args.prompt,
    )  # fmt: skip
    path = args.out or next_episode_path(prefix=f"{spec['id']}-search" + ("-fake" if args.fake else ""))
    if path.exists() and not args.overwrite:
        print(f"error: {path} already exists (pass --overwrite to replace it)", file=sys.stderr)
        return 1

    camera = None
    if args.fake:
        drone = FakeDrone()
        sleep = drone.sleep
        # Same size and lens as the simulated camera, so the dry run plans the
        # same lanes and applies the same pixel limits as the real flight.
        # (With the default 640x480 stand-in, every 40 m mission was refused.)
        camera = SyntheticCamera(
            width=CAMERA_WIDTH, height=CAMERA_HEIGHT, hfov_deg=math.degrees(CAMERA_HFOV_RAD),
            pitch_deg=spec["camera_pitch_deg"],
        )  # fmt: skip
    else:
        from sardrone.sim.camera import CameraError, GazeboCamera
        from sardrone.sim.mavsdk_drone import MavsdkDrone

        drone, sleep = MavsdkDrone(args.url), asyncio.sleep
        try:
            camera = GazeboCamera(
                model=spec["gz_model"], world=spec["gz_world"],
                mount_m=spec["camera_mount_m"], pitch_deg=spec["camera_pitch_deg"],
                boxes_topic=spec.get("boxes_topic"),
            ).open()  # fmt: skip
        except CameraError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1

    started = time.monotonic()
    try:
        episode, recorder = asyncio.run(
            run_search(
                drone, path, meta, camera=camera, polygon=square_polygon(spec["area_m"]),
                truth=truth, config=config, ground_z=ground_z, sleep=sleep,
                overwrite=args.overwrite, extra_sidecar=extra,
            )  # fmt: skip
        )
    except KeyboardInterrupt:
        print("interrupted - land was commanded; the partial episode was saved", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - top-level CLI error report
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        if not args.fake and camera is not None:
            camera.close()

    sidecar = episode.sidecar()
    print(json.dumps({"reports": recorder.reports, "search": sidecar["search"]}, indent=2))
    print(
        f"wrote {episode.path}  ({sidecar['duration_s']:.1f} s flight, "
        f"{time.monotonic() - started:.1f} s wall, found {sidecar['people_found']}/"
        f"{sidecar['people_placed']}, success={sidecar['success']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
