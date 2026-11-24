"""The mission agent flying the simulated drone (issue #8).

    python -m sardrone.agent.flight --world-dir data/worlds/test-004 \\
        --prompt "Search the north field and report anyone you find." --model rule-based --detector v4
    python -m sardrone.agent.flight --world-dir data/worlds/test-004 --prompt "..." --fake   # no simulator

The agent and its six tools are the same as in the mission simulator
(sardrone/agent/sim.py); here the tools fly PX4 in Gazebo and the whole
flight is recorded as an MCAP episode, with every agent decision on
/events. What changes:

* Named areas are halves of the world square: "north field" is the half
  north of the launch point, "south field", "east meadow", "west woods"
  likewise; "whole area" is the square.
* The search runs in the background while the model thinks. get_status
  waits until the search has moved on by about 20 % of the area, a new
  sighting arrives, the area is finished or the battery is low, like one
  step of the mission simulator.
* PX4 needs a continuous stream of position commands in offboard mode, so
  a pilot task always streams them: along the lanes while searching,
  holding position otherwise.
* confirm_target flies the confirmation orbit. When the orbit confirms the
  sighting, the description returned is that of the nearest person in the
  world file (clothing colour and pose); the detector itself only finds
  people and recognises neither. That is a stand-in for an attribute model.
"""

import argparse
import asyncio
import json
import math
import sys
import threading
import time
from pathlib import Path

from sardrone.agent.sim import ALTITUDES_M, AREAS
from sardrone.mcap.episode_writer import EVENT_AGENT, EVENT_LAND, EVENT_SEARCH_COMPLETE, EVENT_STATE, EVENT_TAKEOFF
from sardrone.mission.coverage import lawnmower
from sardrone.mission.footprint import footprint
from sardrone.mission.oracle import Sighting
from sardrone.mission.search import (
    CONFIRM,
    DONE,
    LAND,
    RESUME,
    RTL,
    SEARCH,
    TAKEOFF,
    SearchConfig,
    SearchRecorder,
    bearing_deg,
)
from sardrone.sim.camera import pinhole_intrinsics

STEP_COVERAGE = 20.0  # percent of the area per get_status, as in the mission simulator
STEP_TIMEOUT_S = 90.0  # simulated seconds a get_status may wait
AREA_MARGIN_M = 2.0  # a sighting this far outside the searched area still counts


def area_polygon(area: str, side_m: float) -> list[tuple[float, float]]:
    """(north, east) corners of a named area in a square world centred on the launch point."""
    h = side_m / 2
    halves = {
        "north field": [(0, -h), (h, -h), (h, h), (0, h)],
        "south field": [(-h, -h), (0, -h), (0, h), (-h, h)],
        "east meadow": [(-h, 0), (h, 0), (h, h), (-h, h)],
        "west woods": [(-h, -h), (h, -h), (h, 0), (-h, 0)],
        "whole area": [(-h, -h), (h, -h), (h, h), (-h, h)],
    }
    return halves[area]


def in_area(area: str, side_m: float, east: float, north: float, margin: float = AREA_MARGIN_M) -> bool:
    """Is a ground point inside a named area? The camera looks ahead and
    sideways, beyond the area being searched; people out there are not part
    of this search (in the mission simulator they are never seen)."""
    corners = area_polygon(area, side_m)
    norths, easts = [c[0] for c in corners], [c[1] for c in corners]
    return (min(norths) - margin <= north <= max(norths) + margin
            and min(easts) - margin <= east <= max(easts) + margin)  # fmt: skip


def describe(person: dict | None) -> str:
    if person is None:
        return "a person (no matching record)"
    return f"person in {person['clothing']} clothing, {person['pose']}"


class AgentSearch(SearchRecorder):
    """A SearchRecorder whose decisions are made by the agent's tool calls."""

    def __init__(self, *args, side_m: float, **kwargs):
        super().__init__(*args, **kwargs)
        # The agent's tool calls arrive from another thread through this event
        # loop, so every wait must hand control back to it. A fake drone's
        # sleep only advances a simulated clock and never does.
        inner = self._sleep

        async def yielding_sleep(seconds: float) -> None:
            await inner(seconds)
            await asyncio.sleep(0)

        self._sleep = yielding_sleep
        self.side_m = side_m
        self.pending: dict[str, Sighting] = {}  # target id -> latest sighting, not yet confirmed
        self.confirmed_targets: dict[str, tuple] = {}  # target id -> (sighting, estimate, frames, matched)
        self.reported_targets: list[str] = []
        self.searched: list[tuple[str, int]] = []  # (area, altitude) of each search started
        self.waypoints: list = []
        self.index = 0
        self.searching = False
        self.airborne = False
        self.home_ne = (0.0, 0.0)
        self.hold_at = None
        self._pilot: asyncio.Task | None = None
        self._new_sighting = False

    # ---- set-up and the pilot ------------------------------------------------

    async def prepare(self) -> None:
        fx, fy, cx, cy = pinhole_intrinsics(self.camera.width, self.camera.height, self.camera.hfov_deg)
        await self._sample()
        self.episode.log_calibration(self.now_ns, self.camera.width, self.camera.height, fx, fy, cx, cy)
        self._origin = self._position()
        self._down0 = self.state.down_m
        self._home_latlon = (self.state.latitude_deg, self.state.longitude_deg)
        self.home_ne = (self.state.north_m, self.state.east_m)

    async def _stop_pilot(self) -> None:
        if self._pilot is not None:
            self._pilot.cancel()
            try:
                await self._pilot
            except asyncio.CancelledError:
                pass
            self._pilot = None

    async def _hold(self) -> None:
        """Stream the hold point to PX4 and keep recording until cancelled."""
        period = 1.0 / self.config.rate_hz
        while True:
            north, east, down = self.hold_at
            await self.drone.goto_ned(north, east, down, self._yaw_deg)
            await self._sleep(period)
            await asyncio.sleep(0.01)  # hovering: let real time pass too (a fake clock would race ahead)
            await self._sample()
            self._collect()

    async def _start_hold(self) -> None:
        await self._stop_pilot()
        self.hold_at = self._position()
        self._pilot = asyncio.ensure_future(self._hold())

    def _collect(self):
        """Queue new believed sightings for the agent; never interrupts a lane."""
        area = self.searched[-1][0] if self.searched else None
        for sighting in self._new_sightings():
            if area is None or not in_area(area, self.side_m, *sighting.ground_east_north):
                continue
            key = self._key(sighting)
            self._spots.setdefault(key, sighting.ground_east_north)
            if key not in self.pending and key not in self.confirmed_targets:
                self._new_sighting = True
            self.pending[key] = sighting
        return ("battery", None) if self._battery_low() else None

    async def _lanes(self) -> None:
        down = self._down_for(self._search_altitude_m)
        while self.index < len(self.waypoints):
            waypoint = self.waypoints[self.index]
            lane_yaw = None
            if waypoint.kind == "lane_end" and self.index > 0:
                begin = self.waypoints[self.index - 1]
                lane_yaw = bearing_deg((begin.north_m, begin.east_m), (waypoint.north_m, waypoint.east_m))
            stop = await self._fly_line((waypoint.north_m, waypoint.east_m, down), self.config.search_speed_m_s,
                                        f"reach {waypoint.kind} of lane {waypoint.lane}", yaw_deg=lane_yaw,
                                        watch=self._collect)  # fmt: skip
            if stop is not None:  # battery low: stop the pattern and hold
                break
            self.index += 1
        self.searching = False
        if self.index >= len(self.waypoints):
            self._event(EVENT_SEARCH_COMPLETE, f"{self.searched[-1][0]}: {self.coverage.lanes} lanes flown")
        self.hold_at = self._position()
        await self._hold()

    # ---- what the tools do ---------------------------------------------------

    def coverage_percent(self) -> float:
        return 100.0 * self.index / len(self.waypoints) if self.waypoints else 0.0

    async def start(self, area: str, altitude_m: int) -> None:
        self._search_altitude_m = float(altitude_m)
        if not self.airborne:
            self._enter(TAKEOFF)
            await self.drone.arm()
            await self.drone.takeoff(altitude_m)
            self._event(EVENT_TAKEOFF, f"climbing to {altitude_m:g} m")
            await self._record_until(lambda s: s.height_m - (-self._down0) >= altitude_m - 0.5,
                                     "reach takeoff altitude")  # fmt: skip
            self.airborne = True
        await self._stop_pilot()
        view = footprint(self.model, altitude_m - self.ground_z, self.config.min_person_px)
        here = (self.state.north_m, self.state.east_m)
        self.coverage = lawnmower(area_polygon(area, self.side_m), view, overlap=self.config.overlap,
                                  heading_deg=self.config.lane_heading_deg, start=here)  # fmt: skip
        self.waypoints, self.index, self.searching = self.coverage.waypoints, 0, True
        self.searched.append((area, altitude_m))
        if self.mission_state != SEARCH:
            self._enter(SEARCH, f"{area}: {self.coverage.lanes} lanes at {altitude_m} m")
        else:
            self._event(EVENT_STATE, f"SEARCH {area}: {self.coverage.lanes} lanes at {altitude_m} m")
        self._pilot = asyncio.ensure_future(self._lanes())

    async def wait_step(self, start_coverage: float) -> None:
        """Let the search run until it has moved on, or something needs a decision."""
        self._new_sighting = False
        deadline = self.now_ns + int(STEP_TIMEOUT_S * 1e9)
        while self.searching and not self._battery_low() and not self._new_sighting:
            if self.coverage_percent() >= start_coverage + STEP_COVERAGE or self.now_ns > deadline:
                break
            await asyncio.sleep(0)  # yield to the pilot task
            await self._sleep(0.2)

    async def confirm(self, target: str) -> tuple[bool, str]:
        sighting = self.pending[target]
        resume_at, resume_yaw, was_searching = self._position(), self._yaw_deg, self.searching
        await self._stop_pilot()
        self._enter(CONFIRM, f"{target} (asked by the agent)")
        ok, estimate, frames = await self._confirm(sighting)
        self.pending.pop(target, None)
        if ok:
            matched, _ = self._match_truth(sighting, *estimate)
            self.confirmed_targets[target] = (sighting, estimate, frames, matched)
            self.confirmed.add(target)
            description = describe(self.truth.get(matched))
        else:
            self._rejected_until[target] = self.now_ns + int(self.config.retry_after_s * 1e9)
            self.false_alarms += 1
            description = None
        self._enter(RESUME, "confirmed" if ok else "not confirmed")
        await self._fly_line((resume_at[0], resume_at[1], self._down_for(self._search_altitude_m)),
                             self.config.search_speed_m_s, "return to the search lane")  # fmt: skip
        self._yaw_deg = resume_yaw
        self._enter(SEARCH, "back to the search")
        if was_searching and self.index < len(self.waypoints):
            self.searching = True
            self._pilot = asyncio.ensure_future(self._lanes())
        else:
            await self._start_hold()
        return ok, description

    def report(self, target: str) -> None:
        sighting, estimate, frames, _ = self.confirmed_targets[target]
        self._report(sighting, estimate, frames, target)
        self.reported_targets.append(target)

    async def home(self) -> None:
        await self._stop_pilot()
        self.searching = False
        if self.airborne:
            self._enter(RTL, "asked by the agent")
            await self._fly_line((self.home_ne[0], self.home_ne[1], self._down_for(self._search_altitude_m)),
                                 self.config.search_speed_m_s, "fly home")  # fmt: skip
            self._enter(LAND)
            await self.drone.land()
            self._event(EVENT_LAND, "descending")
            await self._record_until(lambda s: not s.in_air, "touch down")
            self._event(EVENT_LAND, "touchdown")
            self._enter(DONE)
        self.airborne = False


class FlightBackend:
    """The six tools on a real (simulated) flight, with the mission simulator's
    interface: sync calls, dict results, `log`, `errors`, `battery`, `state`."""

    def __init__(self, recorder: AgentSearch, loop: asyncio.AbstractEventLoop, timeout_s: float = 900.0):
        self.recorder, self.loop, self.timeout_s = recorder, loop, timeout_s
        self.log, self.errors = [], 0
        self.area: str | None = None
        self.state = "IDLE"  # IDLE | SEARCHING | SEARCH_COMPLETE | HOME

    def _run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(self.timeout_s)

    def _result(self, tool: str, arguments: dict, ok: bool, **payload) -> dict:
        self.log.append((tool, dict(arguments), ok))
        if not ok:
            self.errors += 1
        return {"ok": ok, **payload}

    @property
    def battery(self) -> float:
        return round(self.recorder.state.battery_percent, 1)

    @property
    def coverage(self) -> float:
        return round(self.recorder.coverage_percent(), 1)

    def _refresh(self) -> None:
        if self.state == "SEARCHING" and not self.recorder.searching:
            self.state = "SEARCH_COMPLETE"

    def set_search_area(self, area: str) -> dict:
        arguments = {"area": area}
        self._refresh()
        if self.state == "HOME":
            return self._result("set_search_area", arguments, False, error="the drone is home; the mission is over")
        if self.state == "SEARCHING":
            return self._result("set_search_area", arguments, False, error="cannot change area while searching")
        if area not in AREAS:
            return self._result("set_search_area", arguments, False, error=f"unknown area; choose one of {list(AREAS)}")
        self.area, self.state = area, "IDLE"
        return self._result("set_search_area", arguments, True, area=area)

    def start_search(self, altitude_m: int = 25) -> dict:
        arguments = {"altitude_m": altitude_m}
        self._refresh()
        if self.state == "HOME":
            return self._result("start_search", arguments, False, error="the drone is home; the mission is over")
        if self.area is None:
            return self._result("start_search", arguments, False, error="set a search area first")
        if altitude_m not in ALTITUDES_M:
            return self._result("start_search", arguments, False, error=f"altitude must be one of {list(ALTITUDES_M)}")
        if self.state == "SEARCHING":
            return self._result("start_search", arguments, False, error="already searching")
        self._run(self.recorder.start(self.area, altitude_m))
        self.state = "SEARCHING"
        return self._result("start_search", arguments, True, state=self.state, battery_percent=self.battery)

    def get_status(self) -> dict:
        if self.state == "SEARCHING":
            self._run(self.recorder.wait_step(self.recorder.coverage_percent()))
        self._refresh()
        r = self.recorder
        return self._result(
            "get_status", {}, True, state=self.state, area=self.area, battery_percent=self.battery,
            coverage_percent=self.coverage, unconfirmed_sightings=sorted(r.pending),
            confirmed_not_reported=[t for t in r.confirmed_targets if t not in r.reported_targets],
            reported=list(r.reported_targets),
        )  # fmt: skip

    def confirm_target(self, target_id: str) -> dict:
        arguments = {"target_id": target_id}
        if self.state == "HOME":
            return self._result("confirm_target", arguments, False, error="the drone is home; the mission is over")
        if target_id in self.recorder.confirmed_targets:
            return self._result("confirm_target", arguments, False, error="already confirmed")
        if target_id not in self.recorder.pending:
            return self._result("confirm_target", arguments, False, error="no such sighting")
        ok, description = self._run(self.recorder.confirm(target_id))
        self._refresh()
        if not ok:
            return self._result("confirm_target", arguments, True, confirmed=False, battery_percent=self.battery,
                                description="nothing confirmed there: a false alarm")  # fmt: skip
        return self._result("confirm_target", arguments, True, confirmed=True, description=description,
                            battery_percent=self.battery)  # fmt: skip

    def report_person(self, target_id: str) -> dict:
        arguments = {"target_id": target_id}
        if target_id not in self.recorder.confirmed_targets:
            return self._result("report_person", arguments, False, error="confirm the target before reporting it")
        if target_id in self.recorder.reported_targets:
            return self._result("report_person", arguments, False, error="already reported")
        self._run(self._async(self.recorder.report, target_id))
        return self._result("report_person", arguments, True, reported=target_id)

    def return_home(self) -> dict:
        if self.state == "HOME":
            return self._result("return_home", {}, False, error="the drone is home; the mission is over")
        self._run(self.recorder.home())
        self.state = "HOME"
        return self._result("return_home", {}, True, state=self.state, battery_percent=self.battery)

    @staticmethod
    async def _async(function, *args):
        return function(*args)

    def reported_people(self) -> list[str]:
        r = self.recorder
        return [r.confirmed_targets[t][3] for t in r.reported_targets if r.confirmed_targets[t][3]]


def fly_mission(model, prompt: str, *, drone, camera, detector, truth, spec, path: Path, meta, sleep,
                max_steps: int = 60, overwrite: bool = False, extra: dict | None = None) -> dict:  # fmt: skip
    """Connect, let the agent fly, record. Returns a summary; always lands."""
    from sardrone.agent.agent import run_agent
    from sardrone.mcap.episode_writer import EpisodeWriter

    config = SearchConfig(altitude_m=spec["altitude_m"])
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    run = lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop).result(900)  # noqa: E731

    run(drone.connect())
    try:
        run(drone.wait_until_ready())
        with EpisodeWriter(path, meta, overwrite=overwrite, camera_mount_m=camera.mount_m,
                           camera_pitch_deg=camera.pitch_deg, extra_sidecar=extra or {}) as episode:  # fmt: skip
            recorder = AgentSearch(drone, episode, camera, polygon=area_polygon("whole area", spec["area_m"]),
                                   detector=detector, config=config, truth=truth, ground_z=-spec["drone_body_height_m"],
                                   sleep=sleep, side_m=spec["area_m"])  # fmt: skip
            run(recorder.prepare())
            backend = FlightBackend(recorder, loop)
            episode.log_event(recorder.now_ns, EVENT_AGENT, f"mission: {prompt}")

            def on_decision(text: str) -> None:
                episode.log_event(recorder.now_ns, EVENT_AGENT, text)

            transcript = None
            try:
                transcript = run_agent(model, prompt, backend, max_steps=max_steps, on_decision=on_decision)
            finally:
                if backend.state != "HOME":  # the agent never came home: bring the drone back anyway
                    episode.log_event(recorder.now_ns, EVENT_AGENT, "safety: the agent did not return home")
                    run(recorder.home())
                else:
                    run(recorder._stop_pilot())
                matched = [recorder.confirmed_targets[t][3] for t in recorder.reported_targets]
                summary = {
                    "prompt": prompt, "model": model.name, "searched": recorder.searched,
                    "reported_people": [m for m in matched if m], "false_reports": sum(1 for m in matched if not m),
                    "tool_calls": [entry[0] for entry in backend.log], "rejected_calls": backend.errors,
                    "ended": transcript.ended if transcript else "error", "battery_at_end": backend.battery,
                    "false_alarms": recorder.false_alarms,
                    "people": {p["id"]: describe(p) for p in truth},
                }  # fmt: skip
                episode.extra_sidecar["agent_flight"] = summary
                episode.extra_sidecar.update(recorder.summary())
        return summary
    finally:
        run(drone.close())
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    from sardrone.agent.evaluate import make_model
    from sardrone.mcap.episode_writer import EpisodeMeta
    from sardrone.mission.projection import CameraModel
    from sardrone.mission.search import PROJECT_ROOT, scorecard_confidence
    from sardrone.sim.record_flight import next_episode_path, world_settings

    parser = argparse.ArgumentParser(description="Let the mission agent fly the simulated drone.")
    parser.add_argument("--world-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="rule-based", choices=["rule-based", "cohere"])
    parser.add_argument("--cohere-model")
    parser.add_argument("--detector", default="oracle", help="'oracle' or a model version, e.g. v4")
    parser.add_argument("--url", default="udpin://0.0.0.0:14540")
    parser.add_argument("--fake", action="store_true", help="FakeDrone and generated images, no simulator")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--launch", action="store_true", help="start PX4 and Gazebo for this world first")
    args = parser.parse_args(argv)
    if args.launch and not args.fake:
        import os

        from sardrone.sim.launch import Simulation, gz_python

        os.environ.setdefault("SARDRONE_GZ_PYTHON", str(gz_python()))
        with Simulation(args.world_dir):
            return _fly(args)
    return _fly(args)


def _fly(args) -> int:
    from sardrone.agent.evaluate import make_model
    from sardrone.mcap.episode_writer import EpisodeMeta
    from sardrone.mission.projection import CameraModel
    from sardrone.mission.search import PROJECT_ROOT, scorecard_confidence
    from sardrone.sim.record_flight import next_episode_path, world_settings

    spec, truth, extra = world_settings(args.world_dir)
    meta = EpisodeMeta(world=spec["world"], weather=spec["weather"], time_of_day=spec["time_of_day"],
                       mission_prompt=args.prompt)  # fmt: skip
    path = args.out or next_episode_path(prefix=f"{spec['id']}-agent")
    ground_z = -spec["drone_body_height_m"]
    if args.fake:
        from sardrone.sim.camera import SyntheticCamera
        from sardrone.sim.drone import FakeDrone
        from sardrone.worlds.sdf import CAMERA_HEIGHT, CAMERA_HFOV_RAD, CAMERA_WIDTH

        drone = FakeDrone()
        sleep = drone.sleep
        camera = SyntheticCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, hfov_deg=math.degrees(CAMERA_HFOV_RAD),
                                 pitch_deg=spec["camera_pitch_deg"])  # fmt: skip
    else:
        from sardrone.sim.camera import GazeboCamera
        from sardrone.sim.mavsdk_drone import MavsdkDrone

        drone, sleep = MavsdkDrone(args.url), asyncio.sleep
        camera = GazeboCamera(model=spec["gz_model"], world=spec["gz_world"], mount_m=spec["camera_mount_m"],
                              pitch_deg=spec["camera_pitch_deg"], boxes_topic=spec.get("boxes_topic")).open()  # fmt: skip
    model_camera = CameraModel.from_source(camera)
    if args.detector == "oracle":
        from sardrone.mission.oracle import OracleDetector

        detector = OracleDetector(truth, model_camera, min_person_px=SearchConfig().min_person_px, ground_z=ground_z)
    else:
        from sardrone.mission.detector import YoloDetector

        run = json.loads((PROJECT_ROOT / "runs" / "train" / args.detector / "run.json").read_text())
        detector = YoloDetector(run["weights"], model_camera, confidence=scorecard_confidence(args.detector),
                                imgsz=run["config"]["imgsz"], device=args.device, ground_z=ground_z,
                                name=f"yolo:{args.detector}")  # fmt: skip
    started = time.monotonic()
    try:
        summary = fly_mission(make_model(args.model, args.cohere_model), args.prompt, drone=drone, camera=camera,
                              detector=detector, truth=truth, spec=spec, path=path, meta=meta, sleep=sleep,
                              max_steps=args.max_steps, extra={**extra, "detector": args.detector})  # fmt: skip
    finally:
        if not args.fake:
            camera.close()
    print(json.dumps(summary, indent=2))
    print(f"wrote {path} ({time.monotonic() - started:.0f} s wall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
