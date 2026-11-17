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
    EVENT_TARGET_ACQUIRED,
    EVENT_TARGET_LOST,
    EVENT_WAYPOINT_REACHED,
    EpisodeMeta,
    EpisodeWriter,
)
from sardrone.mcap.geo import enu_offset_to_latlon
from sardrone.mcap.timeutil import NS_PER_SEC
from sardrone.mission.coverage import CoveragePlan, Point, covered_fraction, lawnmower
from sardrone.mission.metrics import cte_summary
from sardrone.mission.tracker import Tracker, centre
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
    # Tracker (object memory). A sighting is acted on only once its track
    # has been seen `track_min_hits` times; a track may coast unseen for
    # `track_max_age` frames (at 10 Hz, 15 frames = 1.5 s).
    track_min_hits: int = 3
    track_max_age: int = 15
    # Two position estimates closer than this are the same person. Used
    # when the detector gives no identity (a real detector never does).
    same_person_radius_m: float = 4.0
    # A report within this distance of a true person counts as finding them.
    match_radius_m: float = 3.0
    missed_rescue_min_frames: int = 10  # "was visible": in at least this many frames
    # A second, lower pass over tree cover after the main search (None = off).
    second_pass_altitude_m: float | None = None
    # Stricter rules during the second pass. From low over trees the
    # detector sees trunks and canopy shadows large and close and fires on
    # them far more often (19 false alarms against 3 in the first test).
    # None = same as the main search.
    second_pass_track_min_hits: int | None = None  # sightings before a track is acted on
    second_pass_confirm_min_frames: int | None = None  # frames in view during the orbit
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
        cover_polygon: list[Point] | None = None,
        sleep=asyncio.sleep,
        start_wall_ns: int | None = None,
    ):
        config = config or SearchConfig()
        # Where trees stand, as (north, east) corners: map knowledge the
        # mission is given, like a land-cover layer. Only used by the second pass.
        self.cover_polygon = cover_polygon
        self.second_plan: CoveragePlan | None = None
        self._search_altitude_m = config.altitude_m
        self._in_second_pass = False
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
        # Perception, refreshed once per camera frame by _sample().
        self.tracker = Tracker(max_age=config.track_max_age, min_hits=config.track_min_hits)
        self.sightings: list[Sighting] = []  # what the detector saw in the latest frame
        self._actionable: list[Sighting] = []  # ... of which belong to a believed track
        self._spots: dict[str, tuple[float, float]] = {}  # key -> (east, north) of id-less targets
        self._acquired: set[int] = set()
        self._lost_seen = 0
        # Bookkeeping for the mission metrics.
        self.visible_frames: dict[str, int] = {}  # person id -> frames with a ground-truth box
        self.false_alarms = 0
        self.duplicates_avoided = 0  # confirmations that ended on an already-reported spot
        self._search_path: list[tuple[float, float]] = []  # positions flown while searching
        self._lanes_done = 0
        self._ade = {"all": [0.0, 0], "coasting": [0.0, 0]}  # sum of pixel errors, count
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

    async def _sample(self) -> None:
        """One camera frame: record it (as before), then LOOK at it.

        The detector runs exactly once per frame. Its boxes go through the
        tracker, which turns flickering per-frame boxes into persistent
        tracks. What the mission later acts on is `self._actionable`: the
        sightings that belong to a track seen often enough to be believed.
        """
        await super()._sample()
        pose = DronePose.from_state(self.state)
        self.sightings = [
            s for s in self.detector.detect(pose, self.last_jpeg) if s.ground_east_north is not None
        ]
        boxes = [(s.bbox.x0, s.bbox.y0, s.bbox.x1, s.bbox.y1) for s in self.sightings]
        tracks = self.tracker.step(boxes)
        self._actionable = [self.sightings[t.detection] for t in tracks if t.detection is not None]

        self.episode.log_detections(
            self.now_ns,
            [
                {"id": f"track {t.id}" + (" (coasting)" if t.coasting else ""),
                 "x0": t.box[0], "y0": t.box[1], "x1": t.box[2], "y1": t.box[3]}
                for t in tracks
            ],
        )  # fmt: skip
        for track in tracks:
            if track.id not in self._acquired:
                self._acquired.add(track.id)
                self._event(EVENT_TARGET_ACQUIRED, f"track {track.id}")
        for track in self.tracker.lost[self._lost_seen :]:
            if track.id in self._acquired:
                self._event(EVENT_TARGET_LOST, f"track {track.id} after {track.age} frames")
        self._lost_seen = len(self.tracker.lost)
        self._score_frame(tracks)
        if self.mission_state == SEARCH:
            self._search_path.append((self.state.north_m, self.state.east_m))

    def _score_frame(self, tracks) -> None:
        """Compare this frame's tracks with the simulator's label boxes,
        when the camera supplies them (the Gazebo camera does)."""
        truth_boxes = getattr(self.camera, "last_boxes", None)
        if truth_boxes is None:
            return
        centres = []
        for box in truth_boxes:
            if box.width < 1.0 or box.height < 1.0:
                continue
            person = self._label_ids.get(box.label, f"label_{box.label}")
            self.visible_frames[person] = self.visible_frames.get(person, 0) + 1
            centres.append((((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2), max(box.width, box.height)))
        for track in tracks:
            if track.predicted is None or not centres:
                continue
            # ADE uses the PREDICTION (made before looking at this frame):
            # that is the tracker's memory at work, not the detector's box.
            guess = centre(track.predicted)
            (true_centre, size) = min(centres, key=lambda item: math.dist(guess, item[0]))
            error = math.dist(guess, true_centre)
            if error <= max(40.0, 2.0 * size):  # farther away: this track is not that person
                for bucket in ("all", "coasting") if track.coasting else ("all",):
                    self._ade[bucket][0] += error
                    self._ade[bucket][1] += 1

    def _key(self, sighting: Sighting) -> str:
        """Who is this? The oracle says; for a real detector the answer is
        "whoever we already know of near that spot", else a new target."""
        if sighting.person_id:
            return sighting.person_id
        east, north = sighting.ground_east_north
        for key, (known_east, known_north) in self._spots.items():
            if math.hypot(east - known_east, north - known_north) <= self.config.same_person_radius_m:
                return key
        return f"target_{len(self._spots) + 1}"

    def _same_target(self, sighting: Sighting, key: str, east: float, north: float) -> bool:
        if sighting.person_id:
            return sighting.person_id == key
        here = sighting.ground_east_north
        return math.hypot(here[0] - east, here[1] - north) <= self.config.same_person_radius_m

    def _new_sightings(self) -> list[Sighting]:
        """Believed sightings of people not yet reported (and not recently rejected)."""
        fresh = []
        for sighting in self._actionable:
            key = self._key(sighting)
            if key in self.confirmed or self.now_ns < self._rejected_until.get(key, 0):
                continue
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
        key = self._key(sighting)
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
            # _sample() has just run the detector on this frame.
            for seen in self.sightings:
                if self._same_target(seen, key, east, north):
                    frames += 1
                    estimates.append(seen.ground_east_north)
                    break

        if not estimates:
            return False, (east, north), 0
        # Each estimate is a little short, TOWARDS the camera (the bottom of
        # the box is the near side of the body). Seen from many sides,
        # those errors partly cancel, so the average beats any single view.
        mean = (sum(e[0] for e in estimates) / len(estimates), sum(e[1] for e in estimates) / len(estimates))
        needed = config.confirm_min_frames
        if self._in_second_pass and config.second_pass_confirm_min_frames:
            needed = config.second_pass_confirm_min_frames
        return frames >= needed, mean, frames

    def _already_reported_near(self, estimate: tuple[float, float]) -> str | None:
        """The target already reported within `same_person_radius_m` of
        this position, if any."""
        east, north = estimate
        for report in self.reports:
            if math.hypot(east - report["east_m"], north - report["north_m"]) <= self.config.same_person_radius_m:
                return report["target"]
        return None

    def _match_truth(self, sighting: Sighting, east: float, north: float) -> tuple[str | None, float | None]:
        """Which true person does a report belong to, and how far off is it?

        The oracle names the person. Otherwise it is the nearest true
        person within `match_radius_m` who has not been reported yet; if
        there is none, the report is a FALSE report (nobody is there).
        """
        already = {report["matched_person"] for report in self.reports}
        if sighting.person_id and sighting.person_id in self.truth:
            centre_ = self.truth[sighting.person_id]["box_centre"]
            return sighting.person_id, math.hypot(east - centre_[0], north - centre_[1])
        best, best_distance = None, self.config.match_radius_m
        for person_id, person in self.truth.items():
            if person_id in already:
                continue
            distance = math.hypot(east - person["box_centre"][0], north - person["box_centre"][1])
            if distance <= best_distance:
                best, best_distance = person_id, distance
        return best, (best_distance if best else None)

    def _report(self, sighting: Sighting, estimate: tuple[float, float], frames: int, key: str) -> None:
        east, north = estimate
        latitude, longitude = enu_offset_to_latlon(*self._home_latlon, east, north)
        matched, error = self._match_truth(sighting, east, north)
        report = {
            "person_id": sighting.person_id,
            "target": key,
            "matched_person": matched,  # None = nobody is really there
            "east_m": round(east, 2),
            "north_m": round(north, 2),
            "latitude_deg": round(latitude, 7),
            "longitude_deg": round(longitude, 7),
            "frames_in_view": frames,
            "time_s": round((self.now_ns - self._start_wall_ns) / NS_PER_SEC, 1),
        }
        if error is not None:
            report["error_m"] = round(error, 2)
        self.reports.append(report)
        self.confirmed.add(key)
        self._spots[key] = (east, north)
        if matched:
            self.episode.meta.people_found += 1
        detail = f", error {error:.2f} m" if error is not None else ", NOBODY THERE" if self.truth else ""
        self._event(EVENT_PERSON_REPORTED, f"{sighting.person_id or key} at {latitude:.6f}, {longitude:.6f}{detail}")

    async def _handle_sighting(self, sighting: Sighting) -> None:
        """CONFIRM, then REPORT or reject, then RESUME back to the lane."""
        key = self._key(sighting)
        name = sighting.person_id or key
        self._spots.setdefault(key, sighting.ground_east_north)
        resume_at = self._position()
        resume_yaw = self._yaw_deg
        size = max(sighting.bbox.width, sighting.bbox.height)
        self._event(EVENT_PERSON_SEEN, f"{name}, {size:.0f} px in the image")
        self._enter(CONFIRM, f"{name} sighted")
        confirmed, estimate, frames = await self._confirm(sighting)
        duplicate = self._already_reported_near(estimate) if confirmed else None
        if confirmed and duplicate:
            # The orbit ended on a spot that has been reported already: the
            # same object, first sighted from far enough away that its rough
            # position did not match. Reporting it again would send the
            # rescue team to one place twice (the first second-pass flights
            # did exactly that).
            # Remember this target where it was SIGHTED (kept in _spots), so
            # the next far-away sighting of it maps to it and is ignored.
            self.confirmed.add(key)
            self.duplicates_avoided += 1
            self._event(EVENT_PERSON_CONFIRMED, f"{name} in view for {frames} frames, same place as {duplicate}")
            self._enter(RESUME, f"already reported as {duplicate}")
        elif confirmed:
            self._event(EVENT_PERSON_CONFIRMED, f"{name} in view for {frames} frames")
            self._enter(REPORT, name)
            self._report(sighting, estimate, frames, key)
            self._enter(RESUME, "reported")
        else:
            until = self.now_ns + int(self.config.retry_after_s * NS_PER_SEC)
            self._rejected_until[key] = until
            self.false_alarms += 1
            self._event(EVENT_FALSE_ALARM, f"{name} in view for only {frames} frames")
            self._enter(RESUME, "not confirmed")
        await self._fly_line(
            (resume_at[0], resume_at[1], self._down_for(self._search_altitude_m)),
            self.config.search_speed_m_s,
            "return to the search lane",
        )
        self._yaw_deg = resume_yaw

    async def _fly_waypoints(self, waypoints, altitude_m: float, primary: bool) -> str:
        """Fly a list of lane waypoints at `altitude_m`, handling sightings
        on the way. Returns "search complete" or "battery low"."""
        config = self.config
        self._search_altitude_m = altitude_m
        down = self._down_for(altitude_m)
        index = 0
        while index < len(waypoints):
            waypoint = waypoints[index]
            # On a lane the nose points along the lane, so the camera sweeps
            # the strip it was planned for, also right after an interruption.
            lane_yaw = None
            if waypoint.kind == "lane_end":
                begin = waypoints[index - 1]
                lane_yaw = bearing_deg((begin.north_m, begin.east_m), (waypoint.north_m, waypoint.east_m))
            interruption = await self._fly_line(
                (waypoint.north_m, waypoint.east_m, down),
                config.search_speed_m_s,
                f"reach {waypoint.kind} of lane {waypoint.lane}",
                yaw_deg=lane_yaw,
                watch=self._watch_search,
            )
            if interruption is None:
                label = "lane" if primary else "second-pass lane"
                self._event(EVENT_WAYPOINT_REACHED, f"{label} {waypoint.lane} {waypoint.kind}")
                if waypoint.kind == "lane_end" and primary:
                    self._lanes_done += 1
                index += 1
                continue
            kind, sighting = interruption
            if kind == "battery":
                return "battery low"
            await self._handle_sighting(sighting)
            if self._battery_low():
                return "battery low"
            self._enter(SEARCH, "back on the lane")
        return "search complete"

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

        self._planned_path = [home] + [(w.north_m, w.east_m) for w in self.coverage.waypoints]
        self._enter(SEARCH, f"{self.coverage.lanes} lanes, {self.coverage.spacing_m:.0f} m apart")
        reason = await self._fly_waypoints(self.coverage.waypoints, config.altitude_m, primary=True)

        if reason == "search complete" and config.second_pass_altitude_m and self.cover_polygon:
            # Second pass: lower, and at right angles to the first, over the
            # part of the area with tree cover. Lower makes people larger;
            # crossing the first lanes looks under the canopies from another
            # side. Every miss in the first closed-loop batch was a person
            # under trees.
            low_view = footprint(self.model, config.second_pass_altitude_m - self.ground_z, config.min_person_px)
            here = (self.state.north_m, self.state.east_m)
            self.second_plan = lawnmower(
                self.cover_polygon, low_view, overlap=config.overlap,
                heading_deg=(self.coverage.heading_deg + 90.0) % 180.0, start=here,
            )  # fmt: skip
            self._planned_path += [(w.north_m, w.east_m) for w in self.second_plan.waypoints]
            self._in_second_pass = True
            if config.second_pass_track_min_hits:
                self.tracker.min_hits = config.second_pass_track_min_hits
            self._event(
                EVENT_STATE,
                f"SEARCH second pass: {self.second_plan.lanes} lanes at "
                f"{config.second_pass_altitude_m:g} m over tree cover",
            )
            reason = await self._fly_waypoints(
                self.second_plan.waypoints, config.second_pass_altitude_m, primary=False
            )

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
                "second_pass": None if self.second_plan is None else {
                    "altitude_m": self.config.second_pass_altitude_m,
                    "lanes": self.second_plan.lanes,
                    "lane_heading_deg": round(self.second_plan.heading_deg, 1),
                    "planned_path_m": round(self.second_plan.path_length_m(), 1),
                    "track_min_hits": self.tracker.min_hits,
                    "confirm_min_frames": self.config.second_pass_confirm_min_frames or self.config.confirm_min_frames,
                },
            },
            "transitions": [state for state, _ in self.transitions],
            "reports": self.reports,
            "mission_metrics": self.mission_metrics(),
        }  # fmt: skip

    def mission_metrics(self) -> dict:
        """How the mission went, as numbers (see sardrone/mission/metrics.py).

        A value of None means "not measurable in this run", for example
        missed rescues when the camera supplies no ground-truth boxes.
        """
        config, coverage = self.config, self.coverage
        found = {r["matched_person"] for r in self.reports if r["matched_person"]}
        true_reports = [r for r in self.reports if r["matched_person"]]
        missed = sorted(set(self.truth) - found)
        missed_rescues = None
        if self.visible_frames or getattr(self.camera, "last_boxes", None) is not None:
            missed_rescues = sorted(
                p for p in missed if self.visible_frames.get(p, 0) >= config.missed_rescue_min_frames
            )
        coverage_percent = None
        if coverage is not None and coverage.lanes:
            coverage_percent = round(
                100.0 * covered_fraction(self.polygon, coverage) * self._lanes_done / coverage.lanes, 1
            )
        cte = None
        if coverage is not None and len(self._search_path) > 1:
            # Detours to confirm a sighting are flown in other states, so
            # what is left is the lanes and the transits between them.
            cte = cte_summary(self._search_path, getattr(self, "_planned_path", self._search_path))
        errors = [r["error_m"] for r in true_reports if "error_m" in r]

        def ade(bucket):
            total, count = self._ade[bucket]
            return {"mean_px": round(total / count, 2), "samples": count} if count else None

        return {
            "people_placed": len(self.truth),
            "people_found": len(found),
            "missed": missed,
            "missed_rescues": missed_rescues,  # visible in the images, never reported
            "false_reports": sum(1 for r in self.reports if not r["matched_person"]),
            "false_alarms": self.false_alarms,  # sightings the confirmation orbit rejected
            "duplicates_avoided": self.duplicates_avoided,
            "time_to_first_find_s": min((r["time_s"] for r in true_reports), default=None),
            "mean_report_error_m": round(sum(errors) / len(errors), 2) if errors else None,
            "coverage_percent": coverage_percent,
            "cross_track_error": cte,
            "tracker_ade": ade("all"),
            "tracker_ade_coasting": ade("coasting"),
            "tracks_started": len(self._acquired),
        }


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
    cover_polygon: list[Point] | None = None,
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
                ground_z=ground_z, cover_polygon=cover_polygon, sleep=sleep, start_wall_ns=start_wall_ns,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def scorecard_confidence(version: str) -> float:
    """The cut-off chosen for this model on validation data (scorecard.csv)."""
    import csv

    with open(PROJECT_ROOT / "docs" / "results" / "scorecard.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["version"] == version:
                return float(row["confidence"])
    raise ValueError(f"model {version!r} is not in the scorecard; pass --confidence")


def square_polygon(side_m: float) -> list[Point]:
    """A square search area centred on the takeoff point, as (north, east) corners."""
    half = side_m / 2
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


def tree_cover_polygon(spec: dict, margin_m: float = 2.0) -> list[Point] | None:
    """A rectangle around all trees and their canopies, cut to the search
    area: (north, east) corners. None if the world has no trees.

    This is the simplest possible land-cover map. In a meadow with a few
    scattered trees it covers most of the area, and the second pass then
    costs nearly a second search.
    """
    from sardrone.worlds.assets import TREES

    if not spec["trees"]:
        return None
    half = spec["area_m"] / 2
    reach = [
        (t["x"], t["y"], max(abs(v) for v in TREES[t["kind"]].lo[:2] + TREES[t["kind"]].hi[:2]) * t["scale"] + margin_m)
        for t in spec["trees"]
    ]
    east0 = max(-half, min(x - r for x, _, r in reach))
    east1 = min(half, max(x + r for x, _, r in reach))
    north0 = max(-half, min(y - r for _, y, r in reach))
    north1 = min(half, max(y + r for _, y, r in reach))
    return [(north0, east0), (north1, east0), (north1, east1), (north0, east1)]


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
        "--detector", default="oracle",
        help="'oracle' (reads ground truth) or a model version under runs/train, e.g. v1",
    )  # fmt: skip
    parser.add_argument("--confidence", type=float, help="detector cut-off (default: the one in the scorecard)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--second-pass", action="store_true",
                        help="after the search, fly a lower pass over the tree cover")  # fmt: skip
    parser.add_argument("--strict-second-pass", action="store_true",
                        help="during the second pass, act on a track only after 6 sightings and "
                             "confirm only after 20 frames in view")  # fmt: skip
    parser.add_argument(
        "--prompt", default="Search the area, confirm and report every person, then return home."
    )
    args = parser.parse_args(argv)

    spec, truth, extra = world_settings(args.world_dir)
    config = SearchConfig(altitude_m=spec["altitude_m"], battery_rtl_percent=args.battery_rtl)
    config.confirm_altitude_m = safe_confirm_altitude(spec, config.confirm_altitude_m)
    cover = None
    if args.second_pass:
        cover = tree_cover_polygon(spec)
        # As low as is safe above the trees, and never above the main search.
        config.second_pass_altitude_m = min(spec["altitude_m"], max(15.0, safe_confirm_altitude(spec, 10.0)))
        if args.strict_second_pass:
            config.second_pass_track_min_hits, config.second_pass_confirm_min_frames = 6, 20
    ground_z = -spec["drone_body_height_m"]
    meta = EpisodeMeta(
        world=spec["world"], weather=spec["weather"], time_of_day=spec["time_of_day"],
        mission_prompt=args.prompt,
    )  # fmt: skip
    suffix = "-fake" if args.fake else "" if args.detector == "oracle" else f"-{args.detector}"
    path = args.out or next_episode_path(prefix=f"{spec['id']}-search{suffix}")
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

    detector = None  # run_search falls back to the oracle
    if args.detector != "oracle":
        if args.fake:
            print("error: a trained detector needs real images, not --fake", file=sys.stderr)
            return 1
        from sardrone.mission.detector import YoloDetector

        run = json.loads((PROJECT_ROOT / "runs" / "train" / args.detector / "run.json").read_text())
        confidence = args.confidence if args.confidence is not None else scorecard_confidence(args.detector)
        detector = YoloDetector(
            run["weights"], CameraModel.from_source(camera), confidence=confidence,
            imgsz=run["config"]["imgsz"], device=args.device, ground_z=ground_z, name=f"yolo:{args.detector}",
        )  # fmt: skip
        extra = {**extra, "detector_confidence": confidence, "detector_weights_hash": run.get("export_hash")}

    started = time.monotonic()
    try:
        episode, recorder = asyncio.run(
            run_search(
                drone, path, meta, camera=camera, polygon=square_polygon(spec["area_m"]),
                truth=truth, config=config, detector=detector, ground_z=ground_z, cover_polygon=cover,
                sleep=sleep,
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
