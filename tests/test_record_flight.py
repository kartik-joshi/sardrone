"""Step 6 without PX4: the real mission code, flown by FakeDrone on simulated time."""

import asyncio
import json
import math

import pytest

from sardrone.mcap.episode_writer import (
    TOPIC_CALIBRATION,
    TOPIC_EVENTS,
    TOPIC_GPS,
    TOPIC_IMAGE,
    TOPIC_POSE,
    TOPIC_TELEMETRY,
    EpisodeMeta,
)
from sardrone.mcap.frames import Quat, compass_heading_deg
from sardrone.mcap.timeutil import NS_PER_SEC
from sardrone.sim.drone import FakeDrone
from sardrone.sim.record_flight import (
    FlightPlan,
    MissionTimeout,
    next_episode_path,
    record_flight,
)
from tests.conftest import START_NS, Episode


def make_meta() -> EpisodeMeta:
    return EpisodeMeta(world="w", weather="clear", time_of_day="noon", mission_prompt="square")


def fly(path, drone=None, plan=None):
    drone = drone or FakeDrone()
    writer = asyncio.run(
        record_flight(
            drone, path, make_meta(), plan=plan, sleep=drone.sleep, start_wall_ns=START_NS
        )
    )
    return drone, writer


@pytest.fixture(scope="module")
def flight(tmp_path_factory):
    path = tmp_path_factory.mktemp("flights") / "flight-0001.mcap"
    drone, writer = fly(path)
    return drone, writer, Episode(path)


def test_commands_are_sent_in_mission_order(flight):
    drone, _, _ = flight
    assert drone.commands == [
        "connect",
        "wait_until_ready",
        "arm",
        "takeoff 10",
        "goto 20 0 -10 yaw 0",
        "goto 20 20 -10 yaw 90",
        "goto 0 20 -10 yaw 180",
        "goto 0 0 -10 yaw 270",
        "land",
        "close",
    ]


def test_event_story(flight):
    _, _, episode = flight
    assert [log.message.split(":")[0] for log in episode.decoded(TOPIC_EVENTS)] == [
        "takeoff",
        "waypoint_reached",
        "person_seen",
        "waypoint_reached",
        "waypoint_reached",
        "waypoint_reached",
        "land",
        "land",
    ]
    times = episode.log_times(TOPIC_EVENTS)
    assert times == sorted(times)
    assert times[0] == START_NS


def test_drone_climbs_to_ten_metres_z_up_and_lands(flight):
    _, _, episode = flight
    heights = [pose.pose.position.z for pose in episode.decoded(TOPIC_POSE)]
    assert heights[0] == 0.0
    assert max(heights) == pytest.approx(10.0, abs=1e-3)
    assert min(heights) >= 0.0, "z is up: it must never go negative on this flight"
    assert heights[-1] == pytest.approx(0.0, abs=0.05)


def test_path_is_a_twenty_metre_square(flight):
    _, _, episode = flight
    positions = [pose.pose.position for pose in episode.decoded(TOPIC_POSE)]
    xs = [p.x for p in positions]  # east
    ys = [p.y for p in positions]  # north
    assert min(xs) == pytest.approx(0.0, abs=1e-6) and max(xs) == pytest.approx(20.0, abs=1.0)
    assert min(ys) == pytest.approx(0.0, abs=1.0) and max(ys) == pytest.approx(20.0, abs=1.0)

    # First leg goes NORTH: ENU +y grows while x stays 0.
    first_leg = [p for p in positions if p.z > 9.9 and p.x == 0.0]
    assert max(p.y for p in first_leg) > 15.0

    # Horizontal distance flown is about 4 sides of 20 m.
    travelled = sum(
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(positions, positions[1:])
    )
    assert travelled == pytest.approx(80.0, abs=4.0)


def test_heading_follows_each_leg(flight):
    _, _, episode = flight
    headings = []
    for pose in episode.decoded(TOPIC_POSE):
        q = pose.pose.orientation
        heading = round(compass_heading_deg(Quat(q.w, q.x, q.y, q.z))) % 360
        if not headings or headings[-1] != heading:
            headings.append(heading)
    assert headings == [0, 90, 180, 270]  # north, east, south, west


def test_all_streams_share_one_clock_at_ten_hz(flight):
    _, writer, episode = flight
    pose_times = episode.log_times(TOPIC_POSE)
    assert pose_times[0] == START_NS
    assert {b - a for a, b in zip(pose_times, pose_times[1:])} == {NS_PER_SEC // 10}
    for topic in (TOPIC_GPS, TOPIC_TELEMETRY, TOPIC_IMAGE):
        assert episode.log_times(topic) == pose_times
    assert len(episode.messages[TOPIC_CALIBRATION]) == 1
    # PX4 boot-relative time (65 s) must not leak into the file.
    assert episode.statistics.message_start_time == START_NS
    assert writer.sidecar()["duration_s"] == pytest.approx(
        (pose_times[-1] - pose_times[0]) / NS_PER_SEC
    )


def test_telemetry_tracks_modes_and_battery(flight):
    _, _, episode = flight
    telemetry = episode.decoded(TOPIC_TELEMETRY)
    modes = []
    for message in telemetry:
        if not modes or modes[-1] != message["flight_mode"]:
            modes.append(message["flight_mode"])
    # The recorder moves on at 9.5 m, so the brief HOLD after takeoff is skipped.
    assert modes == ["READY", "TAKEOFF", "OFFBOARD", "LAND", "READY"]
    assert telemetry[0]["battery_percent"] == 100.0
    assert telemetry[-1]["battery_percent"] < 100.0
    cruise = [m for m in telemetry if m["flight_mode"] == "OFFBOARD"]
    assert max(m["ground_speed_m_s"] for m in cruise) == pytest.approx(4.0, abs=1e-6)
    climb = [m for m in telemetry if m["flight_mode"] == "TAKEOFF"]
    assert all(m["velocity_enu_m_s"]["z"] > 0 for m in climb[1:]), "climbing = +z velocity"


def test_sidecar_reports_success(flight):
    _, writer, _ = flight
    sidecar = json.loads(writer.sidecar_path.read_text())
    assert sidecar["people_placed"] == 1
    assert sidecar["people_found"] == 1
    assert sidecar["success"] is True
    assert sidecar["event_counts"] == {
        "land": 2,
        "person_seen": 1,
        "takeoff": 1,
        "waypoint_reached": 4,
    }


def test_no_person_placed_means_no_sighting(tmp_path):
    plan = FlightPlan(person_north_east_m=None)
    _, writer = fly(tmp_path / "flight-0001.mcap", plan=plan)
    sidecar = writer.sidecar()
    assert sidecar["people_placed"] == 0 and sidecar["people_found"] == 0
    assert "person_seen" not in sidecar["event_counts"]
    assert sidecar["success"] is True


def test_person_out_of_reach_means_mission_not_successful(tmp_path):
    plan = FlightPlan(person_north_east_m=(200.0, 200.0))
    _, writer = fly(tmp_path / "flight-0001.mcap", plan=plan)
    sidecar = writer.sidecar()
    assert sidecar["people_placed"] == 1 and sidecar["people_found"] == 0
    assert sidecar["success"] is False


class StuckDrone(FakeDrone):
    """Accepts goto commands but never moves sideways."""

    async def goto_ned(self, north_m, east_m, down_m, yaw_deg):
        self.commands.append("goto (ignored)")


def test_timeout_still_lands_and_leaves_a_valid_episode(tmp_path):
    path = tmp_path / "flight-0001.mcap"
    drone = StuckDrone()
    plan = FlightPlan(phase_timeout_s=5.0)
    with pytest.raises(MissionTimeout, match="corner 1"):
        asyncio.run(
            record_flight(
                drone, path, make_meta(), plan=plan, sleep=drone.sleep, start_wall_ns=START_NS
            )
        )
    assert drone.commands[-2:] == ["land", "close"]

    # The partial episode is still readable and honestly marked as failed.
    episode = Episode(path)
    assert len(episode.messages[TOPIC_POSE]) > 50
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["success"] is False


def test_next_episode_path_counts_up(tmp_path):
    assert next_episode_path(tmp_path).name == "flight-0001.mcap"
    (tmp_path / "flight-0001.mcap").touch()
    (tmp_path / "flight-0002.mcap").touch()
    assert next_episode_path(tmp_path).name == "flight-0003.mcap"
