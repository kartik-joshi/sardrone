"""Step 5: a full 10-second episode, written and read back with no PX4 running."""

import io
import json
import math

import pytest
from PIL import Image

from sardrone.mcap.episode_writer import (
    TOPIC_CALIBRATION,
    TOPIC_EVENTS,
    TOPIC_GPS,
    TOPIC_IMAGE,
    TOPIC_POSE,
    TOPIC_TELEMETRY,
    TOPIC_TF,
)
from sardrone.mcap.frames import Quat, Vec3, compass_heading_deg, rotate_vector
from sardrone.mcap.geo import METERS_PER_DEG_LAT, enu_offset_to_latlon, latlon_to_enu_offset
from sardrone.mcap.timeutil import NS_PER_SEC
from sardrone.sim.synthetic import (
    HEIGHT_M,
    HOME_LAT_DEG,
    HOME_LON_DEG,
    RADIUS_M,
    RATE_HZ,
    write_synthetic_episode,
)

DURATION_S = 10.0
N_SAMPLES = int(DURATION_S * RATE_HZ) + 1  # 101: both endpoints included


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    from tests.conftest import START_NS, Episode

    path = tmp_path_factory.mktemp("episodes") / "synthetic-test.mcap"
    writer = write_synthetic_episode(path, duration_s=DURATION_S, start_wall_ns=START_NS)
    return writer, Episode(path), START_NS


def test_file_is_a_valid_mcap(written):
    writer, episode, _ = written
    data = writer.path.read_bytes()
    magic = b"\x89MCAP0\r\n"
    assert data.startswith(magic) and data.endswith(magic)
    assert episode.statistics.message_count == sum(writer.message_counts.values())


def test_message_counts(written):
    writer, episode, _ = written
    assert {topic: len(msgs) for topic, msgs in episode.messages.items()} == {
        TOPIC_IMAGE: N_SAMPLES,
        TOPIC_GPS: N_SAMPLES,
        TOPIC_POSE: N_SAMPLES,
        TOPIC_TELEMETRY: N_SAMPLES,
        TOPIC_TF: 2 * N_SAMPLES,  # drone pose + camera mount, every sample
        TOPIC_CALIBRATION: 1,
        TOPIC_EVENTS: 6,
    }


def test_episode_lasts_exactly_ten_seconds_on_one_clock(written):
    _, episode, start_ns = written
    assert episode.statistics.message_start_time == start_ns
    assert episode.statistics.message_end_time == start_ns + 10 * NS_PER_SEC

    for topic in (TOPIC_IMAGE, TOPIC_GPS, TOPIC_POSE, TOPIC_TELEMETRY):
        times = episode.log_times(topic)
        assert times[0] == start_ns
        assert times[-1] == start_ns + 10 * NS_PER_SEC
        steps = {b - a for a, b in zip(times, times[1:])}
        assert steps == {NS_PER_SEC // RATE_HZ}, f"{topic} is not evenly spaced at 10 Hz"


def test_message_timestamps_equal_log_times(written):
    _, episode, _ = written
    for topic in (TOPIC_IMAGE, TOPIC_GPS, TOPIC_POSE, TOPIC_TF, TOPIC_EVENTS, TOPIC_CALIBRATION):
        for log_time, message in episode.messages[topic]:
            assert 0 <= message.timestamp.nanos <= 999_999_999
            assert message.timestamp.seconds * NS_PER_SEC + message.timestamp.nanos == log_time
    for log_time, message in episode.messages[TOPIC_TELEMETRY]:
        ts = message["timestamp"]
        assert ts["sec"] * NS_PER_SEC + ts["nsec"] == log_time


def test_pose_traces_a_circle_at_ten_metres_up(written):
    _, episode, _ = written
    poses = episode.decoded(TOPIC_POSE)
    for pose in poses:
        p = pose.pose.position
        assert p.z == pytest.approx(HEIGHT_M, abs=1e-9), "z must be UP (+10), not down"
        assert math.hypot(p.x, p.y) == pytest.approx(RADIUS_M, abs=1e-9)

    # Starts due north of home (ENU +y), a quarter lap later it is due east (+x).
    first, quarter, last = poses[0].pose.position, poses[25].pose.position, poses[-1].pose.position
    assert (first.x, first.y) == pytest.approx((0.0, RADIUS_M), abs=1e-9)
    assert (quarter.x, quarter.y) == pytest.approx((RADIUS_M, 0.0), abs=1e-9)
    assert (last.x, last.y) == pytest.approx((first.x, first.y), abs=1e-9)  # full lap


def test_heading_follows_the_direction_of_travel(written):
    _, episode, _ = written
    poses = episode.decoded(TOPIC_POSE)

    def heading(index):
        q = poses[index].pose.orientation
        return compass_heading_deg(Quat(q.w, q.x, q.y, q.z))

    # Clockwise circle starting at the north point: east, south, west, north.
    assert heading(0) == pytest.approx(90.0, abs=1e-6)
    assert heading(25) == pytest.approx(180.0, abs=1e-6)
    assert heading(50) == pytest.approx(270.0, abs=1e-6)
    assert heading(75) == pytest.approx(0.0, abs=1e-6) or heading(75) == pytest.approx(360.0, abs=1e-6)

    # The drone stays level the whole way: body up == world up.
    for pose in poses:
        q = pose.pose.orientation
        up = rotate_vector(Quat(q.w, q.x, q.y, q.z), Vec3(0.0, 0.0, 1.0))
        assert up == pytest.approx(Vec3(0.0, 0.0, 1.0), abs=1e-9)


def test_gps_circle_matches_the_pose_circle(written):
    _, episode, _ = written
    fixes = episode.decoded(TOPIC_GPS)
    poses = episode.decoded(TOPIC_POSE)
    for fix, pose in zip(fixes, poses):
        east, north = latlon_to_enu_offset(HOME_LAT_DEG, HOME_LON_DEG, fix.latitude, fix.longitude)
        assert east == pytest.approx(pose.pose.position.x, abs=1e-6)  # within a micrometre
        assert north == pytest.approx(pose.pose.position.y, abs=1e-6)
        assert fix.altitude == pytest.approx(498.0)

    # Starting point is 20 m north of home: latitude up by 20 / 111_319.49 degrees.
    assert fixes[0].latitude - HOME_LAT_DEG == pytest.approx(20.0 / METERS_PER_DEG_LAT, abs=1e-12)
    assert fixes[0].latitude > HOME_LAT_DEG
    assert fixes[0].longitude == pytest.approx(HOME_LON_DEG, abs=1e-12)
    assert fixes[25].longitude > HOME_LON_DEG  # a quarter lap later: east of home


def test_geo_helper_numbers():
    assert METERS_PER_DEG_LAT == pytest.approx(111_319.49, abs=0.01)
    lat, lon = enu_offset_to_latlon(0.0, 0.0, east_m=111_319.49, north_m=111_319.49)
    assert (lat, lon) == pytest.approx((1.0, 1.0), abs=1e-7)
    # At 60 deg north a degree of longitude is half as long, so 1 km east is 2x the degrees.
    _, lon60 = enu_offset_to_latlon(60.0, 0.0, east_m=1000.0, north_m=0.0)
    _, lon0 = enu_offset_to_latlon(0.0, 0.0, east_m=1000.0, north_m=0.0)
    assert lon60 / lon0 == pytest.approx(2.0, abs=1e-9)


def test_velocity_matches_circle_speed(written):
    _, episode, _ = written
    expected_speed = 2.0 * math.pi * RADIUS_M / DURATION_S  # 12.566 m/s
    for message in episode.decoded(TOPIC_TELEMETRY):
        assert message["ground_speed_m_s"] == pytest.approx(expected_speed, abs=1e-9)
        assert message["velocity_enu_m_s"]["z"] == 0.0
    first = episode.decoded(TOPIC_TELEMETRY)[0]["velocity_enu_m_s"]
    assert (first["x"], first["y"]) == pytest.approx((expected_speed, 0.0), abs=1e-9)  # heading east


def test_images_decode_as_640x480_jpegs(written):
    _, episode, _ = written
    images = episode.decoded(TOPIC_IMAGE)
    for message in (images[0], images[60], images[-1]):
        assert message.format == "jpeg"
        picture = Image.open(io.BytesIO(message.data))
        assert picture.format == "JPEG"
        assert picture.size == (640, 480)

    # The orange "person" marker is at the image centre only around t = 6 s.
    def centre_pixel(index):
        return Image.open(io.BytesIO(images[index].data)).convert("RGB").getpixel((320, 245))

    r, g, b = centre_pixel(60)
    assert r > 200 and b < 80, f"expected orange person marker, got {(r, g, b)}"
    r, g, b = centre_pixel(10)
    assert r < 150, f"expected plain ground, got {(r, g, b)}"


def test_events_tell_the_mission_story_in_order(written):
    _, episode, start_ns = written
    events = [(t, log.message) for t, log in episode.messages[TOPIC_EVENTS]]
    assert [message.split(":")[0] for _, message in events] == [
        "takeoff",
        "waypoint_reached",
        "waypoint_reached",
        "person_seen",
        "waypoint_reached",
        "land",
    ]
    seconds = [(t - start_ns) / NS_PER_SEC for t, _ in events]
    assert seconds == [0.0, 2.5, 5.0, 6.0, 7.5, 10.0]


def test_sidecar(written):
    writer, _, start_ns = written
    sidecar = json.loads(writer.sidecar_path.read_text())
    assert sidecar["world"] == "synthetic_circle"
    assert sidecar["people_placed"] == 1
    assert sidecar["people_found"] == 1
    assert sidecar["success"] is True
    assert sidecar["start_time_ns"] == start_ns
    assert sidecar["duration_s"] == 10.0
    assert sidecar["message_counts"][TOPIC_IMAGE] == N_SAMPLES
