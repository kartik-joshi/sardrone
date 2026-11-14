import json
import math

import pytest

from sardrone.mcap.episode_writer import (
    CAMERA_IN_BODY,
    EVENT_PERSON_SEEN,
    EVENT_TAKEOFF,
    FRAME_BODY,
    FRAME_CAMERA,
    FRAME_WORLD,
    TOPIC_CALIBRATION,
    TOPIC_EVENTS,
    TOPIC_GPS,
    TOPIC_IMAGE,
    TOPIC_POSE,
    TOPIC_TELEMETRY,
    TOPIC_TF,
    EpisodeMeta,
    EpisodeWriter,
)
from sardrone.mcap.frames import Quat, Vec3, rotate_vector
from sardrone.sim.camera import SyntheticCamera, pinhole_intrinsics

IDENTITY = Quat(1.0, 0.0, 0.0, 0.0)


def make_meta() -> EpisodeMeta:
    return EpisodeMeta(
        world="test_world",
        weather="fog",
        time_of_day="dusk",
        mission_prompt="Find the hiker.",
        people_placed=2,
    )


@pytest.fixture
def episode_path(tmp_path):
    return tmp_path / "flight-test.mcap"


def test_every_foxglove_schema_is_protobuf_never_json(episode_path, start_ns, read_episode):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_calibration(start_ns, 640, 480, 381.0, 381.0, 320.0, 240.0)
        ep.log_image(start_ns, SyntheticCamera().read())
        ep.log_gps(start_ns, 47.0, 8.0, 500.0)
        ep.log_pose_ned(start_ns, 0.0, 0.0, 0.0, IDENTITY)
        ep.log_telemetry(
            start_ns,
            battery_percent=90.0,
            battery_voltage_v=15.9,
            flight_mode="HOLD",
            velocity_ned_m_s=(0.0, 0.0, 0.0),
        )
        ep.log_event(start_ns, EVENT_TAKEOFF)

    channels = read_episode(episode_path).channels
    assert channels == {
        TOPIC_IMAGE: ("protobuf", "foxglove.CompressedImage", "protobuf"),
        TOPIC_CALIBRATION: ("protobuf", "foxglove.CameraCalibration", "protobuf"),
        TOPIC_GPS: ("protobuf", "foxglove.LocationFix", "protobuf"),
        TOPIC_POSE: ("protobuf", "foxglove.PoseInFrame", "protobuf"),
        TOPIC_TF: ("protobuf", "foxglove.FrameTransform", "protobuf"),
        TOPIC_EVENTS: ("protobuf", "foxglove.Log", "protobuf"),
        TOPIC_TELEMETRY: ("json", "sardrone.Telemetry", "jsonschema"),
    }
    for message_encoding, schema_name, _ in channels.values():
        if schema_name.startswith("foxglove."):
            assert message_encoding == "protobuf"


def test_pose_is_converted_from_ned_to_z_up(episode_path, start_ns, read_episode):
    # PX4 says: 3 m north, 4 m east, 10 m UP (down = -10), nose north.
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_pose_ned(start_ns, north_m=3.0, east_m=4.0, down_m=-10.0, q_ned_frd=IDENTITY)

    episode = read_episode(episode_path)
    pose = episode.decoded(TOPIC_POSE)[0]
    assert pose.frame_id == FRAME_WORLD
    assert (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) == (4.0, 3.0, 10.0)
    assert pose.pose.position.z > 0, "climbing must make z go UP"

    q = pose.pose.orientation
    forward = rotate_vector(Quat(q.w, q.x, q.y, q.z), Vec3(1.0, 0.0, 0.0))
    assert forward == pytest.approx(Vec3(0.0, 1.0, 0.0), abs=1e-9), "north must stay north"

    # The same pose is also published as the map -> base_link transform.
    tf = episode.decoded(TOPIC_TF)[0]
    assert (tf.parent_frame_id, tf.child_frame_id) == (FRAME_WORLD, FRAME_BODY)
    assert (tf.translation.x, tf.translation.y, tf.translation.z) == (4.0, 3.0, 10.0)
    assert (tf.rotation.w, tf.rotation.x, tf.rotation.y, tf.rotation.z) == (q.w, q.x, q.y, q.z)

    # ...followed by the fixed camera mount, repeated with every pose.
    mount = episode.decoded(TOPIC_TF)[1]
    assert (mount.parent_frame_id, mount.child_frame_id) == (FRAME_BODY, FRAME_CAMERA)
    r = mount.rotation
    assert (r.w, r.x, r.y, r.z) == tuple(CAMERA_IN_BODY)
    assert len(episode.messages[TOPIC_TF]) == 2


def test_timestamps_are_split_correctly_and_match_log_time(episode_path, start_ns, read_episode):
    t = start_ns + 2_123_456_789  # 2.123456789 s into the episode
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_gps(t, 47.0, 8.0, 500.0)

    episode = read_episode(episode_path)
    fix = episode.decoded(TOPIC_GPS)[0]
    assert fix.timestamp.seconds == 1_794_614_402
    assert fix.timestamp.nanos == 123_456_789
    assert 0 <= fix.timestamp.nanos <= 999_999_999
    assert episode.log_times(TOPIC_GPS) == [t]


@pytest.mark.parametrize("bad_ns", [0, 65_000_000_000, 1_794_614_400])
def test_non_wall_clock_times_are_rejected(episode_path, bad_ns):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        with pytest.raises(ValueError, match="before 2020"):
            ep.log_gps(bad_ns, 47.0, 8.0, 500.0)
        with pytest.raises(ValueError, match="before 2020"):
            ep.log_pose_ned(bad_ns, 0.0, 0.0, 0.0, IDENTITY)
        with pytest.raises(ValueError, match="before 2020"):
            ep.log_event(bad_ns, EVENT_TAKEOFF)
    assert ep.message_counts == {}


def test_calibration_fields_and_camera_mount(episode_path, start_ns, read_episode):
    width, height, hfov = 640, 480, 80.0
    fx, fy, cx, cy = pinhole_intrinsics(width, height, hfov)
    assert fx == pytest.approx(381.3611, abs=1e-3)  # 320 / tan(40 deg)
    assert (cx, cy) == (320.0, 240.0)

    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_calibration(start_ns, width, height, fx, fy, cx, cy)
        with pytest.raises(RuntimeError, match="once"):
            ep.log_calibration(start_ns, width, height, fx, fy, cx, cy)

    episode = read_episode(episode_path)
    calibrations = episode.decoded(TOPIC_CALIBRATION)
    assert len(calibrations) == 1
    cal = calibrations[0]
    assert (cal.width, cal.height, cal.frame_id) == (640, 480, FRAME_CAMERA)
    assert list(cal.K) == [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    assert len(cal.D) == 5 and len(cal.R) == 9 and len(cal.P) == 12
    assert list(cal.P) == [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    # A point straight ahead on the optical axis projects to the image centre:
    # [u v w] = K [0 0 1]  ->  (cx, cy)
    assert (cal.K[2] / cal.K[8], cal.K[5] / cal.K[8]) == (320.0, 240.0)


def test_camera_mount_rotation_looks_forward():
    # Optical frame: z into the image, x right, y down. Body frame: FLU.
    assert rotate_vector(CAMERA_IN_BODY, Vec3(0, 0, 1)) == pytest.approx(Vec3(1, 0, 0))  # forward
    assert rotate_vector(CAMERA_IN_BODY, Vec3(1, 0, 0)) == pytest.approx(Vec3(0, -1, 0))  # right
    assert rotate_vector(CAMERA_IN_BODY, Vec3(0, 1, 0)) == pytest.approx(Vec3(0, 0, -1))  # down
    assert math.sqrt(sum(c * c for c in CAMERA_IN_BODY)) == pytest.approx(1.0)


def test_image_is_stored_as_jpeg(episode_path, start_ns, read_episode):
    jpeg = SyntheticCamera().read()
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_image(start_ns, jpeg)
        with pytest.raises(ValueError, match="JPEG"):
            ep.log_image(start_ns, b"\x89PNG not a jpeg")

    image = read_episode(episode_path).decoded(TOPIC_IMAGE)[0]
    assert image.format == "jpeg"
    assert image.frame_id == FRAME_CAMERA
    assert image.data == jpeg


def test_telemetry_velocity_is_stored_z_up(episode_path, start_ns, read_episode):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        # PX4: 3 m/s north, 4 m/s east, climbing at 2 m/s (down = -2).
        ep.log_telemetry(
            start_ns + 500_000_000,
            battery_percent=87.5,
            battery_voltage_v=15.8,
            flight_mode="OFFBOARD",
            velocity_ned_m_s=(3.0, 4.0, -2.0),
        )

    message = read_episode(episode_path).decoded(TOPIC_TELEMETRY)[0]
    assert message["velocity_enu_m_s"] == {"x": 4.0, "y": 3.0, "z": 2.0}
    assert message["ground_speed_m_s"] == pytest.approx(5.0)
    assert message["battery_percent"] == 87.5
    assert message["flight_mode"] == "OFFBOARD"
    assert message["timestamp"] == {"sec": 1_794_614_400, "nsec": 500_000_000}


def test_events_are_log_messages(episode_path, start_ns, read_episode):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_event(start_ns, EVENT_TAKEOFF)
        ep.log_event(start_ns + 1, EVENT_PERSON_SEEN, "near the river")
        with pytest.raises(ValueError, match="unknown event"):
            ep.log_event(start_ns + 2, "exploded")

    logs = read_episode(episode_path).decoded(TOPIC_EVENTS)
    assert [log.message for log in logs] == ["takeoff", "person_seen: near the river"]
    assert all(log.name == "sardrone" for log in logs)
    assert ep.event_counts == {"takeoff": 1, "person_seen": 1}


def test_gps_range_is_checked(episode_path, start_ns):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        with pytest.raises(ValueError, match="latitude"):
            ep.log_gps(start_ns, 91.0, 8.0, 500.0)
        with pytest.raises(ValueError, match="longitude"):
            ep.log_gps(start_ns, 47.0, 181.0, 500.0)


def test_sidecar_has_all_required_fields(episode_path, start_ns, read_episode):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_event(start_ns, EVENT_TAKEOFF)
        ep.log_event(start_ns + 4_000_000_000, EVENT_PERSON_SEEN)
        ep.meta.people_found = 1
        ep.meta.success = False

    sidecar_path = episode_path.with_suffix(".json")
    assert ep.sidecar_path == sidecar_path
    sidecar = json.loads(sidecar_path.read_text())
    for key in (
        "world",
        "weather",
        "time_of_day",
        "mission_prompt",
        "people_placed",
        "people_found",
        "success",
    ):
        assert key in sidecar
    assert sidecar["world"] == "test_world"
    assert sidecar["people_placed"] == 2
    assert sidecar["people_found"] == 1
    assert sidecar["success"] is False
    assert sidecar["episode_id"] == "flight-test"
    assert sidecar["mcap_file"] == "flight-test.mcap"
    assert sidecar["start_time_ns"] == start_ns
    assert sidecar["duration_s"] == 4.0
    assert sidecar["message_counts"] == {TOPIC_EVENTS: 2}

    # The same facts are embedded in the .mcap as a metadata record.
    embedded = read_episode(episode_path).metadata["episode"]
    assert {key: json.loads(value) for key, value in embedded.items()} == sidecar


def test_refuses_to_overwrite_unless_asked(episode_path, start_ns):
    with EpisodeWriter(episode_path, make_meta()) as ep:
        ep.log_event(start_ns, EVENT_TAKEOFF)
    with pytest.raises(Exception):
        EpisodeWriter(episode_path, make_meta())
    with EpisodeWriter(episode_path, make_meta(), overwrite=True) as ep:
        ep.log_event(start_ns, EVENT_TAKEOFF)


def test_logging_after_close_is_an_error_and_close_is_idempotent(episode_path, start_ns):
    ep = EpisodeWriter(episode_path, make_meta())
    ep.close()
    ep.close()
    with pytest.raises(RuntimeError, match="closed"):
        ep.log_event(start_ns, EVENT_TAKEOFF)


def test_path_must_be_mcap(tmp_path):
    with pytest.raises(ValueError, match=".mcap"):
        EpisodeWriter(tmp_path / "flight.bag", make_meta())
