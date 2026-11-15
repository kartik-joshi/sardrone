import math

import pytest

from sardrone.mcap.frames import (
    Quat,
    Vec3,
    compass_heading_deg,
    enu_to_ned,
    enu_yaw_deg,
    frd_to_flu,
    ned_quat_to_enu,
    ned_to_enu,
    quat_from_yaw,
    quat_multiply,
    rotate_vector,
)

TOL = 1e-9

FORWARD = Vec3(1.0, 0.0, 0.0)
BODY_UP_FLU = Vec3(0.0, 0.0, 1.0)
BODY_LEFT_FLU = Vec3(0.0, 1.0, 0.0)

ENU_EAST = Vec3(1.0, 0.0, 0.0)
ENU_NORTH = Vec3(0.0, 1.0, 0.0)
ENU_UP = Vec3(0.0, 0.0, 1.0)


def assert_vec(actual: Vec3, expected: Vec3, tol: float = TOL):
    assert actual == pytest.approx(expected, abs=tol), f"{actual} != {expected}"


def ned_attitude(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> Quat:
    """Aerospace ZYX Euler angles -> NED/FRD quaternion (what PX4 reports)."""
    yaw, pitch, roll = (math.radians(a) / 2 for a in (yaw_deg, pitch_deg, roll_deg))
    qz = Quat(math.cos(yaw), 0, 0, math.sin(yaw))
    qy = Quat(math.cos(pitch), 0, math.sin(pitch), 0)
    qx = Quat(math.cos(roll), math.sin(roll), 0, 0)
    return quat_multiply(quat_multiply(qz, qy), qx)


# ---- positions --------------------------------------------------------------


def test_climbing_makes_z_go_up():
    # PX4: 10 m above home is down = -10.
    ground = ned_to_enu(0.0, 0.0, 0.0)
    airborne = ned_to_enu(0.0, 0.0, -10.0)
    assert airborne.z == 10.0
    assert airborne.z > ground.z


def test_north_stays_north_and_east_stays_east():
    assert ned_to_enu(20.0, 0.0, 0.0) == Vec3(0.0, 20.0, 0.0)  # north -> +y
    assert ned_to_enu(0.0, 20.0, 0.0) == Vec3(20.0, 0.0, 0.0)  # east  -> +x


def test_position_all_axes_at_once():
    assert ned_to_enu(north=3.0, east=4.0, down=-5.0) == Vec3(x=4.0, y=3.0, z=5.0)


def test_position_round_trip():
    n, e, d = 12.5, -7.25, -30.0
    assert enu_to_ned(*ned_to_enu(n, e, d)) == Vec3(n, e, d)


def test_distance_is_preserved():
    p = ned_to_enu(3.0, 4.0, -12.0)
    assert math.sqrt(p.x**2 + p.y**2 + p.z**2) == pytest.approx(13.0, abs=TOL)


def test_body_vector_frd_to_flu():
    assert frd_to_flu(1.0, 2.0, 3.0) == Vec3(1.0, -2.0, -3.0)


# ---- attitudes --------------------------------------------------------------


def test_heading_north_stays_north():
    # Identity in NED = level, nose pointing north.
    q = ned_quat_to_enu(Quat(1.0, 0.0, 0.0, 0.0))
    assert_vec(rotate_vector(q, FORWARD), ENU_NORTH)
    assert enu_yaw_deg(q) == pytest.approx(90.0, abs=1e-9)
    assert compass_heading_deg(q) == pytest.approx(0.0, abs=1e-9)


def test_level_drone_has_body_up_pointing_up():
    q = ned_quat_to_enu(Quat(1.0, 0.0, 0.0, 0.0))
    assert_vec(rotate_vector(q, BODY_UP_FLU), ENU_UP)
    # Facing north, the drone's left side points west (-x in ENU).
    assert_vec(rotate_vector(q, BODY_LEFT_FLU), Vec3(-1.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "ned_yaw_deg, expected_forward, expected_compass",
    [
        (0.0, ENU_NORTH, 0.0),
        (90.0, ENU_EAST, 90.0),
        (180.0, Vec3(0.0, -1.0, 0.0), 180.0),
        (270.0, Vec3(-1.0, 0.0, 0.0), 270.0),
        (-90.0, Vec3(-1.0, 0.0, 0.0), 270.0),
    ],
)
def test_compass_headings(ned_yaw_deg, expected_forward, expected_compass):
    q = ned_quat_to_enu(ned_attitude(yaw_deg=ned_yaw_deg))
    assert_vec(rotate_vector(q, FORWARD), expected_forward)
    assert compass_heading_deg(q) == pytest.approx(expected_compass, abs=1e-6)
    # Yaw alone must keep the drone level.
    assert_vec(rotate_vector(q, BODY_UP_FLU), ENU_UP)


def test_northeast_heading_is_45_degrees():
    q = ned_quat_to_enu(ned_attitude(yaw_deg=45.0))
    s = math.sqrt(0.5)
    assert_vec(rotate_vector(q, FORWARD), Vec3(s, s, 0.0))
    assert enu_yaw_deg(q) == pytest.approx(45.0, abs=1e-9)


def test_pitch_up_points_nose_up():
    # Positive pitch in aerospace convention = nose up. Facing north, 30 deg up.
    q = ned_quat_to_enu(ned_attitude(pitch_deg=30.0))
    forward = rotate_vector(q, FORWARD)
    assert forward.z == pytest.approx(math.sin(math.radians(30.0)), abs=TOL)  # 0.5
    assert forward.y == pytest.approx(math.cos(math.radians(30.0)), abs=TOL)
    assert forward.x == pytest.approx(0.0, abs=TOL)


def test_roll_right_drops_the_right_side():
    # Positive roll = right wing down, so the LEFT side rises. Facing north.
    q = ned_quat_to_enu(ned_attitude(roll_deg=20.0))
    left = rotate_vector(q, BODY_LEFT_FLU)
    assert left.z == pytest.approx(math.sin(math.radians(20.0)), abs=TOL)
    assert left.z > 0


def test_conversion_matches_converting_the_vector_directly():
    # For any attitude: rotating a body vector in ENU must equal rotating it in
    # NED and then converting the resulting world vector.
    q_ned = ned_attitude(yaw_deg=73.0, pitch_deg=-12.0, roll_deg=31.0)
    q_enu = ned_quat_to_enu(q_ned)
    body_frd = Vec3(0.3, -1.1, 0.7)

    world_ned = rotate_vector(q_ned, body_frd)
    expected = ned_to_enu(*world_ned)
    actual = rotate_vector(q_enu, frd_to_flu(*body_frd))
    assert_vec(actual, expected)


def test_result_is_unit_length_with_positive_w():
    for yaw in range(-180, 181, 15):
        q = ned_quat_to_enu(ned_attitude(yaw_deg=yaw, pitch_deg=5.0, roll_deg=-8.0))
        assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0, abs=TOL)
        assert q.w >= 0


def test_unnormalized_input_is_normalized():
    q = ned_quat_to_enu(Quat(2.0, 0.0, 0.0, 0.0))
    assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0, abs=TOL)


def test_quat_from_yaw_in_enu_turns_east_toward_north():
    assert_vec(rotate_vector(quat_from_yaw(math.pi / 2), ENU_EAST), ENU_NORTH)
