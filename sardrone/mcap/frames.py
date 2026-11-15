"""Coordinate-frame conversions: PX4 (NED / FRD) -> Foxglove & FiftyOne (ENU / FLU).

PX4 uses aerospace conventions:
    world frame  NED = x North,   y East,  z Down
    body frame   FRD = x Forward, y Right, z Down

Foxglove, ROS and FiftyOne's 3D views use robotics conventions (Z up):
    world frame  ENU = x East,    y North, z Up
    body frame   FLU = x Forward, y Left,  z Up

Positions:  swap x and y, negate z.
Attitudes:  a quaternion describes "body frame relative to world frame", so we
            have to change BOTH frames:

                q_enu_flu = Q_NED_TO_ENU * q_ned_frd * Q_FRD_TO_FLU

Quaternions here are plain (w, x, y, z) tuples, Hamilton convention, the same
order MAVSDK reports them in.
"""

import math
from typing import NamedTuple


class Vec3(NamedTuple):
    x: float
    y: float
    z: float


class Quat(NamedTuple):
    w: float
    x: float
    y: float
    z: float


_HALF_SQRT2 = math.sqrt(0.5)

# NED -> ENU is a 180 degree turn about the diagonal axis (1, 1, 0)/sqrt(2):
# it swaps x<->y and flips z.
Q_NED_TO_ENU = Quat(0.0, _HALF_SQRT2, _HALF_SQRT2, 0.0)

# FRD -> FLU is a 180 degree turn about the body's forward (x) axis:
# forward stays forward, right becomes left, down becomes up.
Q_FRD_TO_FLU = Quat(0.0, 1.0, 0.0, 0.0)


# ---- positions / vectors ----------------------------------------------------


def ned_to_enu(north: float, east: float, down: float) -> Vec3:
    """World-frame position or velocity: NED -> ENU."""
    return Vec3(x=east, y=north, z=-down)


def enu_to_ned(x: float, y: float, z: float) -> Vec3:
    """Inverse of ned_to_enu. Returns (north, east, down) in a Vec3."""
    return Vec3(x=y, y=x, z=-z)


def frd_to_flu(forward: float, right: float, down: float) -> Vec3:
    """Body-frame vector: FRD -> FLU."""
    return Vec3(x=forward, y=-right, z=-down)


# ---- quaternion helpers -----------------------------------------------------


def quat_multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product a*b: apply rotation b first, then a."""
    return Quat(
        w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    )


def quat_conjugate(q: Quat) -> Quat:
    """For a unit quaternion, the conjugate is the inverse rotation."""
    return Quat(q.w, -q.x, -q.y, -q.z)


def quat_normalize(q: Quat) -> Quat:
    norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    if norm < 1e-9:
        raise ValueError("cannot normalize a zero quaternion")
    return Quat(q.w / norm, q.x / norm, q.y / norm, q.z / norm)


def rotate_vector(q: Quat, v: Vec3) -> Vec3:
    """Rotate v by unit quaternion q (body-frame vector -> world-frame vector)."""
    p = Quat(0.0, v.x, v.y, v.z)
    r = quat_multiply(quat_multiply(q, p), quat_conjugate(q))
    return Vec3(r.x, r.y, r.z)


def quat_from_yaw(yaw_rad: float) -> Quat:
    """Rotation by `yaw_rad` about the world z axis (right-hand rule).

    In NED (z down) positive yaw turns clockwise seen from above: north -> east.
    In ENU (z up) positive yaw turns counter-clockwise: east -> north.
    """
    return Quat(math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2))


def quat_from_rpy(roll_rad: float, pitch_rad: float, yaw_rad: float) -> Quat:
    """Roll, pitch, yaw (the order SDF uses in <pose>) as one quaternion.

    The rotations are applied about the FIXED axes x, then y, then z, which
    is the same as R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    q_roll = Quat(math.cos(roll_rad / 2), math.sin(roll_rad / 2), 0.0, 0.0)
    q_pitch = Quat(math.cos(pitch_rad / 2), 0.0, math.sin(pitch_rad / 2), 0.0)
    return quat_multiply(quat_from_yaw(yaw_rad), quat_multiply(q_pitch, q_roll))


# ---- attitude conversion ----------------------------------------------------


def ned_quat_to_enu(q_ned_frd: Quat) -> Quat:
    """Attitude quaternion: (FRD body in NED world) -> (FLU body in ENU world).

    The result is normalized and has w >= 0 (q and -q are the same rotation;
    picking one sign keeps plots from jumping).
    """
    q = quat_multiply(quat_multiply(Q_NED_TO_ENU, quat_normalize(q_ned_frd)), Q_FRD_TO_FLU)
    q = quat_normalize(q)
    if q.w < 0:
        q = Quat(-q.w, -q.x, -q.y, -q.z)
    return q


def enu_yaw_deg(q_enu_flu: Quat) -> float:
    """ENU yaw in degrees: 0 = facing East, 90 = facing North (counter-clockwise)."""
    forward = rotate_vector(q_enu_flu, Vec3(1.0, 0.0, 0.0))
    return math.degrees(math.atan2(forward.y, forward.x))


def compass_heading_deg(q_enu_flu: Quat) -> float:
    """Compass heading in degrees: 0 = North, 90 = East (clockwise), in [0, 360)."""
    return (90.0 - enu_yaw_deg(q_enu_flu)) % 360.0
