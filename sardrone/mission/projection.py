"""From a point in the world to a pixel in the image, and back.

A pinhole camera maps a 3D point to a pixel in two steps:

1. Express the point in the CAMERA's own axes (x right, y down, z forward).
   That needs the drone's pose and how the camera is mounted on it.
2. Divide by depth: u = fx * x / z + cx,  v = fy * y / z + cy.

Going back, one pixel only gives a direction (a ray), not a distance. To get
a position we intersect that ray with something known: here, flat ground.
"""

import math
from dataclasses import dataclass

from sardrone.mcap.episode_writer import camera_in_body
from sardrone.mcap.frames import (
    Quat,
    Vec3,
    ned_quat_to_enu,
    ned_to_enu,
    quat_conjugate,
    quat_multiply,
    rotate_vector,
)
from sardrone.sim.camera import pinhole_intrinsics


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    mount_m: tuple[float, float, float]  # in the body frame: forward, left, up
    pitch_deg: float  # 0 = straight ahead, 90 = straight down

    @classmethod
    def from_source(cls, camera) -> "CameraModel":
        """Build from anything with width, height, hfov_deg, mount_m, pitch_deg."""
        fx, fy, cx, cy = pinhole_intrinsics(camera.width, camera.height, camera.hfov_deg)
        return cls(camera.width, camera.height, fx, fy, cx, cy, tuple(camera.mount_m), camera.pitch_deg)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2 / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2 / self.fy))


@dataclass(frozen=True)
class DronePose:
    """Drone body pose in the `map` frame (ENU, z up; body x forward, y left)."""

    position: Vec3
    orientation: Quat

    @classmethod
    def from_state(cls, state) -> "DronePose":
        """From a DroneState, which is in PX4's NED / FRD conventions."""
        return cls(
            ned_to_enu(state.north_m, state.east_m, state.down_m),
            ned_quat_to_enu(state.q_ned_frd),
        )


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def inside(self, width: int, height: int) -> bool:
        return self.x0 >= 0 and self.y0 >= 0 and self.x1 <= width - 1 and self.y1 <= height - 1


def camera_pose(model: CameraModel, pose: DronePose) -> tuple[Vec3, Quat]:
    """Position of the camera in `map`, and the rotation optical -> map."""
    offset = rotate_vector(pose.orientation, Vec3(*model.mount_m))
    position = Vec3(*(p + o for p, o in zip(pose.position, offset)))
    return position, quat_multiply(pose.orientation, camera_in_body(model.pitch_deg))


def project(model: CameraModel, pose: DronePose, point) -> tuple[float, float, float] | None:
    """Pixel (u, v) and depth of a `map` point, or None if it is behind the camera."""
    position, q_cam = camera_pose(model, pose)
    relative = Vec3(*(p - c for p, c in zip(point, position)))
    in_camera = rotate_vector(quat_conjugate(q_cam), relative)
    if in_camera.z <= 0.1:
        return None
    u = model.fx * in_camera.x / in_camera.z + model.cx
    v = model.fy * in_camera.y / in_camera.z + model.cy
    return u, v, in_camera.z


def project_box(model: CameraModel, pose: DronePose, corners) -> BBox | None:
    """Smallest image rectangle around the projected 3D corners (None if any
    corner is behind the camera)."""
    pixels = [project(model, pose, corner) for corner in corners]
    if any(pixel is None for pixel in pixels):
        return None
    us, vs = [p[0] for p in pixels], [p[1] for p in pixels]
    return BBox(min(us), min(vs), max(us), max(vs))


def pixel_to_ground(
    model: CameraModel, pose: DronePose, u: float, v: float, ground_z: float = 0.0
) -> tuple[float, float] | None:
    """Where the ray through pixel (u, v) meets flat ground at height
    `ground_z`: (east, north) in `map`. None if the ray points at the sky."""
    position, q_cam = camera_pose(model, pose)
    ray = rotate_vector(q_cam, Vec3((u - model.cx) / model.fx, (v - model.cy) / model.fy, 1.0))
    if ray.z >= -1e-9:
        return None
    scale = (ground_z - position.z) / ray.z
    return position.x + scale * ray.x, position.y + scale * ray.y
