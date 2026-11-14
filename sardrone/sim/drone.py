"""The drone, as the recorder sees it.

`Drone` is a small interface: a handful of commands plus `read_state()`. The
recorder is written against this interface only, which gives us two
implementations:

* `MavsdkDrone` (sardrone/sim/mavsdk_drone.py) - the real PX4 SITL vehicle
* `FakeDrone` (below) - a point that moves toward its target at fixed speed,
  with its own simulated clock, so the whole mission can run in a unit test
  in milliseconds with no simulator

All values are in PX4's native conventions (NED, FRD, microseconds since
boot). Converting to Z-up happens later, inside EpisodeWriter.
"""

import math
from dataclasses import dataclass
from typing import Protocol

from sardrone.mcap.frames import Quat, quat_from_yaw
from sardrone.mcap.geo import enu_offset_to_latlon

# PX4 SITL's default home position (Zurich).
DEFAULT_HOME_LAT_DEG = 47.397742
DEFAULT_HOME_LON_DEG = 8.545594
DEFAULT_HOME_ALT_M = 488.0


@dataclass
class DroneState:
    """One snapshot of the vehicle."""

    px4_time_usec: int | None  # microseconds since PX4 boot; None if unavailable
    north_m: float
    east_m: float
    down_m: float
    v_north_m_s: float
    v_east_m_s: float
    v_down_m_s: float
    q_ned_frd: Quat
    latitude_deg: float
    longitude_deg: float
    absolute_altitude_m: float
    battery_percent: float  # 0..100
    battery_voltage_v: float
    flight_mode: str
    in_air: bool

    @property
    def height_m(self) -> float:
        """Height above the local origin (positive up)."""
        return -self.down_m


class Drone(Protocol):
    async def connect(self) -> None: ...

    async def wait_until_ready(self) -> None:
        """Return once the vehicle has a position estimate and can be armed."""
        ...

    async def read_state(self) -> DroneState: ...

    async def arm(self) -> None: ...

    async def takeoff(self, altitude_m: float) -> None:
        """Start climbing to `altitude_m` above the ground. Returns immediately."""
        ...

    async def goto_ned(self, north_m: float, east_m: float, down_m: float, yaw_deg: float) -> None:
        """Set a position target in the local NED frame. Returns immediately."""
        ...

    async def land(self) -> None:
        """Start landing at the current position. Returns immediately."""
        ...

    async def close(self) -> None: ...


class FakeDrone:
    """A deterministic stand-in for PX4: no physics, just a point that flies
    straight toward its target. Time only advances when `sleep()` is called,
    so a 60-second mission runs instantly."""

    CRUISE_SPEED_M_S = 4.0
    BOOT_TIME_USEC = 65_000_000  # pretend PX4 booted 65 s before we connected

    def __init__(self) -> None:
        self.time_usec = self.BOOT_TIME_USEC
        self.position = [0.0, 0.0, 0.0]  # north, east, down
        self.velocity = [0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0]
        self.yaw_deg = 0.0
        self.mode = "READY"
        self.armed = False
        self.battery_percent = 100.0
        self.commands: list[str] = []  # for tests: what was asked, in order

    async def connect(self) -> None:
        self.commands.append("connect")

    async def wait_until_ready(self) -> None:
        self.commands.append("wait_until_ready")

    async def arm(self) -> None:
        self.armed = True
        self.commands.append("arm")

    async def takeoff(self, altitude_m: float) -> None:
        if not self.armed:
            raise RuntimeError("takeoff rejected: not armed")
        self.target = [self.position[0], self.position[1], -altitude_m]
        self.mode = "TAKEOFF"
        self.commands.append(f"takeoff {altitude_m:g}")

    async def goto_ned(self, north_m: float, east_m: float, down_m: float, yaw_deg: float) -> None:
        self.target = [north_m, east_m, down_m]
        self.yaw_deg = yaw_deg
        self.mode = "OFFBOARD"
        self.commands.append(f"goto {north_m:g} {east_m:g} {down_m:g} yaw {yaw_deg:g}")

    async def land(self) -> None:
        self.target = [self.position[0], self.position[1], 0.0]
        self.mode = "LAND"
        self.commands.append("land")

    async def close(self) -> None:
        self.commands.append("close")

    async def sleep(self, seconds: float) -> None:
        """Advance the simulated world by `seconds` (returns immediately)."""
        self.time_usec += int(round(seconds * 1_000_000))
        offset = [t - p for t, p in zip(self.target, self.position)]
        distance = math.sqrt(sum(d * d for d in offset))
        max_step = self.CRUISE_SPEED_M_S * seconds
        if distance <= max_step:
            self.velocity = [d / seconds for d in offset]
            self.position = list(self.target)
        else:
            self.velocity = [d / distance * self.CRUISE_SPEED_M_S for d in offset]
            self.position = [p + v * seconds for p, v in zip(self.position, self.velocity)]
        if self.armed:
            self.battery_percent = max(0.0, self.battery_percent - 0.05 * seconds)
        if self.mode == "TAKEOFF" and distance <= max_step:
            self.mode = "HOLD"
        if self.mode == "LAND" and self.position[2] >= 0.0:
            self.mode = "READY"
            self.armed = False  # PX4 disarms itself after touchdown

    async def read_state(self) -> DroneState:
        north, east, down = self.position
        latitude, longitude = enu_offset_to_latlon(
            DEFAULT_HOME_LAT_DEG, DEFAULT_HOME_LON_DEG, east, north
        )
        return DroneState(
            px4_time_usec=self.time_usec,
            north_m=north,
            east_m=east,
            down_m=down,
            v_north_m_s=self.velocity[0],
            v_east_m_s=self.velocity[1],
            v_down_m_s=self.velocity[2],
            q_ned_frd=quat_from_yaw(math.radians(self.yaw_deg)),
            latitude_deg=latitude,
            longitude_deg=longitude,
            absolute_altitude_m=DEFAULT_HOME_ALT_M - down,
            battery_percent=self.battery_percent,
            battery_voltage_v=12.6 + 4.2 * self.battery_percent / 100.0,
            flight_mode=self.mode,
            in_air=down < -0.05,
        )
