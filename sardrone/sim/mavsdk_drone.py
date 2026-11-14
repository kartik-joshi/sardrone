"""The real drone: PX4 SITL reached through MAVSDK.

API checked against the INSTALLED mavsdk 4.2.0 by reading its source. v4 is
the new "native" binding and differs from the older gRPC package that most
tutorials show (`from mavsdk import System; drone.telemetry.position()`):

* connect:  `Mavsdk(Configuration...)`, `add_any_connection(url)`,
            `first_autopilot(timeout_s)`
* plugins are separate classes built from the system:
            `TelemetryAsync(system)`, `ActionAsync(system)`, `OffboardAsync(system)`
* every telemetry stream also has a "give me the latest value" getter
            (`await telemetry.position()`), which is what we use: the recorder
            samples everything together at a fixed rate

Flown against PX4 SITL v1.18 + Gazebo Harmonic (gz_x500). The automated
tests use FakeDrone instead, because they must run without a simulator.
"""

import asyncio

from mavsdk import ComponentType, Configuration
from mavsdk.asyncio import Mavsdk
from mavsdk.asyncio.plugins.action import ActionAsync
from mavsdk.asyncio.plugins.offboard import OffboardAsync
from mavsdk.asyncio.plugins.telemetry import TelemetryAsync
from mavsdk.plugins.offboard import PositionNedYaw

from sardrone.mcap.frames import Quat
from sardrone.sim.drone import DroneState

TELEMETRY_RATE_HZ = 20.0  # ask PX4 for 2x our 10 Hz logging rate


class DroneConnectionError(RuntimeError):
    pass


class MavsdkDrone:
    def __init__(self, url: str = "udpin://0.0.0.0:14540", connect_timeout_s: float = 20.0):
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self._mavsdk: Mavsdk | None = None
        self._telemetry: TelemetryAsync | None = None
        self._action: ActionAsync | None = None
        self._offboard: OffboardAsync | None = None
        self._offboard_started = False

    async def connect(self) -> None:
        configuration = Configuration.create_with_component_type(ComponentType.GROUND_STATION)
        self._mavsdk = Mavsdk(configuration)
        await self._mavsdk.add_any_connection(self.url)
        system = await self._mavsdk.first_autopilot(timeout_s=self.connect_timeout_s)
        if system is None:
            raise DroneConnectionError(
                f"no autopilot answered on {self.url} within {self.connect_timeout_s:g} s. "
                "Is PX4 SITL running?  (cd PX4-Autopilot && make px4_sitl gz_x500)"
            )
        self._telemetry = TelemetryAsync(system)
        self._action = ActionAsync(system)
        self._offboard = OffboardAsync(system)

        # Rate requests are best-effort: a refusal only means slower updates.
        for set_rate in (
            self._telemetry.set_rate_position_velocity_ned,
            self._telemetry.set_rate_attitude_quaternion,
            self._telemetry.set_rate_position,
        ):
            try:
                await set_rate(TELEMETRY_RATE_HZ)
            except Exception as error:  # noqa: BLE001 - report and carry on
                print(f"warning: {set_rate.__name__} failed: {error}")

    async def wait_until_ready(self, timeout_s: float = 90.0, settle_s: float = 5.0) -> None:
        """Wait until PX4 allows arming AND its height estimate has settled.

        Right after boot PX4's estimator is still converging: in the first
        real flight it reported 1.9 m of height while sitting on the ground,
        which put the whole flight 2 m too high. So we also require the
        local height to stay within 10 cm for `settle_s` seconds.
        """
        poll_s = 0.5
        waited = 0.0
        stable_since = 0.0
        reference_down = None
        while waited < timeout_s:
            health = await self._telemetry.health()
            healthy = (
                health.is_global_position_ok and health.is_home_position_ok and health.is_armable
            )
            down = (await self._telemetry.position_velocity_ned()).position.down_m
            if not healthy or reference_down is None or abs(down - reference_down) > 0.10:
                reference_down, stable_since = down, waited  # restart the settle timer
            elif waited - stable_since >= settle_s:
                return
            await asyncio.sleep(poll_s)
            waited += poll_s
        raise DroneConnectionError(f"vehicle not ready and settled after {timeout_s:g} s")

    async def read_state(self) -> DroneState:
        position_velocity = await self._telemetry.position_velocity_ned()
        attitude = await self._telemetry.attitude_quaternion()
        global_position = await self._telemetry.position()
        battery = await self._telemetry.battery()
        flight_mode = await self._telemetry.flight_mode()
        in_air = await self._telemetry.in_air()

        ned = position_velocity.position
        velocity = position_velocity.velocity
        # remaining_percent is on a 0..100 scale (confirmed on a real PX4 flight).
        return DroneState(
            px4_time_usec=attitude.timestamp_us or None,
            north_m=ned.north_m,
            east_m=ned.east_m,
            down_m=ned.down_m,
            v_north_m_s=velocity.north_m_s,
            v_east_m_s=velocity.east_m_s,
            v_down_m_s=velocity.down_m_s,
            q_ned_frd=Quat(attitude.w, attitude.x, attitude.y, attitude.z),
            latitude_deg=global_position.latitude_deg,
            longitude_deg=global_position.longitude_deg,
            absolute_altitude_m=global_position.absolute_altitude_m,
            battery_percent=battery.remaining_percent,
            battery_voltage_v=battery.voltage_v,
            flight_mode=flight_mode.name,
            in_air=bool(in_air),
        )

    async def arm(self) -> None:
        await self._action.arm()

    async def takeoff(self, altitude_m: float) -> None:
        await self._action.set_takeoff_altitude(altitude_m)
        await self._action.takeoff()

    async def goto_ned(self, north_m: float, east_m: float, down_m: float, yaw_deg: float) -> None:
        # PX4 only enters offboard mode if a setpoint was sent first.
        await self._offboard.set_position_ned(PositionNedYaw(north_m, east_m, down_m, yaw_deg))
        if not self._offboard_started:
            await self._offboard.start()
            self._offboard_started = True

    async def land(self) -> None:
        if self._offboard_started:
            try:
                await self._offboard.stop()
            except Exception as error:  # noqa: BLE001 - landing matters more
                print(f"warning: offboard stop failed: {error}")
            self._offboard_started = False
        await self._action.land()

    async def close(self) -> None:
        if self._mavsdk is not None:
            self._mavsdk.destroy()
            self._mavsdk = None
