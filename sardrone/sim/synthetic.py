"""Write a fake episode with no simulator running.

The drone flies one circle at constant height. Positions are generated in
PX4's own conventions (NED, boot-relative microseconds) on purpose, so the
episode exercises the same clock anchoring and frame conversion as a real
flight.

    python -m sardrone.sim.synthetic            # -> data/episodes/synthetic-0001.mcap
"""

import argparse
import math
import time
from pathlib import Path

from sardrone.mcap.episode_writer import (
    EVENT_LAND,
    EVENT_PERSON_SEEN,
    EVENT_TAKEOFF,
    EVENT_WAYPOINT_REACHED,
    EpisodeMeta,
    EpisodeWriter,
)
from sardrone.mcap.frames import quat_from_yaw
from sardrone.mcap.geo import enu_offset_to_latlon
from sardrone.mcap.timeutil import EpisodeClock, px4_usec_to_ns
from sardrone.sim.camera import SyntheticCamera, pinhole_intrinsics

# PX4 SITL's default home position (Zurich).
HOME_LAT_DEG = 47.397742
HOME_LON_DEG = 8.545594
HOME_ALT_M = 488.0

RATE_HZ = 10
RADIUS_M = 20.0
HEIGHT_M = 10.0
FAKE_PX4_BOOT_USEC = 65_000_000  # pretend PX4 booted 65 s before the episode


def circle_state_ned(t_s: float, duration_s: float):
    """Position, velocity and yaw on the circle at time t, in NED.

    Starts due north of home and moves clockwise (north -> east), nose
    pointing along the direction of travel.
    """
    omega = 2.0 * math.pi / duration_s  # one lap per episode
    angle = omega * t_s
    north = RADIUS_M * math.cos(angle)
    east = RADIUS_M * math.sin(angle)
    down = -HEIGHT_M
    v_north = -RADIUS_M * omega * math.sin(angle)
    v_east = RADIUS_M * omega * math.cos(angle)
    yaw_ned = math.atan2(v_east, v_north)  # NED yaw: 0 = north, +90 deg = east
    return (north, east, down), (v_north, v_east, 0.0), yaw_ned


def write_synthetic_episode(
    path,
    *,
    duration_s: float = 10.0,
    start_wall_ns: int | None = None,
    overwrite: bool = False,
) -> EpisodeWriter:
    """Write the episode and return the (closed) writer, for its counts and paths."""
    start_wall_ns = time.time_ns() if start_wall_ns is None else start_wall_ns
    clock = EpisodeClock(start_wall_ns)
    clock.anchor("px4", px4_usec_to_ns(FAKE_PX4_BOOT_USEC))

    camera = SyntheticCamera()
    meta = EpisodeMeta(
        world="synthetic_circle",
        weather="clear",
        time_of_day="noon",
        mission_prompt="Fly one circle and report anyone you see.",
        people_placed=1,
    )

    n_steps = int(round(duration_s * RATE_HZ))
    person_step = int(n_steps * 0.6)
    quarter = n_steps // 4

    with EpisodeWriter(path, meta, overwrite=overwrite) as episode:
        fx, fy, cx, cy = pinhole_intrinsics(camera.width, camera.height, camera.hfov_deg)
        episode.log_calibration(start_wall_ns, camera.width, camera.height, fx, fy, cx, cy)

        for step in range(n_steps + 1):
            t_s = step / RATE_HZ
            px4_usec = FAKE_PX4_BOOT_USEC + step * (1_000_000 // RATE_HZ)
            wall_ns = clock.to_wall_ns("px4", px4_usec_to_ns(px4_usec))

            (north, east, down), velocity_ned, yaw_ned = circle_state_ned(t_s, duration_s)

            episode.log_pose_ned(wall_ns, north, east, down, quat_from_yaw(yaw_ned))

            latitude, longitude = enu_offset_to_latlon(HOME_LAT_DEG, HOME_LON_DEG, east, north)
            episode.log_gps(wall_ns, latitude, longitude, HOME_ALT_M + HEIGHT_M)

            episode.log_telemetry(
                wall_ns,
                battery_percent=100.0 - 0.5 * t_s,
                battery_voltage_v=16.2 - 0.02 * t_s,
                flight_mode="OFFBOARD",
                velocity_ned_m_s=velocity_ned,
            )

            camera.east_m, camera.north_m = east, north
            camera.person_visible = person_step <= step < person_step + RATE_HZ
            camera.caption = f"t={t_s:4.1f}s"
            episode.log_image(wall_ns, camera.read())

            if step == 0:
                episode.log_event(wall_ns, EVENT_TAKEOFF)
            elif step == person_step:
                episode.log_event(wall_ns, EVENT_PERSON_SEEN, "1 person, centre of frame")
                episode.meta.people_found += 1
            elif step == n_steps:
                episode.log_event(wall_ns, EVENT_LAND)
            elif quarter and step % quarter == 0:
                episode.log_event(wall_ns, EVENT_WAYPOINT_REACHED, f"{step // quarter}/4")

        episode.meta.success = episode.meta.people_found == episode.meta.people_placed

    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/episodes/synthetic-0001.mcap"))
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    args = parser.parse_args()

    episode = write_synthetic_episode(args.out, duration_s=args.duration, overwrite=True)
    print(f"wrote {episode.path} ({episode.path.stat().st_size / 1024:.0f} KiB)")
    print(f"wrote {episode.sidecar_path}")
    for topic, count in sorted(episode.message_counts.items()):
        print(f"  {topic:28s} {count:5d} messages")


if __name__ == "__main__":
    main()
