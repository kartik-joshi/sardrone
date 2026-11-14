"""Time handling for episodes.

Everything inside this project is an *integer number of nanoseconds*. We only
split into (sec, nsec) at the last moment, when filling in a message.

Three clocks show up in a simulated flight:

* wall clock  - Unix time on the Mac (nanoseconds since 1970)
* PX4 clock   - microseconds since the flight controller booted
* sim clock   - Gazebo simulation time, which starts at 0

They have different zero points, so they must never be mixed. `EpisodeClock`
anchors each source once and converts everything to wall-clock nanoseconds.
"""

from typing import NamedTuple

NS_PER_SEC = 1_000_000_000
NS_PER_USEC = 1_000

# 2020-01-01T00:00:00Z and 2100-01-01T00:00:00Z, in Unix seconds.
# Any "wall clock" value outside this window is almost certainly a
# boot-relative or sim-relative time (or the wrong unit) used by mistake.
UNIX_FLOOR_NS = 1_577_836_800 * NS_PER_SEC
UNIX_CEILING_NS = 4_102_444_800 * NS_PER_SEC


class Ts(NamedTuple):
    """A timestamp split the way Foxglove/ROS messages want it."""

    sec: int
    nsec: int  # always 0..999_999_999


def _require_int(value, name: str) -> None:
    # bool is a subclass of int in Python, so exclude it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an int (got {type(value).__name__}). "
            "Floats cannot hold epoch nanoseconds exactly."
        )


def ns_to_ts(ns: int) -> Ts:
    """Split integer nanoseconds into (sec, nsec).

    >>> ns_to_ts(1_500_000_000)
    Ts(sec=1, nsec=500000000)
    """
    _require_int(ns, "ns")
    if ns < 0:
        raise ValueError(f"ns must be >= 0, got {ns}")
    return Ts(sec=ns // NS_PER_SEC, nsec=ns % NS_PER_SEC)


def ts_to_ns(ts: Ts) -> int:
    """Inverse of ns_to_ts."""
    return ts.sec * NS_PER_SEC + ts.nsec


def safe_ts(wall_ns: int) -> Ts:
    """Like ns_to_ts, but for values that are supposed to be wall-clock time.

    Raises ValueError if the value is before 2020 or after 2100, which catches
    boot-relative times, sim times, and seconds/millis/micros passed as nanos.
    """
    _require_int(wall_ns, "wall_ns")
    if wall_ns < UNIX_FLOOR_NS:
        raise ValueError(
            f"{wall_ns} ns is before 2020 - this looks like a boot-relative or "
            "sim-relative time (or the wrong unit), not wall-clock nanoseconds"
        )
    if wall_ns >= UNIX_CEILING_NS:
        raise ValueError(
            f"{wall_ns} ns is after 2100 - check the unit (expected nanoseconds)"
        )
    return ns_to_ts(wall_ns)


def px4_usec_to_ns(usec: int) -> int:
    """PX4 timestamps are microseconds since boot; convert to nanoseconds."""
    _require_int(usec, "usec")
    return usec * NS_PER_USEC


def sim_seconds_to_ns(seconds: float) -> int:
    """Gazebo sim time arrives as float seconds; convert to integer nanoseconds.

    Sim time is small (starts at 0), so a float holds it accurately here.
    """
    return round(seconds * NS_PER_SEC)


class EpisodeClock:
    """Converts readings from any source clock onto one episode timeline.

    Usage:
        clock = EpisodeClock(start_wall_ns=time.time_ns())
        clock.anchor("px4", px4_usec_to_ns(first_px4_timestamp))
        clock.anchor("sim", 0)
        ...
        wall_ns = clock.to_wall_ns("px4", px4_usec_to_ns(sample_timestamp))

    The rule is:  wall = episode_start + (reading - reading_at_episode_start)
    """

    def __init__(self, start_wall_ns: int):
        safe_ts(start_wall_ns)  # validates; raises if not real wall-clock time
        self.start_wall_ns = start_wall_ns
        self._anchors: dict[str, int] = {}

    def anchor(self, source: str, source_ns_at_start: int) -> None:
        """Record what `source` read at the episode start. Allowed once only."""
        _require_int(source_ns_at_start, "source_ns_at_start")
        if source in self._anchors:
            raise ValueError(
                f"clock source {source!r} is already anchored; "
                "re-anchoring mid-episode would make time jump"
            )
        self._anchors[source] = source_ns_at_start

    def elapsed_ns(self, source: str, source_ns: int) -> int:
        """Nanoseconds since episode start, for a reading from `source`."""
        _require_int(source_ns, "source_ns")
        if source not in self._anchors:
            raise KeyError(f"clock source {source!r} was never anchored")
        elapsed = source_ns - self._anchors[source]
        if elapsed < 0:
            raise ValueError(
                f"{source!r} reading {source_ns} is before its anchor "
                f"{self._anchors[source]}; time must not run backwards"
            )
        return elapsed

    def to_wall_ns(self, source: str, source_ns: int) -> int:
        """Wall-clock nanoseconds for a reading from `source`."""
        return self.start_wall_ns + self.elapsed_ns(source, source_ns)

    def to_ts(self, source: str, source_ns: int) -> Ts:
        """Same as to_wall_ns, already split and range-checked."""
        return safe_ts(self.to_wall_ns(source, source_ns))
