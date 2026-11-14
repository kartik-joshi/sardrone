import pytest

from sardrone.mcap.timeutil import (
    NS_PER_SEC,
    UNIX_CEILING_NS,
    UNIX_FLOOR_NS,
    EpisodeClock,
    Ts,
    ns_to_ts,
    px4_usec_to_ns,
    safe_ts,
    sim_seconds_to_ns,
    ts_to_ns,
)

# 2026-11-14T00:00:00Z, a realistic episode start.
START_NS = 1_794_614_400 * NS_PER_SEC


# ---- ns_to_ts ---------------------------------------------------------------


@pytest.mark.parametrize(
    "ns, expected",
    [
        (0, Ts(0, 0)),
        (1, Ts(0, 1)),
        (999_999_999, Ts(0, 999_999_999)),
        (1_000_000_000, Ts(1, 0)),
        (1_500_000_000, Ts(1, 500_000_000)),
        (START_NS + 123_456_789, Ts(1_794_614_400, 123_456_789)),
    ],
)
def test_ns_to_ts_splits_exactly(ns, expected):
    assert ns_to_ts(ns) == expected


def test_nsec_is_never_the_full_epoch_value():
    ts = ns_to_ts(START_NS + 42)
    assert 0 <= ts.nsec <= 999_999_999
    assert ts.nsec == 42
    assert ts.nsec != START_NS + 42


def test_round_trip_is_lossless():
    for ns in (0, 1, 999_999_999, NS_PER_SEC, START_NS + 987_654_321):
        assert ts_to_ns(ns_to_ts(ns)) == ns


def test_ns_to_ts_rejects_negative():
    with pytest.raises(ValueError):
        ns_to_ts(-1)


@pytest.mark.parametrize("bad", [1.5e18, "123", None, True])
def test_ns_to_ts_rejects_non_int(bad):
    with pytest.raises(TypeError):
        ns_to_ts(bad)


def test_why_floats_are_rejected():
    # A float cannot represent this epoch-nanosecond value: it drifts by 1 ns+.
    ns = START_NS + 123_456_789
    assert int(float(ns)) != ns


# ---- safe_ts ----------------------------------------------------------------


def test_safe_ts_accepts_real_wall_time():
    assert safe_ts(START_NS) == Ts(1_794_614_400, 0)
    assert safe_ts(UNIX_FLOOR_NS) == Ts(1_577_836_800, 0)


@pytest.mark.parametrize(
    "bad_ns",
    [
        0,  # sim time at start
        px4_usec_to_ns(65_000_000),  # PX4: 65 s since boot
        1_794_614_400,  # wall time in SECONDS passed as ns
        1_794_614_400_000,  # ... in milliseconds
        1_794_614_400_000_000,  # ... in microseconds
        UNIX_FLOOR_NS - 1,
    ],
)
def test_safe_ts_rejects_non_wall_time(bad_ns):
    with pytest.raises(ValueError, match="before 2020"):
        safe_ts(bad_ns)


def test_safe_ts_rejects_far_future():
    with pytest.raises(ValueError, match="after 2100"):
        safe_ts(UNIX_CEILING_NS)


# ---- unit helpers -----------------------------------------------------------


def test_px4_usec_to_ns():
    assert px4_usec_to_ns(1) == 1_000
    assert px4_usec_to_ns(65_000_000) == 65 * NS_PER_SEC


def test_sim_seconds_to_ns():
    assert sim_seconds_to_ns(0.0) == 0
    assert sim_seconds_to_ns(0.004) == 4_000_000  # one 250 Hz physics step
    assert sim_seconds_to_ns(12.5) == 12_500_000_000


# ---- EpisodeClock -----------------------------------------------------------


def make_clock() -> EpisodeClock:
    clock = EpisodeClock(START_NS)
    clock.anchor("px4", px4_usec_to_ns(65_000_000))  # PX4 booted 65 s earlier
    clock.anchor("sim", sim_seconds_to_ns(3.0))  # Gazebo had run for 3 s
    return clock


def test_anchor_reading_maps_to_episode_start():
    clock = make_clock()
    assert clock.to_wall_ns("px4", px4_usec_to_ns(65_000_000)) == START_NS
    assert clock.to_wall_ns("sim", sim_seconds_to_ns(3.0)) == START_NS


def test_two_sources_agree_on_the_same_instant():
    # 2.5 s into the episode, as seen by each clock.
    clock = make_clock()
    from_px4 = clock.to_wall_ns("px4", px4_usec_to_ns(67_500_000))
    from_sim = clock.to_wall_ns("sim", sim_seconds_to_ns(5.5))
    assert from_px4 == from_sim == START_NS + 2_500_000_000
    assert clock.elapsed_ns("px4", px4_usec_to_ns(67_500_000)) == 2_500_000_000


def test_to_ts_is_split_and_in_range():
    clock = make_clock()
    ts = clock.to_ts("px4", px4_usec_to_ns(65_000_001))  # 1 microsecond in
    assert ts == Ts(1_794_614_400, 1_000)


def test_clock_start_must_be_wall_time():
    with pytest.raises(ValueError):
        EpisodeClock(0)


def test_cannot_anchor_twice():
    clock = make_clock()
    with pytest.raises(ValueError, match="already anchored"):
        clock.anchor("px4", 0)


def test_unanchored_source_is_an_error():
    clock = EpisodeClock(START_NS)
    with pytest.raises(KeyError):
        clock.to_wall_ns("px4", 123)


def test_time_before_anchor_is_an_error():
    clock = make_clock()
    with pytest.raises(ValueError, match="backwards"):
        clock.to_wall_ns("px4", px4_usec_to_ns(64_999_999))
