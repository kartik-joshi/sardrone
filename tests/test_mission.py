"""Phase 2B: camera footprint, lawnmower coverage, oracle, and the search
state machine flown by FakeDrone on simulated time."""

import asyncio
import json
import math

import pytest

from sardrone.mcap.episode_writer import TOPIC_EVENTS, TOPIC_POSE, EpisodeMeta
from sardrone.mcap.frames import Quat, Vec3, quat_from_yaw
from sardrone.mission.coverage import covered_fraction, lawnmower, point_in_polygon
from sardrone.mission.footprint import footprint, person_pixels
from sardrone.mission.oracle import OracleDetector, Sighting, box_corners
from sardrone.mission.projection import (
    BBox,
    CameraModel,
    DronePose,
    pixel_to_ground,
    project,
    project_box,
)
from sardrone.mission.search import (
    ALLOWED,
    SearchConfig,
    bearing_deg,
    run_search,
    safe_confirm_altitude,
    square_polygon,
)
from sardrone.sim.camera import SyntheticCamera, pinhole_intrinsics
from sardrone.sim.drone import FakeDrone
from sardrone.sim.record_flight import people_in_map
from sardrone.worlds.layout import build_world
from sardrone.worlds.matrix import VariationRow
from tests.conftest import START_NS, Episode

S = math.sqrt(0.5)


def gazebo_model(pitch_deg=45.0) -> CameraModel:
    """The simulated camera: 1280x960, 99.7 deg wide."""
    fx, fy, cx, cy = pinhole_intrinsics(1280, 960, 99.7)
    return CameraModel(1280, 960, fx, fy, cx, cy, (0.0, 0.0, 0.0), pitch_deg)


def pose_at(east, north, up, heading_deg=0.0) -> DronePose:
    """Level drone at a map position; heading is a compass bearing (0 = north)."""
    # ENU yaw is counter-clockwise from east, so north (0 deg compass) is +90 deg.
    return DronePose(Vec3(east, north, up), quat_from_yaw(math.radians(90.0 - heading_deg)))


# ---- projection --------------------------------------------------------------


def test_image_centre_looks_ahead_by_height_over_tan_pitch():
    model = gazebo_model(45.0)
    pose = pose_at(0, 0, 15, heading_deg=0)  # facing north
    # 45 deg down from 15 m: the centre ray meets the ground 15 m to the north.
    u, v, depth = project(model, pose, (0.0, 15.0, 0.0))
    assert (u, v) == pytest.approx((640.0, 480.0), abs=1e-6)
    assert depth == pytest.approx(15 / S)  # slant range 21.2 m
    east, north = pixel_to_ground(model, pose, 640.0, 480.0)
    assert (east, north) == pytest.approx((0.0, 15.0), abs=1e-9)


def test_image_directions():
    model = gazebo_model(45.0)
    pose = pose_at(0, 0, 15, heading_deg=0)
    centre = project(model, pose, (0.0, 15.0, 0.0))
    # Facing north, east is to the RIGHT: larger u.
    assert project(model, pose, (5.0, 15.0, 0.0))[0] > centre[0]
    # Farther away is HIGHER in the image: smaller v.
    assert project(model, pose, (0.0, 30.0, 0.0))[1] < centre[1]
    # Something taller than the ground at the same spot is also higher.
    assert project(model, pose, (0.0, 15.0, 1.8))[1] < centre[1]
    # Behind the drone: not projectable.
    assert project(model, pose, (0.0, -30.0, 0.0)) is None


def test_heading_turns_the_view():
    model = gazebo_model(45.0)
    east_facing = pose_at(0, 0, 15, heading_deg=90)
    u, v, _ = project(model, east_facing, (15.0, 0.0, 0.0))
    assert (u, v) == pytest.approx((640.0, 480.0), abs=1e-6)
    # Facing east, north is to the LEFT.
    assert project(model, east_facing, (15.0, 5.0, 0.0))[0] < 640.0


@pytest.mark.parametrize("point", [(3.0, 12.0), (-8.0, 25.0), (20.0, 40.0), (0.5, 2.0)])
def test_pixel_to_ground_inverts_project(point):
    model = gazebo_model(45.0)
    pose = pose_at(1.0, -2.0, 15.0, heading_deg=20.0)
    u, v, _ = project(model, pose, (*point, -0.227))
    assert pixel_to_ground(model, pose, u, v, ground_z=-0.227) == pytest.approx(point, abs=1e-6)


def test_pixel_above_the_horizon_has_no_ground_point():
    model = gazebo_model(20.0)  # top of the image is above the horizon
    assert pixel_to_ground(model, pose_at(0, 0, 15), 640.0, 0.0) is None


def test_mount_offset_moves_the_camera():
    fx, fy, cx, cy = pinhole_intrinsics(1280, 960, 99.7)
    model = CameraModel(1280, 960, fx, fy, cx, cy, (0.16, 0.0, -0.04), 45.0)
    # The camera is 0.16 m ahead and 0.04 m below the body, so its centre ray
    # lands 0.16 - 0.04 = 0.12 m farther north than without the offset.
    east, north = pixel_to_ground(model, pose_at(0, 0, 15), 640.0, 480.0)
    assert (east, north) == pytest.approx((0.0, 15.12), abs=1e-9)


# ---- footprint ---------------------------------------------------------------


def test_footprint_numbers_at_15_m():
    view = footprint(gazebo_model(45.0), 15.0)
    assert view.centre_ground_m == pytest.approx(15.0)
    assert view.centre_slant_m == pytest.approx(21.213, abs=1e-3)  # 15 / sin(45)
    # Half-width = slant * tan(hfov / 2) = 21.213 * tan(49.85 deg).
    assert view.half_width_m == pytest.approx(21.213 * math.tan(math.radians(49.85)), abs=1e-3)
    assert view.half_width_m == pytest.approx(25.15, abs=0.01)
    assert view.limited_by == "field_of_view"
    # Bottom of the image looks almost straight down; the top nearly at the horizon.
    assert 0.0 < view.near_ground_m < 1.0
    assert view.far_ground_m > 200.0
    # A 1.7 m person in the middle: 540 px focal length * 1.7 / 21.2 m = 43 px.
    assert view.person_px_centre == pytest.approx(43.3, abs=0.1)
    assert view.person_px_edge < view.person_px_centre


def test_higher_is_wider_but_people_get_smaller():
    low, mid = footprint(gazebo_model(), 15.0), footprint(gazebo_model(), 25.0)
    assert mid.half_width_m > low.half_width_m
    assert mid.person_px_centre == pytest.approx(low.person_px_centre * 15 / 25)


def test_at_40_m_resolution_not_field_of_view_limits_the_swath():
    view = footprint(gazebo_model(), 40.0, min_person_px=12.0)
    assert view.limited_by == "resolution"
    assert view.person_px_edge == pytest.approx(12.0, abs=1e-6)  # exactly the limit at the edge
    assert view.person_px_centre == pytest.approx(16.2, abs=0.1)


def test_too_high_for_the_required_pixels_is_refused():
    with pytest.raises(ValueError, match="Fly lower"):
        footprint(gazebo_model(), 40.0, min_person_px=20.0)
    with pytest.raises(ValueError, match="horizon"):
        footprint(gazebo_model(pitch_deg=5.0), 15.0)


def test_nadir_camera_looks_straight_down():
    view = footprint(gazebo_model(pitch_deg=90.0), 20.0)
    assert view.centre_ground_m == pytest.approx(0.0, abs=1e-9)
    assert view.centre_slant_m == pytest.approx(20.0)
    assert person_pixels(gazebo_model(), 20.0) == pytest.approx(540 * 1.7 / 20, rel=1e-3)


# ---- coverage ----------------------------------------------------------------

SQUARE_120 = square_polygon(120.0)


def test_lawnmower_on_a_square_at_15_m():
    view = footprint(gazebo_model(), 15.0)
    plan = lawnmower(SQUARE_120, view, overlap=0.25)
    # Swath 50.3 m, 25 % overlap -> at most 37.7 m between lanes -> 120 / 37.7 = 3.2 -> 4 lanes.
    assert plan.lanes == 4
    assert plan.spacing_m == pytest.approx(30.0)
    assert plan.overlap >= 0.25 and plan.overlap == pytest.approx(1 - 30 / 50.3, abs=0.01)
    assert plan.lead_in_m == pytest.approx(15.0)
    assert len(plan.waypoints) == 8
    assert [w.kind for w in plan.waypoints[:4]] == ["lane_start", "lane_end"] * 2
    assert covered_fraction(SQUARE_120, plan) == 1.0


def test_lanes_alternate_direction_and_are_evenly_spaced():
    plan = lawnmower(SQUARE_120, footprint(gazebo_model(), 15.0), heading_deg=0.0)
    lanes = list(zip(plan.waypoints[0::2], plan.waypoints[1::2]))
    directions = [math.copysign(1, end.north_m - begin.north_m) for begin, end in lanes]
    assert directions == [1, -1, 1, -1] or directions == [-1, 1, -1, 1]
    easts = [begin.east_m for begin, _ in lanes]
    assert all(begin.east_m == end.east_m for begin, end in lanes)  # lanes run north-south
    gaps = [abs(b - a) for a, b in zip(easts, easts[1:])]
    assert gaps == pytest.approx([30.0, 30.0, 30.0])
    assert min(abs(e) for e in easts) == pytest.approx(15.0)  # centred: -45, -15, 15, 45
    # Each lane starts 15 m before the area's edge (the lead-in) and ends on the far edge.
    for begin, end in lanes:
        assert abs(begin.north_m) == pytest.approx(75.0) and abs(end.north_m) == pytest.approx(60.0)


def test_more_overlap_means_more_lanes_and_a_longer_path():
    view = footprint(gazebo_model(), 15.0)
    loose, tight = lawnmower(SQUARE_120, view, overlap=0.2), lawnmower(SQUARE_120, view, overlap=0.5)
    assert tight.lanes > loose.lanes
    assert tight.path_length_m() > loose.path_length_m()


def test_flying_higher_needs_fewer_lanes():
    low = lawnmower(SQUARE_120, footprint(gazebo_model(), 15.0))
    high = lawnmower(SQUARE_120, footprint(gazebo_model(), 25.0))
    assert high.lanes < low.lanes


@pytest.mark.parametrize(
    "polygon",
    [
        [(0, 0), (80, 0), (80, 30), (0, 30)],  # rectangle
        [(0, 0), (90, 20), (40, 100)],  # triangle
        [(10, -40), (70, 10), (20, 80), (-50, 30)],  # tilted quadrilateral
        [(0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100)],  # L shape (not convex)
    ],
)
@pytest.mark.parametrize("heading", [None, 0.0, 37.0, 90.0])
def test_every_part_of_any_polygon_is_covered(polygon, heading):
    plan = lawnmower(polygon, footprint(gazebo_model(), 15.0), heading_deg=heading, start=(0.0, 0.0))
    assert covered_fraction(polygon, plan, step_m=3.0) == 1.0


def test_default_heading_follows_the_longest_side():
    plan = lawnmower([(0, 0), (30, 0), (30, 200), (0, 200)], footprint(gazebo_model(), 15.0))
    assert plan.heading_deg == pytest.approx(90.0)  # the long sides run east-west
    assert plan.lanes == 1


def test_first_lane_starts_near_the_drone():
    plan = lawnmower(SQUARE_120, footprint(gazebo_model(), 15.0), heading_deg=0.0, start=(-60.0, 60.0))
    first = plan.waypoints[0]
    assert first.east_m == pytest.approx(45.0) and first.north_m < 0


def test_planner_rejects_bad_input():
    view = footprint(gazebo_model(), 15.0)
    with pytest.raises(ValueError, match="3 corners"):
        lawnmower([(0, 0), (1, 1)], view)
    with pytest.raises(ValueError, match="overlap"):
        lawnmower(SQUARE_120, view, overlap=0.95)


def test_point_in_polygon():
    assert point_in_polygon((1, 1), SQUARE_120) and not point_in_polygon((61, 0), SQUARE_120)


def test_bearing():
    assert bearing_deg((0, 0), (10, 0)) == pytest.approx(0.0)  # north
    assert bearing_deg((0, 0), (0, 10)) == pytest.approx(90.0)  # east
    assert bearing_deg((0, 0), (-10, 0)) == pytest.approx(180.0)
    assert bearing_deg((0, 0), (0, -10)) == pytest.approx(270.0)


# ---- oracle ------------------------------------------------------------------


def standing_person(east, north, person_id="person_1") -> dict:
    return {
        "id": person_id, "pose": "standing", "clothing": "red", "occlusion": "none",
        "box_centre": [east, north, 0.9], "box_size": [0.55, 0.36, 1.8],
        "box_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }  # fmt: skip


def test_box_corners():
    corners = box_corners(standing_person(5.0, 10.0))
    assert len(corners) == 8
    assert min(c[2] for c in corners) == pytest.approx(0.0) and max(c[2] for c in corners) == pytest.approx(1.8)
    assert min(c[0] for c in corners) == pytest.approx(5.0 - 0.275)


def test_oracle_sees_a_person_ahead_but_not_behind_or_too_far():
    model = gazebo_model()
    oracle = OracleDetector(
        [standing_person(0, 15, "ahead"), standing_person(0, -15, "behind"), standing_person(0, 150, "far")],
        model, min_person_px=12.0,
    )  # fmt: skip
    sightings = oracle.detect(pose_at(0, 0, 15, heading_deg=0))
    assert [s.person_id for s in sightings] == ["ahead"]
    seen = sightings[0]
    # In the middle of the image, and about as tall as the footprint predicts.
    assert seen.bbox.centre[0] == pytest.approx(640.0, abs=1.0)
    assert 30 < seen.bbox.height < 50
    # The estimate comes from the box's bottom edge, which is the near side
    # of the body: slightly short of the true 15 m, by less than half a metre.
    east, north = seen.ground_east_north
    assert east == pytest.approx(0.0, abs=0.05)
    assert 14.5 < north < 15.0


def test_oracle_requires_the_whole_box_in_the_image():
    model = gazebo_model()
    # Directly below the drone the camera (looking ahead) does not see.
    assert OracleDetector([standing_person(0, 0)], model).detect(pose_at(0, 0, 15)) == []
    assert BBox(-1, 5, 20, 30).inside(1280, 960) is False
    assert project_box(model, pose_at(0, 0, 15), box_corners(standing_person(0, -15))) is None


# ---- the search mission ------------------------------------------------------


def world_people(**overrides):
    row = VariationRow(**{
        "id": "t", "world": "meadow", "seed": 5, "people": 3, "altitude_m": 15.0, "area_m": 60.0,
        "poses": ("standing", "sitting", "lying"), **overrides,
    })  # fmt: skip
    spec = build_world(row).to_dict()
    return spec, people_in_map(spec)


def search(tmp_path, *, truth, detector=None, config=None, drone=None, area_m=60.0):
    drone = drone or FakeDrone()
    camera = SyntheticCamera(width=1280, height=960, hfov_deg=99.7, pitch_deg=45.0)
    meta = EpisodeMeta(world="meadow", weather="clear", time_of_day="noon", mission_prompt="search")
    path = tmp_path / "search.mcap"
    episode, recorder = asyncio.run(
        run_search(
            drone, path, meta, camera=camera, polygon=square_polygon(area_m), truth=truth,
            config=config or SearchConfig(), detector=detector, ground_z=-0.227,
            sleep=drone.sleep, start_wall_ns=START_NS,
        )  # fmt: skip
    )
    return drone, recorder, Episode(path), json.loads(path.with_suffix(".json").read_text())


@pytest.fixture(scope="module")
def found_everyone(tmp_path_factory):
    _, truth = world_people()
    return (*search(tmp_path_factory.mktemp("search"), truth=truth), truth)


def test_search_finds_and_reports_everyone(found_everyone):
    _, recorder, _, sidecar, truth = found_everyone
    assert sidecar["people_placed"] == 3 and sidecar["people_found"] == 3
    assert sidecar["success"] is True
    assert sidecar["detector"] == "oracle"
    assert sorted(r["person_id"] for r in sidecar["reports"]) == ["person_1", "person_2", "person_3"]
    by_id = {person["id"]: person for person in truth}
    for report in sidecar["reports"]:
        true_east, true_north = by_id[report["person_id"]]["box_centre"][:2]
        # Reported within 1 m of the truth, and the stored error is that distance.
        distance = math.hypot(report["east_m"] - true_east, report["north_m"] - true_north)
        assert distance < 1.0
        assert report["error_m"] == pytest.approx(distance, abs=0.02)
        assert report["frames_in_view"] >= 10


def test_state_machine_story(found_everyone):
    _, recorder, _, sidecar, _ = found_everyone
    states = [state for state, _ in recorder.transitions]
    assert states[:2] == ["TAKEOFF", "SEARCH"]
    assert states[-3:] == ["RTL", "LAND", "DONE"]
    # Three people: three times SEARCH -> CONFIRM -> REPORT -> RESUME -> SEARCH.
    middle = states[2:-3]
    assert middle == ["CONFIRM", "REPORT", "RESUME", "SEARCH"] * 3
    assert sidecar["transitions"] == states
    # Every step was a legal transition.
    for before, after in zip([None, *states], states):
        assert after in ALLOWED[before]
    assert recorder.transitions[-3] == ("RTL", "search complete")


def test_every_transition_is_in_the_event_log(found_everyone):
    _, recorder, episode, sidecar, _ = found_everyone
    messages = [log.message for log in episode.decoded(TOPIC_EVENTS)]
    state_lines = [m for m in messages if m.startswith("state: ")]
    assert len(state_lines) == len(recorder.transitions)
    assert state_lines[0] == "state: START -> TAKEOFF"
    assert "state: SEARCH -> CONFIRM (person_" in " ".join(state_lines)
    assert sidecar["event_counts"]["person_seen"] == 3
    assert sidecar["event_counts"]["person_confirmed"] == 3
    assert sidecar["event_counts"]["person_reported"] == 3
    assert sidecar["event_counts"]["search_complete"] == 1
    assert "battery_low" not in sidecar["event_counts"]
    times = episode.log_times(TOPIC_EVENTS)
    assert times == sorted(times)


def test_search_plan_is_recorded(found_everyone):
    _, recorder, _, sidecar, _ = found_everyone
    plan = sidecar["search"]
    # 60 m area, 50 m swath, 25 % overlap -> 2 lanes 30 m apart.
    assert plan["lanes"] == 2 and plan["lane_spacing_m"] == pytest.approx(30.0)
    # The camera is 15 m above the takeoff origin = 15.227 m above the ground:
    # 15.227 / sin(45 deg) * tan(49.85 deg) = 25.53 m.
    assert plan["swath_half_width_m"] == pytest.approx(25.53, abs=0.02)
    assert plan["overlap"] >= 0.25
    assert plan["altitude_m"] == 15.0


def test_drone_descends_to_confirm_and_returns_home(found_everyone):
    _, _, episode, _, _ = found_everyone
    positions = [pose.pose.position for pose in episode.decoded(TOPIC_POSE)]
    heights = [p.z for p in positions]
    assert max(heights) == pytest.approx(15.0, abs=0.05)  # search altitude
    in_flight = heights[200:-200]
    assert min(in_flight) == pytest.approx(10.0, abs=0.05)  # confirmation altitude
    # Lands where it took off.
    assert math.hypot(positions[-1].x, positions[-1].y) < 1.0
    assert heights[-1] == pytest.approx(0.0, abs=0.05)


def test_drone_circles_at_the_radius_that_centres_the_target(found_everyone):
    _, recorder, episode, sidecar, truth = found_everyone
    orbit = recorder.orbits[0]
    # Height 10 m above the origin = 10.227 m above the ground; at 45 deg
    # the camera centre looks that far ahead, so that is the orbit radius.
    assert orbit["radius_m"] == pytest.approx(10.227, abs=1e-3)
    assert len(recorder.orbits) == 3
    distances = [
        math.hypot(p.pose.position.x - orbit["east_m"], p.pose.position.y - orbit["north_m"])
        for p in episode.decoded(TOPIC_POSE)
        if abs(p.pose.position.z - 10.0) < 0.05
    ]
    on_circle = [d for d in distances if abs(d - 10.227) < 0.25]
    assert len(on_circle) >= 100  # a 120 deg arc at 2 m/s is about 10.7 s = 107 samples
    # The circle is centred within 1 m of where the person was finally reported.
    report = next(r for r in sidecar["reports"] if r["time_s"] == min(x["time_s"] for x in sidecar["reports"]))
    assert math.hypot(orbit["east_m"] - report["east_m"], orbit["north_m"] - report["north_m"]) < 1.0


class NoPeopleDetector:
    name = "blind"

    def detect(self, pose, jpeg=None):
        return []


def test_nobody_seen_means_full_coverage_then_home(tmp_path):
    _, truth = world_people()
    _, recorder, _, sidecar = search(tmp_path, truth=truth, detector=NoPeopleDetector())
    assert [state for state, _ in recorder.transitions] == ["TAKEOFF", "SEARCH", "RTL", "LAND", "DONE"]
    assert sidecar["people_found"] == 0 and sidecar["success"] is False
    assert sidecar["detector"] == "blind"
    assert sidecar["event_counts"]["waypoint_reached"] == 4  # 2 lanes x (start, end)


def test_empty_world_search_succeeds(tmp_path):
    _, recorder, _, sidecar = search(tmp_path, truth=[])
    assert sidecar["people_placed"] == 0 and sidecar["success"] is True


class GlimpseDetector:
    """Sees something once on the lane, never again while circling."""

    name = "glimpse"

    def __init__(self):
        self.calls = 0

    def detect(self, pose, jpeg=None):
        self.calls += 1
        if self.calls == 150:
            return [Sighting("ghost", BBox(600, 400, 640, 460), (pose.position.x, pose.position.y + 15.0))]
        return []


def test_unconfirmed_sighting_is_a_false_alarm_and_search_resumes(tmp_path):
    _, truth = world_people()
    # track_min_hits=1: act on a single sighting, as a mission without a tracker would.
    _, recorder, _, sidecar = search(
        tmp_path, truth=truth, detector=GlimpseDetector(), config=SearchConfig(track_min_hits=1)
    )
    states = [state for state, _ in recorder.transitions]
    assert states == ["TAKEOFF", "SEARCH", "CONFIRM", "RESUME", "SEARCH", "RTL", "LAND", "DONE"]
    assert sidecar["mission_metrics"]["false_alarms"] == 1
    assert ("RESUME", "not confirmed") in recorder.transitions
    assert sidecar["event_counts"]["false_alarm"] == 1
    assert "person_reported" not in sidecar["event_counts"]
    assert sidecar["reports"] == [] and sidecar["people_found"] == 0
    assert sidecar["event_counts"]["search_complete"] == 1  # the lanes were still finished


class ThirstyDrone(FakeDrone):
    BATTERY_DRAIN_PERCENT_PER_S = 1.0  # 100 % -> 30 % in 70 s


def test_low_battery_ends_the_search_early(tmp_path):
    _, truth = world_people()
    drone, recorder, episode, sidecar = search(
        tmp_path, truth=truth, detector=NoPeopleDetector(), drone=ThirstyDrone()
    )
    states = [state for state, _ in recorder.transitions]
    assert states == ["TAKEOFF", "SEARCH", "RTL", "LAND", "DONE"]
    assert recorder.transitions[2] == ("RTL", "battery low")
    assert sidecar["event_counts"]["battery_low"] == 1
    assert "search_complete" not in sidecar["event_counts"]
    assert sidecar["success"] is False, "an unfinished search is not a success"
    assert drone.commands[-2:] == ["land", "close"]
    # It turned for home at the 30 % threshold, not later.
    low = next(log.message for log in episode.decoded(TOPIC_EVENTS) if log.message.startswith("battery_low"))
    assert low == "battery_low: 30 %"


def test_illegal_transition_raises(tmp_path):
    _, truth = world_people()
    _, recorder, _, _ = search(tmp_path, truth=truth, detector=NoPeopleDetector())
    with pytest.raises(RuntimeError, match="illegal transition DONE -> SEARCH"):
        recorder._enter("SEARCH")


def test_confirm_altitude_stays_above_the_trees():
    spec, _ = world_people()
    tallest = max(6.53 * t["scale"] if t["kind"] == "oak" else 5.08 * t["scale"] for t in spec["trees"])
    altitude = safe_confirm_altitude(spec, 10.0)
    assert altitude >= tallest + 3.0 or altitude == spec["altitude_m"]
    assert altitude <= spec["altitude_m"]
    assert safe_confirm_altitude({**spec, "trees": []}, 10.0) == 10.0


# ---- detection in the loop: tracker, id-less detector, mission metrics --------


def test_tracker_filters_a_one_frame_glimpse(tmp_path):
    """With the tracker's default of three sightings, a single-frame false
    alarm never interrupts the search: no detour, no wasted battery."""
    _, truth = world_people()
    _, recorder, _, sidecar = search(tmp_path, truth=truth, detector=GlimpseDetector())
    assert [state for state, _ in recorder.transitions] == ["TAKEOFF", "SEARCH", "RTL", "LAND", "DONE"]
    assert sidecar["mission_metrics"]["false_alarms"] == 0
    assert "false_alarm" not in sidecar["event_counts"]


class NamelessDetector:
    """The oracle with the names removed, as a real detector would be:
    boxes and positions, but no idea who anyone is."""

    name = "nameless"

    def __init__(self, oracle, ghost_at=None):
        self.oracle, self.ghost_at = oracle, ghost_at

    def detect(self, pose, jpeg=None):
        found = [Sighting(None, s.bbox, s.ground_east_north, 0.8) for s in self.oracle.detect(pose)]
        if self.ghost_at is not None:
            # A steady false positive: something on the ground that looks like a person.
            from sardrone.mission.projection import pixel_to_ground, project

            pixel = project(self.oracle.model, pose, (*self.ghost_at, -0.227))
            if pixel and 30 < pixel[0] < 1250 and 60 < pixel[1] < 930:
                box = BBox(pixel[0] - 10, pixel[1] - 30, pixel[0] + 10, pixel[1])
                ground = pixel_to_ground(self.oracle.model, pose, pixel[0], pixel[1], -0.227)
                found.append(Sighting(None, box, ground, 0.6))
        return found


def nameless(truth, ghost_at=None):
    camera = SyntheticCamera(width=1280, height=960, hfov_deg=99.7, pitch_deg=45.0)
    oracle = OracleDetector(truth, CameraModel.from_source(camera), ground_z=-0.227)
    return NamelessDetector(oracle, ghost_at)


@pytest.fixture(scope="module")
def nameless_run(tmp_path_factory):
    _, truth = world_people()
    return (*search(tmp_path_factory.mktemp("nameless"), truth=truth, detector=nameless(truth)), truth)


def test_people_are_found_without_being_told_who_they_are(nameless_run):
    _, recorder, _, sidecar, truth = nameless_run
    metrics = sidecar["mission_metrics"]
    assert metrics["people_placed"] == 3 and metrics["people_found"] == 3
    assert metrics["missed"] == [] and metrics["false_reports"] == 0
    assert sidecar["people_found"] == 3 and sidecar["success"] is True
    # Each report was matched to a different true person, by position alone.
    assert sorted(r["matched_person"] for r in sidecar["reports"]) == ["person_1", "person_2", "person_3"]
    assert all(r["person_id"] is None and r["target"].startswith("target_") for r in sidecar["reports"])
    assert all(r["error_m"] < 1.5 for r in sidecar["reports"])
    # Nobody is confirmed twice: 3 detours, not more.
    assert sidecar["event_counts"]["person_reported"] == 3
    assert [s for s, _ in recorder.transitions].count("CONFIRM") == 3


def test_mission_metrics_are_complete_and_sensible(nameless_run):
    _, recorder, episode, sidecar, _ = nameless_run
    metrics = sidecar["mission_metrics"]
    first = min(r["time_s"] for r in sidecar["reports"])
    assert metrics["time_to_first_find_s"] == first and first > 0
    assert metrics["coverage_percent"] == 100.0  # both lanes flown to the end
    cte = metrics["cross_track_error"]
    # The fake drone follows its setpoints exactly, so it stays on the planned path.
    assert cte["samples"] > 200 and cte["mean_m"] < 0.5 and cte["max_m"] < 2.0
    assert metrics["mean_report_error_m"] < 1.5
    assert metrics["tracks_started"] >= 3
    # The generated test camera supplies no label boxes, so these cannot be measured.
    assert metrics["missed_rescues"] is None and metrics["tracker_ade"] is None


def test_tracks_are_logged_with_every_frame_and_as_events(nameless_run):
    from sardrone.mcap.episode_writer import TOPIC_DETECTIONS, TOPIC_IMAGE

    _, _, episode, sidecar, _ = nameless_run
    assert episode.log_times(TOPIC_DETECTIONS) == episode.log_times(TOPIC_IMAGE)
    labels = [t.text for message in episode.decoded(TOPIC_DETECTIONS) for t in message.texts]
    assert any(label == "track 1" for label in labels)
    assert sidecar["event_counts"]["target_acquired"] >= 3
    acquired = [m for m in (log.message for log in episode.decoded(TOPIC_EVENTS)) if m.startswith("target_acquired")]
    assert acquired[0] == "target_acquired: track 1"


def test_a_steady_false_positive_is_reported_as_nobody_there(tmp_path):
    """A detector that keeps seeing a "person" where there is none fools
    the confirmation orbit too. The mission metrics must show it."""
    _, truth = world_people()
    ghost = (5.0, -20.0)  # east, north: nobody stands here
    assert all(math.hypot(p["box_centre"][0] - ghost[0], p["box_centre"][1] - ghost[1]) > 6 for p in truth)
    _, recorder, episode, sidecar = search(tmp_path, truth=truth, detector=nameless(truth, ghost))
    metrics = sidecar["mission_metrics"]
    assert metrics["false_reports"] == 1
    false = next(r for r in sidecar["reports"] if r["matched_person"] is None)
    assert math.hypot(false["east_m"] - ghost[0], false["north_m"] - ghost[1]) < 1.5 and "error_m" not in false
    assert metrics["people_found"] == 3, "the real people are still found"
    assert sidecar["people_found"] == 3, "a false report does not count as a person found"
    assert any("NOBODY THERE" in log.message for log in episode.decoded(TOPIC_EVENTS))


def test_partial_search_has_partial_coverage(tmp_path):
    _, truth = world_people()
    _, recorder, _, sidecar = search(tmp_path, truth=truth, detector=NoPeopleDetector(), drone=ThirstyDrone())
    metrics = sidecar["mission_metrics"]
    assert metrics["coverage_percent"] in (0.0, 50.0)  # battery ran low before both lanes were done
    assert metrics["people_found"] == 0 and metrics["missed"] == ["person_1", "person_2", "person_3"]
    assert metrics["time_to_first_find_s"] is None


class TruthFeedingDetector:
    """Wraps a detector and, each frame, puts the simulator's label boxes on
    the camera, the way the Gazebo camera supplies them in a real flight."""

    def __init__(self, inner, oracle, camera, labels):
        self.inner, self.oracle, self.camera, self.labels = inner, oracle, camera, labels
        self.name = inner.name

    def detect(self, pose, jpeg=None):
        from sardrone.sim.camera import LabelBox

        self.camera.last_boxes = tuple(
            LabelBox(self.labels[s.person_id], s.bbox.x0, s.bbox.y0, s.bbox.x1, s.bbox.y1)
            for s in self.oracle.detect(pose)
        )
        return self.inner.detect(pose, jpeg)


def search_with_labels(tmp_path, truth, inner_factory):
    drone = FakeDrone()
    camera = SyntheticCamera(width=1280, height=960, hfov_deg=99.7, pitch_deg=45.0)
    camera.last_boxes = ()
    oracle = OracleDetector(truth, CameraModel.from_source(camera), ground_z=-0.227)
    detector = TruthFeedingDetector(inner_factory(oracle), oracle, camera, {p["id"]: p["label"] for p in truth})
    meta = EpisodeMeta(world="meadow", weather="clear", time_of_day="noon", mission_prompt="search")
    path = tmp_path / "search.mcap"
    asyncio.run(
        run_search(
            drone, path, meta, camera=camera, polygon=square_polygon(60.0), truth=truth, detector=detector,
            ground_z=-0.227, sleep=drone.sleep, start_wall_ns=START_NS,
        )  # fmt: skip
    )
    return json.loads(path.with_suffix(".json").read_text())


def test_missed_rescue_means_visible_but_never_reported(tmp_path):
    _, truth = world_people()
    # A detector that sees nothing, while the label boxes show all three people.
    sidecar = search_with_labels(tmp_path, truth, lambda oracle: NoPeopleDetector())
    metrics = sidecar["mission_metrics"]
    assert metrics["people_found"] == 0
    assert metrics["missed_rescues"] == ["person_1", "person_2", "person_3"]
    assert metrics["tracker_ade"] is None  # no tracks, nothing to score


def test_tracker_error_is_measured_against_label_boxes(tmp_path):
    _, truth = world_people()
    sidecar = search_with_labels(tmp_path, truth, lambda oracle: NamelessDetector(oracle))
    metrics = sidecar["mission_metrics"]
    assert metrics["missed_rescues"] == [] and metrics["people_found"] == 3
    ade = metrics["tracker_ade"]
    # The tracker predicts where each box will be one frame ahead; with
    # exact detections on a smoothly moving drone that is a few pixels off.
    assert ade["samples"] > 100 and 0.0 < ade["mean_px"] < 15.0


# ---- second pass over tree cover ----------------------------------------------


class LowOnlyDetector:
    """Finds people only from below 20 m: a detector for which a person under
    trees is too small from the main search altitude."""

    name = "low-only"

    def __init__(self, inner):
        self.inner = inner

    def detect(self, pose, jpeg=None):
        return self.inner.detect(pose, jpeg) if pose.position.z < 20.0 else []


def test_second_pass_finds_people_the_high_pass_missed(tmp_path):
    spec, truth = world_people(altitude_m=25.0)
    cover = [(-30.0, -30.0), (30.0, -30.0), (30.0, 30.0), (-30.0, 30.0)]

    def run(second_altitude):
        folder = tmp_path / str(second_altitude)
        folder.mkdir()
        drone = FakeDrone()
        camera = SyntheticCamera(width=1280, height=960, hfov_deg=99.7, pitch_deg=45.0)
        meta = EpisodeMeta(world="forest_edge", weather="clear", time_of_day="noon", mission_prompt="search")
        config = SearchConfig(altitude_m=25.0, second_pass_altitude_m=second_altitude)
        path = folder / "search.mcap"
        _, recorder = asyncio.run(
            run_search(
                drone, path, meta, camera=camera, polygon=square_polygon(60.0), truth=truth, config=config,
                detector=LowOnlyDetector(nameless(truth)), ground_z=-0.227, cover_polygon=cover,
                sleep=drone.sleep, start_wall_ns=START_NS,
            )  # fmt: skip
        )
        return recorder, Episode(path), json.loads(path.with_suffix(".json").read_text())

    _, _, without = run(None)
    assert without["mission_metrics"]["people_found"] == 0 and without["search"]["second_pass"] is None

    recorder, episode, with_pass = run(15.0)
    assert with_pass["mission_metrics"]["people_found"] == 3 and with_pass["success"] is True
    second = with_pass["search"]["second_pass"]
    assert second["altitude_m"] == 15.0 and second["lanes"] >= 2
    # The second pass crosses the first: its lanes are at right angles.
    assert abs(second["lane_heading_deg"] - with_pass["search"]["lane_heading_deg"]) == pytest.approx(90.0)
    messages = [log.message for log in episode.decoded(TOPIC_EVENTS)]
    assert any(m.startswith("state: SEARCH second pass:") and "15 m over tree cover" in m for m in messages)
    assert any(m.startswith("waypoint_reached: second-pass lane") for m in messages)
    # It flew at 15 m for the second pass, and coverage still refers to the main search.
    heights = [p.pose.position.z for p in episode.decoded(TOPIC_POSE)]
    assert sum(abs(h - 15.0) < 0.05 for h in heights) > 100
    assert with_pass["mission_metrics"]["coverage_percent"] == 100.0
    assert with_pass["duration_s"] > without["duration_s"], "the second pass costs flight time"


def test_tree_cover_polygon_wraps_the_trees_inside_the_area():
    from sardrone.mission.search import tree_cover_polygon

    spec = {"area_m": 60.0, "trees": [
        {"kind": "pine", "x": 10.0, "y": 20.0, "scale": 1.0},   # canopy radius 1.59 m
        {"kind": "oak", "x": -5.0, "y": 28.0, "scale": 1.0},    # canopy radius 5.92 m: sticks out of the area
    ]}  # fmt: skip
    (north0, east0), (north1, _), (_, east1), _ = tree_cover_polygon(spec, margin_m=2.0)
    assert east0 == pytest.approx(-5.0 - 5.92 - 2.0) and east1 == pytest.approx(10.0 + 1.59 + 2.0)
    assert north0 == pytest.approx(20.0 - 1.59 - 2.0)
    assert north1 == 30.0  # cut at the edge of the 60 m area
    assert tree_cover_polygon({"area_m": 60.0, "trees": []}) is None


def test_the_same_spot_is_never_reported_twice(tmp_path):
    """A target sighted from far away can get a rough position more than
    4 m from where it was already reported, so it looks like a new target;
    its orbit then ends on the reported spot. That must not be reported again.

    The detector below forces exactly that: first visit, sightings from the
    search altitude are 5 m east of the person and those from the orbit
    2 m east (3 m apart, so the orbit confirms; the report lands at +2);
    after that, sightings from the search altitude are 3 m WEST (5 m from
    the report: a "new" target) and those from the orbit exactly on the
    person (3 m from the sighting, so it confirms; 2 m from the report).
    """
    _, truth = world_people(poses=("standing",), people=1)

    class Drifting(NamelessDetector):
        def __init__(self, oracle):
            super().__init__(oracle)
            self.was_high, self.orbited, self.second_visit = False, False, False

        def detect(self, pose, jpeg=None):
            found = super().detect(pose, jpeg)
            low = pose.position.z < 12.0
            # Low AFTER having been high = the first orbit (takeoff passes
            # below 12 m too, and must not count).
            self.orbited |= low and self.was_high
            self.second_visit |= self.orbited and not low
            self.was_high |= not low
            if not self.second_visit:
                shift = 2.0 if low else 5.0
            else:
                shift = 0.0 if low else -3.0
            return [Sighting(None, s.bbox, (s.ground_east_north[0] + shift, s.ground_east_north[1]), s.confidence)
                    for s in found]  # fmt: skip

    _, recorder, episode, sidecar = search(tmp_path, truth=truth, detector=Drifting(nameless(truth).oracle))
    metrics = sidecar["mission_metrics"]
    assert metrics["duplicates_avoided"] >= 1, "the scenario must reach the duplicate case"
    assert len(sidecar["reports"]) == 1 and metrics["false_reports"] == 0 and metrics["people_found"] == 1
    assert ("RESUME", "already reported as target_1") in recorder.transitions
    # After one duplicate the target is remembered where it was sighted: no endless re-confirming.
    assert metrics["duplicates_avoided"] == 1
