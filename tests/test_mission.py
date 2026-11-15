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
    _, recorder, _, sidecar = search(tmp_path, truth=truth, detector=GlimpseDetector())
    states = [state for state, _ in recorder.transitions]
    assert states == ["TAKEOFF", "SEARCH", "CONFIRM", "RESUME", "SEARCH", "RTL", "LAND", "DONE"]
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
