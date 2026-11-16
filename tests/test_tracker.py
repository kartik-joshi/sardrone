"""Phase 3D groundwork: the tracker's memory and the mission metrics."""

import math

import pytest

from sardrone.mission.metrics import (
    average_displacement_error,
    cross_track_errors,
    cte_summary,
    distance_to_segment,
)
from sardrone.mission.tracker import BoxFilter, Tracker, associate, box_iou, centre


def box_at(cx, cy, w=20.0, h=40.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ---- Kalman filter -----------------------------------------------------------


def test_filter_learns_a_steady_velocity():
    # A person drifting 5 px right and 2 px down per frame.
    kalman = BoxFilter(box_at(100, 100))
    for step in range(1, 30):
        kalman.predict()
        kalman.update(box_at(100 + 5 * step, 100 + 2 * step))
    vx, vy = kalman.velocity
    assert vx == pytest.approx(5.0, abs=0.2) and vy == pytest.approx(2.0, abs=0.2)
    # With no detection it keeps moving at that velocity.
    start = centre(kalman.box)
    for _ in range(4):
        kalman.predict()
    moved = centre(kalman.box)
    assert moved[0] - start[0] == pytest.approx(20.0, abs=1.0)
    assert moved[1] - start[1] == pytest.approx(8.0, abs=0.5)


def test_filter_smooths_a_noisy_detector():
    import random

    rng = random.Random(3)
    kalman = BoxFilter(box_at(200, 200))
    raw_error, filtered_error = [], []
    for _ in range(200):
        noisy = (200 + rng.gauss(0, 3), 200 + rng.gauss(0, 3))
        kalman.predict()
        kalman.update(box_at(*noisy))
        raw_error.append(math.dist(noisy, (200, 200)))
        filtered_error.append(math.dist(centre(kalman.box), (200, 200)))
    # A still person: the filtered position is steadier than the raw boxes.
    assert sum(filtered_error[50:]) < 0.75 * sum(raw_error[50:])


def test_uncertainty_grows_while_coasting_and_shrinks_on_a_detection():
    kalman = BoxFilter(box_at(100, 100))
    kalman.predict()
    kalman.update(box_at(100, 100))
    sure = kalman.P[0, 0]
    for _ in range(5):
        kalman.predict()
    assert kalman.P[0, 0] > sure
    unsure = kalman.P[0, 0]
    kalman.update(box_at(100, 100))
    assert kalman.P[0, 0] < unsure


# ---- association -------------------------------------------------------------


def test_iou():
    assert box_iou(box_at(0, 0), box_at(0, 0)) == 1.0
    assert box_iou(box_at(0, 0), box_at(100, 0)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_association_maximises_total_overlap():
    tracks = [box_at(100, 100), box_at(112, 100)]
    detections = [box_at(109, 100), box_at(121, 100)]  # both people moved 9 px right
    # Greedy would give detection 0 to track 1 (3 px away, the best single
    # overlap), leaving track 0 with a far detection. The right answer pairs in order.
    assert sorted(associate(tracks, detections, 0.1)) == [(0, 0), (1, 1)]
    assert associate(tracks, [], 0.1) == [] and associate([], detections, 0.1) == []
    assert associate([box_at(0, 0)], [box_at(500, 500)], 0.1) == []  # too far apart to pair


# ---- tracker -----------------------------------------------------------------


def walk(frames, start=(100.0, 200.0), velocity=(4.0, 1.0)):
    return [box_at(start[0] + velocity[0] * i, start[1] + velocity[1] * i) for i in range(frames)]


def test_track_is_reported_only_after_min_hits():
    tracker = Tracker(min_hits=3)
    path = walk(5)
    assert tracker.step([path[0]]) == [] and tracker.step([path[1]]) == []
    confirmed = tracker.step([path[2]])
    assert [t.id for t in confirmed] == [1] and confirmed[0].hits == 3


def test_identity_survives_a_gap_in_detections():
    tracker = Tracker(max_age=10, min_hits=3)
    path = walk(30)
    ids, coasting = [], []
    for frame, box in enumerate(path):
        seen = [] if 12 <= frame < 18 else [box]  # the detector misses 6 frames
        tracks = tracker.step(seen)
        ids.append([t.id for t in tracks])
        coasting.append(bool(tracks and tracks[0].coasting))
    assert ids[2:] == [[1]] * 28, "one person, one identity, also through and after the gap"
    assert coasting[12:18] == [True] * 6 and not any(coasting[18:])
    assert tracker.lost == []


def test_coasting_follows_the_motion_model():
    tracker = Tracker(max_age=10)
    path = walk(30)
    errors = []
    for frame, box in enumerate(path):
        tracks = tracker.step([] if 12 <= frame < 18 else [box])
        if 12 <= frame < 18:
            errors.append(math.dist(centre(tracks[0].box), centre(box)))
    # While blind, the prediction stays within a few pixels of the truth
    # (the person moved 4.1 px per frame, 25 px over the gap).
    assert max(errors) < 5.0


def test_track_is_forgotten_after_max_age_and_a_new_person_gets_a_new_id():
    tracker = Tracker(max_age=3, min_hits=1)
    for box in walk(5):
        tracker.step([box])
    for _ in range(4):
        tracks = tracker.step([])
    assert tracks == [] and [t.id for t in tracker.lost] == [1]
    assert [t.id for t in tracker.step([box_at(600, 400)])] == [2]


def test_two_people_keep_their_own_ids_when_one_is_missed():
    tracker = Tracker(max_age=10, min_hits=2)
    first, second = walk(20, (100, 100), (3, 0)), walk(20, (400, 300), (-3, 0))
    for frame in range(20):
        seen = [first[frame]] if 8 <= frame < 12 else [first[frame], second[frame]]
        tracks = tracker.step(seen)
    by_id = {t.id: t for t in tracks}
    assert sorted(by_id) == [1, 2]
    assert centre(by_id[1].box)[0] == pytest.approx(centre(first[-1])[0], abs=3)
    assert centre(by_id[2].box)[0] == pytest.approx(centre(second[-1])[0], abs=3)


def test_one_frame_false_alarm_never_becomes_a_person():
    tracker = Tracker(min_hits=3, max_age=2)
    reported = []
    for frame in range(10):
        reported += tracker.step([box_at(300, 300)] if frame == 4 else [])
    assert reported == []


def test_history_marks_detected_and_coasted_frames():
    tracker = Tracker(max_age=5, min_hits=1)
    for frame, box in enumerate(walk(6)):
        tracker.step([] if frame in (3, 4) else [box])
    states = [state for _, _, state in tracker.tracks[0].history]
    assert states == ["detected", "detected", "detected", "coasted", "coasted", "detected"]


# ---- mission metrics ---------------------------------------------------------


def test_distance_to_segment():
    a, b = (0.0, 0.0), (10.0, 0.0)
    assert distance_to_segment((5.0, 3.0), a, b) == 3.0  # straight above the middle
    assert distance_to_segment((5.0, 0.0), a, b) == 0.0  # on the line
    assert distance_to_segment((13.0, 4.0), a, b) == 5.0  # past the end: distance to (10, 0)
    assert distance_to_segment((-3.0, -4.0), a, b) == 5.0
    assert distance_to_segment((1.0, 1.0), a, a) == pytest.approx(math.sqrt(2))  # zero-length segment


def test_cross_track_error_uses_the_nearest_leg_of_the_path():
    path = [(0.0, 0.0), (100.0, 0.0), (100.0, 30.0), (0.0, 30.0)]  # up one lane, across, back
    flown = [(10.0, 1.0), (50.0, -2.0), (101.5, 15.0), (40.0, 29.0)]
    assert cross_track_errors(flown, path) == pytest.approx([1.0, 2.0, 1.5, 1.0])
    summary = cte_summary(flown, path)
    assert summary == {"samples": 4, "mean_m": 1.375, "p95_m": 2.0, "max_m": 2.0}
    with pytest.raises(ValueError):
        cross_track_errors(flown, [(0.0, 0.0)])


def test_average_displacement_error():
    predicted = [(0.0, 0.0), (3.0, 4.0), (10.0, 10.0)]
    actual = [(0.0, 0.0), (0.0, 0.0), (10.0, 13.0)]
    assert average_displacement_error(predicted, actual) == pytest.approx((0 + 5 + 3) / 3)
    assert average_displacement_error([], []) == 0.0
    with pytest.raises(ValueError):
        average_displacement_error(predicted, actual[:2])


def test_mission_report_totals(tmp_path):
    from sardrone.mission.report import row_from_sidecar, totals, write_report

    def sidecar(episode, placed, found, rescues, false_reports, first):
        return {
            "episode_id": episode, "world": "meadow", "time_of_day": "noon", "altitude_m": 15, "occlusion": "none",
            "detector": "yolo:v1", "duration_s": 200.04, "event_counts": {"search_complete": 1},
            "mission_metrics": {
                "people_placed": placed, "people_found": found, "missed_rescues": rescues,
                "false_reports": false_reports, "false_alarms": 1, "time_to_first_find_s": first,
                "mean_report_error_m": 0.5, "coverage_percent": 100.0,
                "cross_track_error": {"mean_m": 0.4, "max_m": 1.5}, "tracker_ade": {"mean_px": 3.0, "samples": 9},
                "tracker_ade_coasting": None,
            },
        }  # fmt: skip

    rows = [
        row_from_sidecar(sidecar("test-001", 3, 3, [], 0, 40.0)),
        row_from_sidecar(sidecar("test-002", 2, 1, ["person_2"], 1, 60.0)),
        row_from_sidecar(sidecar("test-003", 1, 0, None, 0, None)),  # nothing measurable
    ]
    assert rows[1]["missed_rescues"] == 1 and rows[2]["missed_rescues"] == "" and rows[2]["time_to_first_find_s"] == ""
    summary = totals(rows)
    assert (summary["people_placed"], summary["people_found"]) == (6, 4)
    assert summary["recall"] == pytest.approx(4 / 6, abs=1e-4)
    assert summary["episodes_all_found"] == 1 and summary["false_reports"] == 1 and summary["false_alarms"] == 3
    assert summary["missed_rescues"] == 1
    assert summary["mean_time_to_first_find_s"] == "50.00"  # the episode with no find is left out
    assert summary["mean_ade_coasting_px"] == "n/a"
    write_report(rows, "t", "Test", out_dir=tmp_path)
    text = (tmp_path / "t.md").read_text()
    assert "| People found (reported within 3 m) | 4 |" in text and "| test-002 |" in text
    assert (tmp_path / "t.csv").read_text().splitlines()[0].startswith("episode_id,world")
