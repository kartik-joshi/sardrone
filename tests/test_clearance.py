"""Clearance (issue #17): no mission flies lower than the tallest object plus
a margin, because there is no obstacle avoidance."""

import pytest

from sardrone.agent.flight import fly_mission
from sardrone.agent.models import ScriptedModel
from sardrone.mcap.episode_writer import EpisodeMeta
from sardrone.mission.clearance import (
    CLEARANCE_M,
    clearance_problems,
    lowest_safe_altitude_m,
    object_tops,
    require_clearance,
    tallest_object_m,
)
from sardrone.mission.oracle import OracleDetector
from sardrone.mission.projection import CameraModel
from sardrone.sim.camera import SyntheticCamera
from sardrone.sim.drone import FakeDrone
from sardrone.worlds.assets import ROCK_MODEL, TREES, VEHICLE_MODELS


def spec_with(trees=(), clutter=(), altitude=25):
    return {"id": "w", "altitude_m": altitude, "area_m": 60.0, "drone_body_height_m": 0.227,
            "trees": list(trees), "clutter": list(clutter)}  # fmt: skip


def test_tops_come_from_each_models_box_and_scale():
    oak = {"kind": "oak", "x": 0, "y": 0, "scale": 2.0}
    box = {"kind": "box", "z": 0.3, "size": [1.0, 1.0, 0.6]}
    suv = {"kind": "vehicle:suv", "z": 0.0, "size": [1.0, 1.0, 1.0]}
    rock = {"kind": "rock_mesh", "z": 0.2, "size": [1.1, 1.1, 1.1]}
    tops = dict(object_tops(spec_with([oak], [box, suv, rock])))
    assert tops["tree oak"] == pytest.approx(TREES["oak"].hi[2] * 2.0)
    assert tops["box"] == pytest.approx(0.9)  # centre + full height: an upper bound
    assert tops["vehicle:suv"] == pytest.approx(VEHICLE_MODELS["suv"].hi[2])
    assert tops["rock_mesh"] == pytest.approx(0.2 + ROCK_MODEL.hi[2] * 1.1)
    assert tallest_object_m(spec_with()) == 0.0


def test_a_too_low_altitude_is_named_and_refused():
    tall = {"kind": "oak", "x": 0, "y": 0, "scale": 2.0}  # about 13 m
    spec = spec_with([tall])
    top = TREES["oak"].hi[2] * 2.0
    problems = clearance_problems(spec, {"search": 25, "confirm": 10, "second pass": None})
    assert len(problems) == 1 and problems[0].startswith("confirm altitude 10 m")
    with pytest.raises(ValueError, match="confirm altitude"):
        require_clearance(spec, {"confirm": 10})
    require_clearance(spec, {"confirm": top + CLEARANCE_M})  # exactly the margin is fine
    assert lowest_safe_altitude_m(spec) == pytest.approx(top + CLEARANCE_M)


def _fly(tmp_path, script, spec):
    drone = FakeDrone()
    camera = SyntheticCamera(width=1280, height=960, hfov_deg=99.7, pitch_deg=45.0)
    detector = OracleDetector([], CameraModel.from_source(camera), ground_z=-0.227)
    meta = EpisodeMeta(world="forest_edge", weather="clear", time_of_day="noon", mission_prompt="search")
    return fly_mission(ScriptedModel(script), "search", drone=drone, camera=camera, detector=detector, truth=[],
                       spec=spec, path=tmp_path / "a.mcap", meta=meta, sleep=drone.sleep)  # fmt: skip


def test_the_agent_cannot_fly_below_the_tallest_tree(tmp_path):
    spec = spec_with([{"kind": "oak", "x": 5, "y": 5, "scale": 2.0}], altitude=40)  # ~13 m oak: 15 m is too low
    script = [("set_search_area", {"area": "north field"}), ("start_search", {"altitude_m": 15})]
    summary = _fly(tmp_path, script, spec)
    assert summary["rejected_calls"] == 1 and summary["searched"] == []


def test_the_agents_confirm_orbit_is_clamped_above_the_trees(tmp_path, monkeypatch):
    import sardrone.agent.flight as flight

    seen = {}
    real = flight.AgentSearch.__init__

    def spy(self, *args, **kwargs):
        seen["confirm"] = kwargs["config"].confirm_altitude_m
        real(self, *args, **kwargs)

    monkeypatch.setattr(flight.AgentSearch, "__init__", spy)
    spec = spec_with([{"kind": "oak", "x": 5, "y": 5, "scale": 1.4}], altitude=25)  # ~9 m, like train-042
    _fly(tmp_path, [("set_search_area", {"area": "north field"})], spec)
    assert seen["confirm"] >= TREES["oak"].hi[2] * 1.4 + CLEARANCE_M  # was a fixed 10 m
