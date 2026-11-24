"""The agent flying the drone (issue #8), with FakeDrone and the oracle
detector: no simulator, no language model. Same tools, same agent loop as
the mission simulator; here the tools fly and record."""

import json

import pytest

from sardrone.agent.flight import area_polygon, describe, fly_mission
from sardrone.agent.models import RuleBasedModel, ScriptedModel
from sardrone.mcap.episode_writer import EpisodeMeta
from sardrone.mission.oracle import OracleDetector
from sardrone.mission.projection import CameraModel
from sardrone.sim.camera import SyntheticCamera
from sardrone.sim.drone import FakeDrone

SPEC = {"altitude_m": 25, "area_m": 60.0, "drone_body_height_m": 0.227}


def person(person_id, east, north, clothing="red", pose="standing"):
    return {
        "id": person_id, "pose": pose, "clothing": clothing, "occlusion": "none",
        "box_centre": [east, north, 0.9], "box_size": [0.55, 0.36, 1.8], "box_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }  # fmt: skip


def fly(tmp_path, model, prompt, truth):
    drone = FakeDrone()
    camera = SyntheticCamera(width=1280, height=960, hfov_deg=99.7, pitch_deg=45.0)
    detector = OracleDetector(truth, CameraModel.from_source(camera), ground_z=-0.227)
    meta = EpisodeMeta(world="meadow", weather="clear", time_of_day="noon", mission_prompt=prompt)
    path = tmp_path / "agent.mcap"
    summary = fly_mission(model, prompt, drone=drone, camera=camera, detector=detector, truth=truth, spec=SPEC,
                          path=path, meta=meta, sleep=drone.sleep)  # fmt: skip
    return drone, summary, path


def test_areas_are_halves_of_the_world_square():
    north = area_polygon("north field", 60)
    assert min(n for n, _ in north) == 0 and max(n for n, _ in north) == 30
    west = area_polygon("west woods", 60)
    assert max(e for _, e in west) == 0 and min(e for _, e in west) == -30
    assert describe(person("p", 0, 0, "blue", "lying")) == "person in blue clothing, lying"


def test_the_rule_based_agent_flies_finds_reports_and_lands(tmp_path):
    truth = [person("person_1", 5.0, 12.0)]  # in the north field
    drone, summary, path = fly(tmp_path, RuleBasedModel(), "Search the north field and report anyone you find.", truth)
    assert summary["ended"] == "returned_home"
    assert summary["searched"] == [["north field", 25]] or summary["searched"] == [("north field", 25)]
    assert summary["reported_people"] == ["person_1"] and summary["false_reports"] == 0
    assert summary["rejected_calls"] == 0
    assert summary["tool_calls"][:2] == ["set_search_area", "start_search"]
    assert summary["tool_calls"][-1] == "return_home"
    assert drone.position[2] > -0.05 and "land" in drone.commands  # landed
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["agent_flight"]["reported_people"] == ["person_1"]
    assert path.exists() and path.stat().st_size > 0


def test_the_drone_comes_home_even_if_the_agent_does_not(tmp_path):
    # A model that starts a search and then stops answering (text, no tool call).
    script = [("set_search_area", {"area": "south field"}), ("start_search", {"altitude_m": 25})]
    drone, summary, _ = fly(tmp_path, ScriptedModel(script), "Search the south field.", [person("person_1", 0, -12)])
    assert summary["ended"] == "model_stopped"
    assert drone.position[2] > -0.05 and "land" in drone.commands  # the safety return still landed it
