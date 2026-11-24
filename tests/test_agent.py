"""Phase 4A: the mission simulator, the agent loop and the scoring.

No language model is called here. A reference policy that is TOLD the right
answer checks that every written mission can be solved and that the scorer
accepts a correct run; scripted runs check that wrong behaviour is caught.
"""

import json

import pytest

from sardrone.agent.agent import run_agent
from sardrone.agent.evaluate import HARD_MISSIONS, build_sim, check, load_missions, run_mission, summarise
from sardrone.agent.models import Reply, RuleBasedModel, ScriptedModel, ToolCall
from sardrone.agent.sim import MissionSim, Person, Scenario
from sardrone.agent.tools import TOOL_NAMES, TOOLS, call_tool

MISSIONS = {m["id"]: m for m in load_missions()}


def sim_with(*people, battery=100.0):
    return MissionSim(Scenario(people=tuple(people), battery=battery))


RED = Person("p1", "north field", "adult in a red jacket, sitting", 40)


# ---- simulator ---------------------------------------------------------------


def test_search_costs_battery_and_reveals_people_as_coverage_grows():
    sim = sim_with(RED)
    assert sim.set_search_area("north field")["ok"]
    assert sim.start_search(25)["battery_percent"] == 97.0  # take-off costs 3
    first = sim.get_status()
    assert (first["coverage_percent"], first["battery_percent"], first["unconfirmed_sightings"]) == (20.0, 91.0, [])
    second = sim.get_status()
    assert second["coverage_percent"] == 40.0 and second["unconfirmed_sightings"] == ["target_1"]
    for _ in range(3):
        last = sim.get_status()
    # Five status calls: 100 % covered, 3 + 5 x 6 = 33 % of the battery used.
    assert last["state"] == "SEARCH_COMPLETE" and last["battery_percent"] == 67.0


def test_a_person_must_be_confirmed_before_being_reported():
    sim = sim_with(RED)
    sim.set_search_area("north field"), sim.start_search(25), sim.get_status(), sim.get_status()
    assert sim.report_person("target_1") == {"ok": False, "error": "confirm the target before reporting it"}
    confirmed = sim.confirm_target("target_1")
    assert confirmed["description"] == "adult in a red jacket, sitting" and confirmed["battery_percent"] == 81.0
    assert sim.report_person("target_1")["ok"] and sim.reported_people() == ["p1"]
    assert sim.report_person("target_1")["ok"] is False  # not twice
    assert sim.errors == 2


def test_people_outside_the_searched_area_are_not_seen():
    sim = sim_with(RED)
    sim.set_search_area("south field"), sim.start_search(25)
    for _ in range(5):
        status = sim.get_status()
    assert status["unconfirmed_sightings"] == [] and status["state"] == "SEARCH_COMPLETE"
    whole = sim_with(RED)
    whole.set_search_area("whole area"), whole.start_search(25)
    assert [whole.get_status()["unconfirmed_sightings"] for _ in range(2)][-1] == ["target_1"]


def test_bad_calls_are_errors_the_model_can_read():
    sim = sim_with()
    assert "set a search area first" in sim.start_search(25)["error"]
    assert "unknown area" in sim.set_search_area("the moon")["error"]
    sim.set_search_area("north field")
    assert "altitude must be" in sim.start_search(30)["error"]
    assert "unknown tool" in call_tool(sim, "self_destruct", {})["error"]
    assert "bad arguments" in call_tool(sim, "start_search", {"height": 25})["error"]
    assert sim.errors == 5
    sim.return_home()
    assert "mission is over" in sim.start_search(25)["error"]


def test_flying_until_the_battery_is_empty_crashes():
    sim = sim_with(battery=20.0)
    sim.set_search_area("whole area"), sim.start_search(25)
    for _ in range(3):
        status = sim.get_status()
    assert status["state"] == "CRASHED" and status["battery_percent"] == 0.0
    assert sim.return_home()["ok"] is False


def test_tool_schemas_match_the_simulator():
    assert TOOL_NAMES == ("set_search_area", "start_search", "get_status", "confirm_target", "report_person", "return_home")
    for tool in TOOLS:
        assert callable(getattr(MissionSim, tool["name"]))
        assert tool["parameters"]["type"] == "object" and tool["description"]
    json.dumps(TOOLS)  # must be plain JSON to send to a model


# ---- agent loop --------------------------------------------------------------


def test_agent_loop_runs_tools_and_stops_at_home():
    script = [("set_search_area", {"area": "north field"}), ("start_search", {"altitude_m": 25}),
              ("get_status", {}), ("return_home", {}), ("get_status", {})]  # fmt: skip
    decisions = []
    transcript = run_agent(ScriptedModel(script), "search", sim_with(RED), on_decision=decisions.append)
    assert transcript.tool_names() == ["set_search_area", "start_search", "get_status", "return_home"]
    assert transcript.ended == "returned_home" and transcript.model_calls == 4
    assert (transcript.input_tokens, transcript.output_tokens) == (400, 40)
    assert decisions[0] == 'set_search_area({"area": "north field"}) -> ok'
    assert transcript.steps[2].battery_after == 91.0 and transcript.steps[2].coverage_after == 20.0


def test_agent_loop_gives_one_reminder_then_stops():
    transcript = run_agent(ScriptedModel([]), "search", sim_with())
    assert transcript.ended == "model_stopped" and transcript.steps == [] and transcript.model_calls == 2


def test_agent_loop_is_cut_off_after_max_steps():
    class Looper:
        name = "looper"

        def reply(self, messages, tools):
            return Reply((ToolCall("x", "get_status", {}),))

    transcript = run_agent(Looper(), "search", sim_with(), max_steps=7)
    assert transcript.ended == "max_steps" and len(transcript.steps) == 7


def test_tool_results_reach_the_model_as_messages():
    seen = []

    class Recorder(ScriptedModel):
        def reply(self, messages, tools):
            seen.append([m["role"] for m in messages])
            return super().reply(messages, tools)

    run_agent(Recorder([("get_status", {}), ("return_home", {})]), "status please", sim_with())
    assert seen[0] == ["system", "user"]
    assert seen[1] == ["system", "user", "assistant", "tool"]


# ---- scoring -----------------------------------------------------------------


class ReferenceModel:
    """A policy that is handed the mission's expected outcome. It exists to
    prove each mission is solvable and that the scorer passes a correct run."""

    name = "reference"

    def __init__(self, mission):
        self.expect, self.areas, self.started = mission["expect"], list(mission["expect"]["areas"]), False
        people = {p["id"]: p for p in (mission.get("scenario") or {}).get("people") or []}
        self.wanted = {people[i]["description"] for i in self.expect["report"]}
        self.confirm_all = "confirm" in self.expect

    def reply(self, messages, tools):
        last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else {}
        calls = [c for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]
        previous = calls[-1]["function"]["name"] if calls else None

        def call(name, **arguments):
            return Reply((ToolCall(f"r{len(calls)}", name, arguments),))

        if self.expect.get("no_flight"):
            return call("get_status") if not calls else Reply(text="The drone is idle on the ground.")
        if previous == "confirm_target":
            target = json.loads(calls[-1]["function"]["arguments"])["target_id"]
            if last["description"] in self.wanted:
                return call("report_person", target_id=target)
        limit = self.expect.get("battery_limit")
        if limit is not None and self.started and last.get("battery_percent", 100) <= limit:
            return call("return_home")
        if not self.started:
            if previous != "set_search_area":
                return call("set_search_area", area=self.areas[0])
            self.started = True
            return call("start_search", altitude_m=self.expect.get("altitude_m", 25))
        if previous != "get_status":
            return call("get_status")
        if last["unconfirmed_sightings"]:
            return call("confirm_target", target_id=last["unconfirmed_sightings"][0])
        if last["state"] == "SEARCH_COMPLETE":
            self.areas.pop(0)
            if self.areas:
                self.started = False
                return call("set_search_area", area=self.areas[0])
            return call("return_home")
        return call("get_status")


HARD = {m["id"]: m for m in load_missions(HARD_MISSIONS)}


@pytest.mark.parametrize("mission_id", sorted(MISSIONS) + sorted(HARD))
def test_every_mission_is_solvable_and_a_correct_run_passes(mission_id):
    mission = MISSIONS.get(mission_id) or HARD[mission_id]
    result = run_mission(ReferenceModel(mission), mission)
    assert result["success"], result["failed_checks"]
    assert result["valid_calls"] == result["calls"], "a correct run makes no rejected calls"


def test_there_are_twenty_distinct_missions():
    assert len(MISSIONS) == 20 and len({m["prompt"] for m in MISSIONS.values()}) == 20


def test_the_hard_set_is_twelve_new_missions():
    assert len(HARD) == 12
    prompts = {m["prompt"] for m in HARD.values()}
    assert len(prompts) == 12 and not prompts & {m["prompt"] for m in MISSIONS.values()}
    assert not set(HARD) & set(MISSIONS)


def scripted(mission_id, script):
    mission = MISSIONS[mission_id]
    sim = build_sim(mission)
    transcript = run_agent(ScriptedModel(script), mission["prompt"], sim)
    return check(mission, sim, transcript)


SEARCH_NORTH = [("set_search_area", {"area": "north field"}), ("start_search", {"altitude_m": 25})]
STATUS = ("get_status", {})


def test_scorer_catches_reporting_the_wrong_person():
    # m15: report anyone EXCEPT people in black. Reporting the first sighting (black) is wrong.
    checks = scripted("m15", SEARCH_NORTH + [STATUS, ("confirm_target", {"target_id": "target_1"}),
                      ("report_person", {"target_id": "target_1"}), STATUS, STATUS, STATUS, STATUS, ("return_home", {})])  # fmt: skip
    assert checks["reported_exactly_the_right_people"] is False and checks["finished_every_search"] is True


def test_scorer_catches_ignoring_the_battery_limit():
    # m03: come back at 75 % or lower. Battery is 73 after the 4th status; flying on is a violation.
    checks = scripted("m03", [("set_search_area", {"area": "east meadow"}), ("start_search", {"altitude_m": 25})]
                      + [STATUS] * 5 + [("return_home", {})])  # fmt: skip
    assert checks["obeyed_the_battery_limit"] is False
    obedient = scripted("m03", [("set_search_area", {"area": "east meadow"}), ("start_search", {"altitude_m": 25})]
                        + [STATUS] * 4 + [("return_home", {})])  # fmt: skip
    assert obedient["obeyed_the_battery_limit"] is True and all(obedient.values())


def test_scorer_catches_going_home_before_the_search_is_done():
    checks = scripted("m01", SEARCH_NORTH + [STATUS, ("return_home", {})])
    assert checks["finished_every_search"] is False and checks["reported_exactly_the_right_people"] is False


def test_scorer_catches_wrong_area_wrong_altitude_and_never_coming_home():
    wrong_area = scripted("m12", [("set_search_area", {"area": "whole area"}), ("start_search", {"altitude_m": 25})]
                          + [STATUS] * 5 + [("return_home", {})])  # fmt: skip
    assert wrong_area["searched_the_right_areas_in_order"] is False
    wrong_altitude = scripted("m04", [("set_search_area", {"area": "west woods"}), ("start_search", {"altitude_m": 25}), STATUS])
    assert wrong_altitude["flew_at_the_requested_altitude"] is False
    assert wrong_altitude["ended_at_home"] is False


def test_scorer_catches_launching_when_told_not_to():
    assert scripted("m20", SEARCH_NORTH)["did_not_take_off"] is False
    assert all(scripted("m20", [STATUS]).values())


def test_rule_based_baseline_is_deterministic_and_fails_what_needs_understanding():
    results = [run_mission(RuleBasedModel(), mission) for mission in MISSIONS.values()]
    again = [run_mission(RuleBasedModel(), mission) for mission in MISSIONS.values()]
    assert [r["success"] for r in results] == [r["success"] for r in again]
    passed = {r["id"] for r in results if r["success"]}
    assert {"m01", "m02", "m03", "m09", "m10"} <= passed  # plain area, colour, battery
    # Negation, ordering, "do not launch", two attributes: keyword rules cannot do these.
    assert not {"m05", "m15", "m17", "m20"} & passed
    summary = summarise("rule-based", results)
    assert summary["missions"] == 20 and summary["succeeded"] == len(passed)
    assert summary["valid_tool_call_rate"] == 1.0 and summary["cost_usd"] is None


def test_cost_is_computed_only_from_given_prices():
    results = [{"success": True, "calls": 4, "valid_calls": 4, "steps": 4, "input_tokens": 2_000_000,
                "output_tokens": 500_000, "latency_s": 2.0}]  # fmt: skip
    assert summarise("m", results)["cost_usd"] is None
    assert summarise("m", results, price_in=2.5, price_out=10.0)["cost_usd"] == 10.0  # 2 x 2.5 + 0.5 x 10


# ---- decisions on the episode timeline ----------------------------------------


def test_agent_decisions_are_written_to_events(tmp_path):
    from sardrone.agent.record import record_mission
    from sardrone.mcap.episode_writer import TOPIC_EVENTS
    from tests.conftest import START_NS, Episode

    mission = MISSIONS["m02"]  # red jacket only: the baseline confirms two people and reports one
    path = tmp_path / "agent.mcap"
    _, transcript, checks = record_mission(RuleBasedModel(), mission, path, start_wall_ns=START_NS)
    assert all(checks.values())

    episode = Episode(path)
    messages = [log.message for log in episode.decoded(TOPIC_EVENTS)]
    assert messages[0] == f"agent: mission: {mission['prompt']}"
    assert messages[1] == 'agent: set_search_area({"area": "south field"}) -> ok'
    assert messages[-1] == "agent: ended: returned_home"
    # One event per tool call, plus the mission line and the ending line.
    assert len(messages) == len(transcript.steps) + 2
    assert sum("report_person" in m for m in messages) == 1 and sum("confirm_target" in m for m in messages) == 2
    times = episode.log_times(TOPIC_EVENTS)
    assert times[0] == START_NS and times == sorted(times)

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["success"] is True and sidecar["agent_model"] == "rule-based"
    assert sidecar["mission_prompt"] == mission["prompt"] and sidecar["people_found"] == 1
    assert sidecar["agent_tools"] == transcript.tool_names()
    assert sidecar["event_counts"] == {"agent": len(messages)}


def test_a_failed_mission_is_recorded_as_failed(tmp_path):
    from sardrone.agent.record import record_mission
    from tests.conftest import START_NS

    path = tmp_path / "agent.mcap"
    _, _, checks = record_mission(RuleBasedModel(), MISSIONS["m15"], path, start_wall_ns=START_NS)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["success"] is False
    assert sidecar["agent_checks"]["reported_exactly_the_right_people"] is False
