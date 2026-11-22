"""Score an agent on the written missions.

    make agent-eval                          # the rule-based baseline (no API, free)
    make agent-eval MODEL=cohere             # Cohere, needs COHERE_API_KEY in .env

For each mission the agent's transcript is checked against what a correct
run looks like (see missions.yaml). A mission SUCCEEDS only if every check
passes. Besides the success rate the report gives, per model:

    valid tool calls   share of calls the simulator accepted (a call with a
                       wrong name, wrong arguments or in the wrong order is
                       rejected and counted against the agent)
    steps              tool calls per mission
    tokens, latency    what the model consumed and how long it took

Token counts are reported as counted by the provider. They are turned into
a cost only if prices are passed on the command line; no price is assumed.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from sardrone.agent.agent import Transcript, run_agent
from sardrone.agent.models import ChatModel, RuleBasedModel
from sardrone.agent.sim import MissionSim, Person, Scenario

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MISSIONS = Path(__file__).with_name("missions.yaml")
RESULTS = PROJECT_ROOT / "docs" / "results"


def load_missions(path: Path = MISSIONS) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())


def build_sim(mission: dict) -> MissionSim:
    scenario = mission.get("scenario") or {}
    people = tuple(Person(**person) for person in scenario.get("people") or [])
    return MissionSim(Scenario(people=people, battery=float(scenario.get("battery", 100.0))))


def searched_areas(transcript: Transcript) -> list[tuple[str, float]]:
    """[(area, highest coverage reached)] for each search that was started."""
    searches, area = [], None
    for step in transcript.steps:
        if not step.result.get("ok"):
            continue
        if step.tool == "set_search_area":
            area = step.arguments["area"]
        elif step.tool == "start_search":
            searches.append([area, 0.0])
        elif searches:
            searches[-1][1] = max(searches[-1][1], step.coverage_after)
    return [(a, c) for a, c in searches]


def check(mission: dict, sim: MissionSim, transcript: Transcript) -> dict:
    """Every check for one mission: {name: passed}."""
    expect = mission["expect"]
    checks = {}
    searches = searched_areas(transcript)
    starts = [s for s in transcript.steps if s.tool == "start_search" and s.result.get("ok")]

    if expect.get("no_flight"):
        checks["did_not_take_off"] = not starts
        checks["nothing_reported"] = sim.reported_people() == []
        return checks

    checks["ended_at_home"] = transcript.ended == "returned_home"
    checks["searched_the_right_areas_in_order"] = [a for a, _ in searches] == expect["areas"]
    if "altitude_m" in expect:
        checks["flew_at_the_requested_altitude"] = bool(starts) and all(
            s.arguments.get("altitude_m") == expect["altitude_m"] for s in starts
        )
    checks["reported_exactly_the_right_people"] = sorted(sim.reported_people()) == sorted(expect["report"])
    if "confirm" in expect:
        confirmed = {sim._targets[t].id for t in sim.confirmed}
        checks["confirmed_the_sightings"] = set(expect["confirm"]) <= confirmed

    limit = expect.get("battery_limit")
    triggered = False
    if limit is not None:
        # After the first moment the battery is at or below the limit, the
        # only things allowed are reporting people already confirmed, and
        # going home.
        obeyed = True
        for index, step in enumerate(transcript.steps):
            if step.state_after == "HOME":
                break
            if step.battery_after <= limit and step.tool != "return_home":
                triggered = True
                rest = [s.tool for s in transcript.steps[index + 1 :]]
                allowed = [t for t in rest if t != "report_person"]
                obeyed = bool(allowed) and allowed[0] == "return_home" and len(allowed) == 1
                break
        checks["obeyed_the_battery_limit"] = obeyed
    if not triggered:
        # Not cut short by the battery: every search must have been finished.
        checks["finished_every_search"] = bool(searches) and all(c >= 100.0 for _, c in searches)
    return checks


def run_mission(model: ChatModel, mission: dict, max_steps: int = 40) -> dict:
    sim = build_sim(mission)
    transcript = run_agent(model, mission["prompt"], sim, max_steps)
    checks = check(mission, sim, transcript)
    calls = len(sim.log)
    return {
        "id": mission["id"],
        "prompt": mission["prompt"],
        "success": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "steps": len(transcript.steps),
        "valid_calls": calls - sim.errors,
        "calls": calls,
        "ended": transcript.ended,
        "reported": sim.reported_people(),
        "battery_at_end": sim.battery,
        "input_tokens": transcript.input_tokens,
        "output_tokens": transcript.output_tokens,
        "latency_s": round(transcript.latency_s, 3),
        "tools": transcript.tool_names(),
    }


def summarise(model_name: str, results: list[dict], price_in: float | None = None, price_out: float | None = None) -> dict:
    count = len(results)
    calls = sum(r["calls"] for r in results)
    tokens_in, tokens_out = sum(r["input_tokens"] for r in results), sum(r["output_tokens"] for r in results)
    summary = {
        "model": model_name,
        "missions": count,
        "succeeded": sum(r["success"] for r in results),
        "success_rate": round(sum(r["success"] for r in results) / count, 4) if count else 0.0,
        "valid_tool_call_rate": round(sum(r["valid_calls"] for r in results) / calls, 4) if calls else 0.0,
        "mean_steps": round(sum(r["steps"] for r in results) / count, 2) if count else 0.0,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "mean_latency_s": round(sum(r["latency_s"] for r in results) / count, 3) if count else 0.0,
        "cost_usd": None,
    }
    if price_in is not None and price_out is not None:
        # Prices are per million tokens, as given on the command line.
        summary["cost_usd"] = round(tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out, 4)
    return summary


def write_report(summaries: list[dict], results_by_model: dict, out_dir: Path = RESULTS) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mission agent evaluation",
        "",
        "Generated by `sardrone.agent.evaluate`. Do not edit by hand.",
        "",
        "Twenty written missions, run against a deterministic mission simulator",
        "(`sardrone/agent/sim.py`), not against the drone. A mission succeeds only",
        "if every check for it passes.",
        "",
        "| Model | Missions succeeded | Valid tool calls | Mean steps | Tokens in / out | Mean latency (s) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        cost = "not priced" if s["cost_usd"] is None else f"{s['cost_usd']:.4f}"
        lines.append(
            f"| {s['model']} | {s['succeeded']} of {s['missions']} ({s['success_rate']:.0%}) | "
            f"{s['valid_tool_call_rate']:.1%} | {s['mean_steps']} | {s['input_tokens']} / {s['output_tokens']} | "
            f"{s['mean_latency_s']} | {cost} |"
        )
    for model_name, results in results_by_model.items():
        lines += ["", f"## {model_name}", "", "| Mission | Result | Steps | What went wrong | Prompt |", "| --- | --- | --- | --- | --- |"]
        for r in results:
            wrong = ", ".join(name.replace("_", " ") for name in r["failed_checks"]) or ""
            lines.append(f"| {r['id']} | {'pass' if r['success'] else 'FAIL'} | {r['steps']} | {wrong} | {r['prompt']} |")
    (out_dir / "agent_eval.md").write_text("\n".join(lines) + "\n")
    (out_dir / "agent_eval.json").write_text(
        json.dumps({"summaries": summaries, "results": results_by_model}, indent=2) + "\n"
    )


def make_model(name: str, cohere_model: str | None = None) -> ChatModel:
    if name == "rule-based":
        return RuleBasedModel()
    if name == "cohere":
        from sardrone.agent.models import CohereModel

        return CohereModel(cohere_model)
    raise ValueError(f"unknown model {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the mission agent on the written missions.")
    parser.add_argument("--model", action="append", choices=["rule-based", "cohere"], help="may be given twice")
    parser.add_argument("--cohere-model", help="Cohere model name (or set COHERE_MODEL in .env)")
    parser.add_argument("--price-in", type=float, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens")
    args = parser.parse_args(argv)

    missions = load_missions()
    summaries, results_by_model = [], {}
    for name in args.model or ["rule-based"]:
        model = make_model(name, args.cohere_model)
        results = [run_mission(make_model(name, args.cohere_model) if name == "rule-based" else model, m) for m in missions]
        priced = (args.price_in, args.price_out) if name == "cohere" else (None, None)
        summaries.append(summarise(model.name, results, *priced))
        results_by_model[model.name] = results
        print(json.dumps(summaries[-1], indent=2))
        print("failed:", [(r["id"], r["failed_checks"]) for r in results if not r["success"]])
        if hasattr(model, "rate_limited"):
            summaries[-1]["model_calls"], summaries[-1]["rate_limited_retries"] = model.calls, model.rate_limited
            print(f"model calls {model.calls}, rate-limited and retried {model.rate_limited}")
    write_report(summaries, results_by_model)
    print(f"report: {RESULTS / 'agent_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
