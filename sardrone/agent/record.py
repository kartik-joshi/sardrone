"""Run one agent mission and keep its decisions as an episode.

    python -m sardrone.agent.record --mission m02 --model rule-based

Writes `data/episodes/agent-<mission>-<model>.mcap` with one `/events`
message per decision ("agent: start_search({"altitude_m": 25}) -> ok"),
plus the usual JSON sidecar holding the prompt, the transcript and the
score. The file opens in the same viewers as a flight, so an agent's
reasoning can be read on a timeline like any other mission.

The agent acts on the mission SIMULATOR here, not on the drone, so the
episode has events only: no camera, no pose. Event times are spaced one
second apart per tool call; they order the decisions and are not flight
time.
"""

import argparse
import json
import time
from pathlib import Path

from sardrone.agent.agent import run_agent
from sardrone.agent.evaluate import build_sim, check, load_missions, make_model
from sardrone.mcap.episode_writer import EVENT_AGENT, EpisodeMeta, EpisodeWriter
from sardrone.mcap.timeutil import NS_PER_SEC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "episodes"


def record_mission(model, mission: dict, path: Path, start_wall_ns: int | None = None, overwrite: bool = True):
    """Run `mission` with `model`; returns (episode writer, transcript, checks)."""
    start = time.time_ns() if start_wall_ns is None else start_wall_ns
    sim = build_sim(mission)
    meta = EpisodeMeta(world="mission simulator", weather="n/a", time_of_day="n/a", mission_prompt=mission["prompt"])
    extra = {"detector": "none", "agent_model": model.name, "mission_id": mission["id"]}
    with EpisodeWriter(path, meta, overwrite=overwrite, extra_sidecar=extra) as episode:
        step = 0

        def on_decision(text: str) -> None:
            nonlocal step
            episode.log_event(start + step * NS_PER_SEC, EVENT_AGENT, text)
            step += 1

        episode.log_event(start, EVENT_AGENT, f"mission: {mission['prompt']}")
        step = 1
        transcript = run_agent(model, mission["prompt"], sim, on_decision=on_decision)
        checks = check(mission, sim, transcript)
        episode.log_event(start + step * NS_PER_SEC, EVENT_AGENT, f"ended: {transcript.ended}")
        meta.people_placed = len(sim.scenario.people)
        meta.people_found = len(sim.reported)
        meta.success = all(checks.values())
        episode.extra_sidecar.update(
            {
                "agent_checks": checks,
                "agent_ended": transcript.ended,
                "agent_tools": transcript.tool_names(),
                "agent_tokens": {"input": transcript.input_tokens, "output": transcript.output_tokens},
            }
        )
    return episode, transcript, checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record one agent mission as an episode.")
    parser.add_argument("--mission", required=True, help="mission id from missions.yaml, e.g. m02")
    parser.add_argument("--model", default="rule-based", choices=["rule-based", "cohere"])
    parser.add_argument("--cohere-model")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    mission = next((m for m in load_missions() if m["id"] == args.mission), None)
    if mission is None:
        print(f"no mission {args.mission!r}")
        return 1
    model = make_model(args.model, args.cohere_model)
    path = args.out or DEFAULT_DIR / f"agent-{mission['id']}-{model.name.replace(':', '-')}.mcap"
    episode, transcript, checks = record_mission(model, mission, path)
    print(json.dumps({"mission": mission["id"], "model": model.name, "success": all(checks.values()),
                      "ended": transcript.ended, "steps": len(transcript.steps), "checks": checks}, indent=2))  # fmt: skip
    print(f"wrote {episode.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
