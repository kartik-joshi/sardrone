"""The agent loop.

    transcript = run_agent(model, "Search the north field ...", sim)

One turn of the loop: the model sees the whole conversation so far and
answers with a tool call; we run the tool on the simulator and append the
result; repeat. It ends when the drone is home, when the model stops
calling tools, or after `max_steps` calls (a guard against a model that
goes in circles).

Every decision is kept in the transcript: what was called, with what
arguments, what came back, how long the model took and how many tokens it
used. That record is what the evaluation scores.
"""

import json
import time
from dataclasses import dataclass, field

from sardrone.agent.models import ChatModel
from sardrone.agent.sim import MissionSim
from sardrone.agent.tools import SYSTEM_PROMPT, TOOLS, as_text, call_tool


@dataclass
class Step:
    tool: str
    arguments: dict
    result: dict
    battery_after: float
    state_after: str
    coverage_after: float
    latency_s: float


@dataclass
class Transcript:
    mission: str
    model: str
    steps: list = field(default_factory=list)
    ended: str = ""  # returned_home | crashed | model_stopped | max_steps
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    latency_s: float = 0.0

    def tool_names(self) -> list[str]:
        return [step.tool for step in self.steps]


def run_agent(model: ChatModel, mission: str, sim: MissionSim, max_steps: int = 40, on_decision=None) -> Transcript:
    """`on_decision(text)` is called for each tool call, e.g. to write the
    agent's decisions to an episode's /events channel."""
    transcript = Transcript(mission, model.name)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": mission}]
    nudged = False
    while len(transcript.steps) < max_steps:
        started = time.perf_counter()
        reply = model.reply(messages, TOOLS)
        elapsed = time.perf_counter() - started
        transcript.model_calls += 1
        transcript.latency_s += elapsed
        transcript.input_tokens += reply.input_tokens
        transcript.output_tokens += reply.output_tokens

        if not reply.tool_calls:
            if nudged:
                transcript.ended = "model_stopped"
                return transcript
            nudged = True  # one reminder, then give up
            messages.append({"role": "assistant", "content": reply.text or "(no tool call)"})
            messages.append({"role": "user", "content": "Continue the mission with a tool call, or call return_home."})
            continue

        messages.append(
            {
                "role": "assistant",
                "tool_plan": reply.text or "",
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in reply.tool_calls
                ],
            }
        )
        for call in reply.tool_calls:
            result = call_tool(sim, call.name, call.arguments)
            transcript.steps.append(
                Step(call.name, call.arguments, result, sim.battery, sim.state, sim.coverage, elapsed / len(reply.tool_calls))
            )
            if on_decision:
                on_decision(f"{call.name}({json.dumps(call.arguments)}) -> {'ok' if result.get('ok') else result.get('error')}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": as_text(result)})
        if sim.state == "HOME":
            transcript.ended = "returned_home"
            return transcript
        if sim.state == "CRASHED":
            transcript.ended = "crashed"
            return transcript
    transcript.ended = "max_steps"
    return transcript
